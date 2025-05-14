import torch
import numpy as np
from metrics import get_gt_matrix, get_statistics, top_k, compute_precision_k
from utils import get_adj, get_edgeindex, to_word2vec_format
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from model.GAE import get_embedding
from model.node2vec import node2vec
import argparse
from time import time


def autoencoder_loss(decoded, source_feats, inversed_decoded, target_feats):
    num_examples1 = source_feats.shape[0]
    num_examples2 = target_feats.shape[0]
    straight_loss = (num_examples1 - (decoded * source_feats).sum()) / num_examples1
    inversed_loss = (num_examples2 - (inversed_decoded * target_feats).sum()) / num_examples2
    loss = straight_loss + inversed_loss
    return loss


class MappingModel(nn.Module):
    def __init__(self, embedding_dim=800, hidden_dim1=1200, hidden_dim2=1600, source_embedding=None,
                 target_embedding=None):
        """
        Parameters
        ----------
        embedding_dim: int
            Embedding dim of nodes
        hidden_dim1: int
            Number of hidden neurons in the first hidden layer of MLP
        hidden_dim2: int
            Number of hidden neurons in the second hidden layer of MLP
        source_embedding: torch.Tensor or Embedding_model
            Used to get embedding vectors for source nodes
        target_embedding: torch.Tensor or Embedding_model
            Used to get embedding vectors for target nodes
        """

        super(MappingModel, self).__init__()
        self.source_embedding = F.normalize(source_embedding, dim=1)
        self.target_embedding = F.normalize(target_embedding, dim=1)

        # theta is a MLP nn (encoder)
        self.theta = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim1, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim2, embedding_dim, bias=True)
        ])
        # inversed_theta is a MLP nn (decoder)
        self.inversed_theta = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim1, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim2, embedding_dim, bias=True)
        ])

    def forward(self, source_feats, mode='d'):
        encoded = self.theta(source_feats)
        encoded = F.normalize(encoded, dim=1)
        if mode != 'd':
            return encoded
        decoded = self.inversed_theta(encoded)
        decoded = F.normalize(decoded, dim=1)
        return decoded

    def inversed_forward(self, target_feats):
        inversed_encoded = self.inversed_theta(target_feats)
        inversed_encoded = F.normalize(inversed_encoded, dim=1)
        inversed_decoded = self.theta(inversed_encoded)
        inversed_decoded = F.normalize(inversed_decoded, dim=1)
        return inversed_decoded

    def supervised_loss(self, source_batch, target_batch, alpha=0.8, k=5):
        source_feats = self.source_embedding[source_batch]
        target_feats = self.target_embedding[target_batch]

        source_after_map = self.forward(source_feats, mode='e')

        map_loss = 0
        reward_source_target = 0
        reward_target_source = 0

        for i in range(source_feats.shape[0]):
            embedding_of_ua = source_feats[i]  # u_a
            embedding_of_target_of_ua = target_feats[i]  # u'_a
            embedding_of_ua_after_map = source_after_map[i]  # theta(u_a)
            map_loss += 1 - torch.sum(embedding_of_ua_after_map * embedding_of_target_of_ua)
            # map_loss += -torch.log(torch.sum(embedding_of_ua_after_map * embedding_of_target_of_ua) + 1)
            top_k_simi = self.find_topk_simi(embedding_of_ua_after_map, self.target_embedding, k=k)
            reward_source_target += self.compute_rst(embedding_of_target_of_ua, top_k_simi)
            reward_target_source += self.compute_rts(embedding_of_ua, top_k_simi)

        map_loss = map_loss / source_feats.shape[0]
        st_loss = -alpha * reward_source_target / source_feats.shape[0]
        ts_loss = -(1 - alpha) * reward_target_source / target_feats.shape[0]
        loss = map_loss + st_loss + ts_loss
        return loss, map_loss, st_loss, ts_loss
        # return map_loss

    def unsupervised_loss(self, source_batch, target_batch):
        source_feats = self.source_embedding[source_batch]
        target_feats = self.target_embedding[target_batch]
        decoded = self.forward(source_feats)
        inversed_decoded = self.inversed_forward(target_feats)
        loss = autoencoder_loss(decoded, source_feats, inversed_decoded, target_feats)
        return loss

    def compute_rst(self, embedding_of_ua_after_map, top_k_simi):
        top_k_embedding = self.target_embedding[top_k_simi]
        cosin = torch.sum(embedding_of_ua_after_map * top_k_embedding, dim=1)
        reward = torch.mean(torch.log(cosin + 1))
        return reward

    def compute_rts(self, embedding_of_ua, top_k_simi):
        top_k_embedding = self.target_embedding[top_k_simi]
        top_k_simi_after_inversed_map = self.inversed_theta(top_k_embedding)
        top_k_simi_after_inversed_map = F.normalize(top_k_simi_after_inversed_map, dim=1)
        cosin = torch.sum(embedding_of_ua * top_k_simi_after_inversed_map, dim=1)
        reward = torch.mean(torch.log(cosin + 1))
        return reward

    def find_topk_simi(self, embedding_of_ua_after_map, target_embedding, k):
        cosin_simi_matrix = torch.matmul(embedding_of_ua_after_map, target_embedding.t())
        top_k_index = cosin_simi_matrix.sort()[1][-k:]
        return top_k_index


class dataset(data.Dataset):

    def __init__(self, train_anchor):
        self.s_indices = train_anchor.T[0]
        self.t_indices = train_anchor.T[1]

    def __getitem__(self, index):
        return self.s_indices[index], self.t_indices[index]

    def __len__(self):
        return self.s_indices.shape[0]


def mapping_train(s_embedding, t_embedding, train_anchor, full_anchor, gt, bs=32, us_lr=0.01, s_lr=0.01, \
    us_epochs=500, s_epochs=10000, alpha=0.8, k=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s_embedding = s_embedding.to(device)
    t_embedding = t_embedding.to(device)
    embedding_dim = s_embedding.shape[1]
    model = MappingModel(
        embedding_dim=embedding_dim,
        hidden_dim1=2 * embedding_dim,
        hidden_dim2=2 * embedding_dim,
        source_embedding=s_embedding,
        target_embedding=t_embedding
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=us_lr)
    Data = dataset(full_anchor)
    Data = DataLoader(Data, batch_size=bs, shuffle=True)
    model = model.to(device)
    for epoch in range(us_epochs):
        model.train()
        Loss = 0
        Map_loss = 0
        rst_loss = 0
        rts_loss = 0
        for data in Data:
            optimizer.zero_grad()
            source_batch, target_batch = data
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            
            loss = model.unsupervised_loss(source_batch, target_batch)
            
            loss.backward()
            optimizer.step()
            Loss += loss
        if epoch % 100 == 0:
            print('Epoch: {:03d}, Loss_train: {:.8f}'.format(epoch, Loss))
    print('unsupervised_mapping has done!')
    optimizer = torch.optim.SGD(model.parameters(), lr=s_lr)
    Data = dataset(train_anchor)
    Data = DataLoader(Data, batch_size=bs, shuffle=True)
    for epoch in range(s_epochs):
        model.train()
        Loss = 0
        Map_loss = 0
        rst_loss = 0
        rts_loss = 0
        for data in Data:
            optimizer.zero_grad()
            source_batch, target_batch = data
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            loss, map_loss, st_loss, ts_loss = model.supervised_loss(source_batch, target_batch, alpha=alpha, k=k)
            Map_loss += map_loss
            rst_loss += st_loss
            rts_loss = ts_loss
            loss.backward()
            optimizer.step()
            Loss += loss
        # if epoch % 100 == 0:
        #     p30 = evaluate(model, s_embedding, t_embedding, gt)
        #     print('Epoch: {:03d}, Loss_train: {:.8f}, Map_loss: {:.8f}, rst_loss: {:.8f}, rts_loss: {:.8f}, p30: {:.4f}'.format(
        #                                                                  epoch, Loss, Map_loss, rst_loss, rts_loss, p30))
    model.eval()
    print('supervised_mapping has done!')
    s_embedding = F.normalize(s_embedding, dim=1)
    s_after_mapping = model(s_embedding, 'val').detach().cpu()
    return s_after_mapping

@torch.no_grad()
def evaluate(model, z1, z2, gt):
    model.eval()
    z1_p = model(z1, 'val')
    v1 = z1_p.detach().cpu()
    v2 = z2.detach().cpu()
    S = cosine_similarity(v1, v2)
    pred_top_30 = top_k(S, 30)
    precision_30 = compute_precision_k(pred_top_30, gt)
    return precision_30

def parse_args():
    parser = argparse.ArgumentParser(description="DeepLink")
    parser.add_argument('--s_edge', default='./data/douban/online/raw/edgelist')
    parser.add_argument('--t_edge', default='./data/douban/offline/raw/edgelist')
    parser.add_argument('--train_path', default='./data/douban/anchor/node,split=0.8.train.dict')
    parser.add_argument('--gt_path', default='./data/douban/anchor/node,split=0.8.test.dict')
    parser.add_argument('--out_path', default='./data/douban/anchor/embeddings')
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=3000, type=int)
    parser.add_argument('--bs', default=32, type=int)
    parser.add_argument('--us_lr', default=0.001, type=float)
    parser.add_argument('--us_epochs', default=500, type=int)
    parser.add_argument('--s_lr', default=0.001, type=float)
    parser.add_argument('--s_epochs', default=10000, type=int)
    parser.add_argument('--alpha', default=0.8, type=float)
    parser.add_argument('--k', default=5, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    results = dict.fromkeys(('Acc', 'MRR', 'AUC', 'Hit', 'Precision@1', 'Precision@5', 'Precision@10', 'Precision@15', \
        'Precision@20', 'Precision@25', 'Precision@30', 'time'), 0) # save results
    N = 10 # repeat times for average, default: 1
    for i in range(N):
        start_time = time()
        args = parse_args()

        print('Load data...')
        # genetate adjacency matrix
        s_adj = get_adj(args.s_edge)
        t_adj = get_adj(args.t_edge)
        # generate edge_index(pyG version)
        s_e = get_edgeindex(args.s_edge)
        t_e = get_edgeindex(args.t_edge)
        s_num = s_adj.shape[0]
        t_num = t_adj.shape[0]
        # load train anchor links
        train_anchor = torch.LongTensor(np.loadtxt(args.train_path, dtype=int))
        test_anchor = torch.LongTensor(np.loadtxt(args.gt_path, dtype=int))
        full_anchor = torch.vstack((train_anchor, test_anchor))
        # generate test anchor matrix for evaluation
        groundtruth_matrix = get_gt_matrix(args.gt_path, (s_num, t_num))

        print('Generate deepwalk embeddings as input X...')
        s_x = node2vec(s_adj, P=1, Q=1, WINDOW_SIZE=10, NUM_WALKS=10, WALK_LENGTH=80, DIMENSIONS=128, \
            WORKERS=8, ITER=5, verbose=1)
        t_x = node2vec(t_adj, P=1, Q=1, WINDOW_SIZE=10, NUM_WALKS=10, WALK_LENGTH=80, DIMENSIONS=128, \
            WORKERS=8, ITER=5, verbose=1)
        s_x = torch.FloatTensor(s_x)
        t_x = torch.FloatTensor(t_x)
        time1 = time()
        t1 = time1 - start_time # the runtime of data processing (include DeepWalk)
        print('Finished in %.4f s!'%(t1))

        s_embedding = get_embedding(s_x, s_e, dim=args.dim, lr=args.lr, epochs=args.epochs)
        t_embedding = get_embedding(t_x, t_e, dim=args.dim, lr=args.lr, epochs=args.epochs)
        print('embedding stage has done!')

        s_after_mapping = mapping_train(s_embedding, t_embedding, train_anchor, full_anchor, groundtruth_matrix, \
                bs=args.bs, us_lr=args.us_lr, s_lr=args.s_lr, us_epochs=args.us_epochs, s_epochs=args.s_epochs, \
                alpha=args.alpha, k=args.k)
        t_embedding = t_embedding.detach().cpu()
        print('matching stage has done!')
        t2 = time() - time1 # the runtime of embedding
        print('Finished in %.4f s!'%(t2))
        
        # to_word2vec_format(s_embedding, args.out_path, 'source_emb')
        # to_word2vec_format(t_embedding, args.out_path, 'target_emb')
        # to_word2vec_format(s_after_mapping, args.out_path, 'source_after_map')

        print('Evaluating...')
        scores = cosine_similarity(s_after_mapping, t_embedding)
        result = get_statistics(scores, groundtruth_matrix)
        t3 = time() - start_time
        for k, v in result.items():
            print('%s: %.4f' % (k, v))
            results[k] += v

        results['time'] += t2
        print('Total runtime: %.4f s'%(t3))
    for k in results.keys():
        results[k] /= N
        
    print('\nDeepLink')
    print(args)
    print('Average results:')
    for k, v in results.items():
        print('%s: %.4f' % (k, v))
