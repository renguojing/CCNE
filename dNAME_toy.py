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
from pale import dataset

def parse_args():
    parser = argparse.ArgumentParser(description="GANmatch")
    parser.add_argument('--s_edge', default='./data/douban/online/raw/edgelist')
    parser.add_argument('--t_edge', default='./data/douban/offline/raw/edgelist')
    parser.add_argument('--train_path', default='./data/douban/anchor/node,split=0.8.train.dict')
    parser.add_argument('--gt_path', default='./data/douban/anchor/node,split=0.8.test.dict')
    parser.add_argument('--out_path', default='./data/douban/anchor/embeddings')
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--lr_phi', default=0.0005, type=float)
    parser.add_argument('--lr_d', default=1e-05, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bs', default=32, type=int)
    return parser.parse_args()

class Phi_model(nn.Module):
    def __init__(self,embedding_dim):
        super().__init__()
        self.phi=nn.Sequential(*[
        nn.Linear(embedding_dim, 2*embedding_dim, bias=True),
        nn.ReLU(),
        # nn.BatchNorm1d(2*embedding_dim),
        nn.Linear(2*embedding_dim, embedding_dim, bias=True),
        ])

    def forward(self, source_batch):
        x = self.phi(source_batch)
        # x = F.normalize(x, dim=1)
        return x

class D_model(nn.Module):
    def __init__(self,embedding_dim):
        super().__init__()
        hid = int(embedding_dim/2)
        self.d=nn.Sequential(*[
        nn.Linear(embedding_dim, hid, bias=True),
        nn.ReLU(),
        nn.BatchNorm1d(hid),
        nn.Linear(hid, 1, bias=True),
        nn.Sigmoid()
        ])

    def forward(self, source_batch):
        x = self.d(source_batch)
        # x = F.normalize(x, dim=1)
        return x


def train(z1, z2, train_anchor, gt, bs=32, lr_phi=0.0005, lr_d=0.00001, epochs=300, lamda=2):
    dim = z1.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Phi = Phi_model(dim)
    D = D_model(dim)
    optimizer_Phi = torch.optim.Adam(Phi.parameters(),lr=lr_phi)
    optimizer_D = torch.optim.Adam(D.parameters(),lr=lr_d)

    Data = dataset(train_anchor)
    Data = DataLoader(Data, batch_size=bs, shuffle=True, drop_last=True)
    map_loss = nn.MSELoss()

    Phi = Phi.to(device)
    D = D.to(device)
    z1 = z1.to(device)
    z2 = z2.to(device)
    for epoch in range(epochs):
        Phi.train()
        D.train()
        M_Loss = 0
        D_Loss = 0
        G_Loss = 0
        for data in Data:
            s_indices, t_indices = data
            source_batch = z1[s_indices]
            target_batch = z2[t_indices]
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)

            s_trans = Phi(source_batch)
            pro_real = D(target_batch)
            pro_fake = D(s_trans)
            D_loss = -torch.mean(torch.log(pro_real)+torch.log(1-pro_fake))
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()
            D_Loss += D_loss

            s_trans = Phi(source_batch)
            pro_fake = D(s_trans)
            m_loss = map_loss(s_trans, target_batch)
            M_Loss += m_loss
            G_loss = -torch.mean(torch.log(pro_fake))
            G_Loss += G_loss
            G_loss += lamda * m_loss
            optimizer_Phi.zero_grad()
            G_loss.backward()
            optimizer_Phi.step()

        # if epoch % 10 == 0:
        #     p30 = evaluate(Phi, z1, z2, gt)
        #     print('Epoch: {:03d}, M_loss: {:.8f}, D_loss: {:.8f}, G_loss: {:.8f}, p30: {:.4f}'.format(epoch,\
        #         M_Loss, D_loss, G_loss, p30))
    D.eval()
    s_after_map = Phi(z1).detach().cpu()
    return s_after_map

@torch.no_grad()
def evaluate(Phi, z1, z2, gt):
    z1_p = Phi(z1)
    v1 = z1_p.detach().cpu()
    v2 = z2.detach().cpu()
    S = cosine_similarity(v1, v2)
    pred_top_30 = top_k(S, 30)
    precision_30 = compute_precision_k(pred_top_30, gt)
    return precision_30

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

        s_embedding = get_embedding(s_x, s_e, dim=args.dim, lr=0.001, epochs=3000)
        t_embedding = get_embedding(t_x, t_e, dim=args.dim, lr=0.001, epochs=3000)
        print('embedding stage has done!')

        s_after_mapping = train(s_embedding, t_embedding, train_anchor, groundtruth_matrix, bs=args.bs, \
            lr_phi=args.lr_phi, lr_d=args.lr_d, epochs=args.epochs, lamda=2)

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
        
    print('\nGANmatch')
    print(args)
    print('Average results:')
    for k, v in results.items():
        print('%s: %.4f' % (k, v))
