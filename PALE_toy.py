import torch
import numpy as np
from metrics import get_gt_matrix, get_statistics
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

class dataset(data.Dataset):

    def __init__(self, train_anchor):
        self.s_indices = train_anchor.T[0]
        self.t_indices = train_anchor.T[1]
        
    def __getitem__(self, index):
        return self.s_indices[index], self.t_indices[index]
        
    def __len__(self):
        return self.s_indices.shape[0]

class MappingModel(nn.Module):
    def __init__(self,embedding_dim):
        super().__init__()
        self.mlp=nn.Sequential(*[
        nn.Linear(embedding_dim, 2*embedding_dim, bias=True),
        nn.ReLU(),
        nn.Linear(2*embedding_dim, embedding_dim, bias=True),
        ])

    def forward(self, source_batch):
        x = self.mlp(source_batch)
        x = F.normalize(x, dim=1)
        return x

    def mapping_loss(self,source_batch_after_mapping,target_batch):
        fn_loss=nn.MSELoss()
        loss=fn_loss(source_batch_after_mapping, target_batch)
        return loss   
    
def mapping(s_embedding,t_embedding,train_anchor,bs=32,lr=0.01,epochs=100):
    embedding_dim = s_embedding.shape[1]
    model = MappingModel(embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    Data = dataset(train_anchor)
    Data = DataLoader(Data, batch_size=bs, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    s_embedding = s_embedding.to(device)
    t_embedding = t_embedding.to(device)
    model.train()
    for epoch in range(epochs):
        Loss = 0
        for data in Data:
            optimizer.zero_grad()
            s_indices, t_indices = data
            source_batch = s_embedding[s_indices]
            target_batch = t_embedding[t_indices]
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            s_trans = model(source_batch)
            loss = model.mapping_loss(s_trans, target_batch)
            loss.backward()
            optimizer.step()
            Loss += loss
        if epoch % 10 == 0:
            print('Epoch: {:03d}, Loss_train: {:.8f}'.format(epoch, Loss))
    model.eval()
    s_after_mapping=model(s_embedding).detach().cpu()
    return s_after_mapping

def parse_args():
    parser = argparse.ArgumentParser(description="PALE")
    parser.add_argument('--s_edge', default='./data/douban/online/raw/edgelist')
    parser.add_argument('--t_edge', default='./data/douban/offline/raw/edgelist')
    parser.add_argument('--train_path', default='./data/douban/anchor/node,split=0.8.train.dict')
    parser.add_argument('--gt_path', default='./data/douban/anchor/node,split=0.8.test.dict')
    parser.add_argument('--out_path', default='./data/douban/anchor/embeddings')
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=3000, type=int)
    parser.add_argument('--mbs', default=32, type=int)
    parser.add_argument('--mlr', default=0.01, type=float)
    parser.add_argument('--mepochs', default=100, type=int)
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

        s_after_mapping = mapping(s_embedding, t_embedding, train_anchor, bs=args.mbs, lr=args.mlr,epochs=args.mepochs)
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
        
    print('\nPALE')
    print(args)
    print('Average results:')
    for k, v in results.items():
        print('%s: %.4f' % (k, v))
    
