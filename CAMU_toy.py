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
    parser = argparse.ArgumentParser(description="CAMU")
    parser.add_argument('--s_edge', default='./data/douban/online/raw/edgelist')
    parser.add_argument('--t_edge', default='./data/douban/offline/raw/edgelist')
    parser.add_argument('--train_path', default='./data/douban/anchor/node,split=0.8.train.dict')
    parser.add_argument('--gt_path', default='./data/douban/anchor/node,split=0.8.test.dict')
    parser.add_argument('--out_path', default='./data/douban/anchor/embeddings')
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--lr_g', default=0.001, type=float)
    parser.add_argument('--lr_d', default=0.0002, type=float)
    parser.add_argument('--pre_epochs', default=50, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--d_step', default=5, type=int)
    parser.add_argument('--g_step', default=1, type=int)
    parser.add_argument('--m', default=10, type=float)
    parser.add_argument('--lamda', default=2, type=float)
    return parser.parse_args()

# def sample(us, ut, bs):
#     s_num = us.shape[0]
#     s_nodes = np.arange(s_num, dtype=int)
#     s_index = np.random.choice(s_nodes, size=bs, replace=False)

#     t_num = ut.shape[0]
#     t_nodes = np.arange(t_num, dtype=int)
#     t_index = np.random.choice(t_nodes, size=bs, replace=False)
#     return us[s_index], ut[t_index]

class Gst_model(nn.Module):
    def __init__(self,embedding_dim=800, hidden_dim1=1200, hidden_dim2=1600):
        super().__init__()
        self.gst = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim1, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim2, embedding_dim, bias=True)
        ])

    def forward(self, us):
        x = self.gst(us)
        x = F.normalize(x, dim=1)
        return x

class Gts_model(nn.Module):
    def __init__(self,embedding_dim=800, hidden_dim1=1200, hidden_dim2=1600):
        super().__init__()
        self.gts = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim1, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim2, embedding_dim, bias=True)
        ])

    def forward(self, ut):
        x = self.gts(ut)
        x = F.normalize(x, dim=1)
        return x

class Ds_model(nn.Module):
    def __init__(self,embedding_dim=800, hidden_dim=400):
        super().__init__()
        self.encoder = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim, bias=True)
        ])

        self.decoder = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim, bias=True)
        ])

    def forward(self, u):
        x = self.encoder(u)
        # x = F.normalize(x, dim=1)
        x = self.decoder(x)
        x = F.normalize(x, dim=1)
        dist = torch.norm(x - u, p=2, dim=1).mean()
        return dist

class Dt_model(nn.Module):
    def __init__(self,embedding_dim=800, hidden_dim=400):
        super().__init__()
        self.encoder = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim, bias=True)
        ])

        self.decoder = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim, bias=True)
        ])

    def forward(self, u):
        x = self.encoder(u)
        # x = F.normalize(x, dim=1)
        x = self.decoder(x)
        x = F.normalize(x, dim=1)
        dist = torch.norm(x - u, p=2, dim=1).mean()
        return dist

def train(us, ut, train_anchor, gt, dim=128, bs=128, lr_g=0.001, lr_d=0.0002, epochs=300, D_step=10, G_step=1, m=1, lamda=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gst = Gst_model(embedding_dim=dim, hidden_dim1=200, hidden_dim2=256)
    gts = Gts_model(embedding_dim=dim, hidden_dim1=200, hidden_dim2=256)
    ds = Ds_model(embedding_dim=dim, hidden_dim=64)
    dt = Dt_model(embedding_dim=dim, hidden_dim=64)
    optimizer_g = torch.optim.Adam([{'params': gst.parameters(),'lr': lr_g}, {'params': gts.parameters()}])
    optimizer_d = torch.optim.Adam([{'params': ds.parameters(),'lr': lr_d}, {'params': dt.parameters()}])
    gst = gst.to(device)
    gts = gts.to(device)
    ds = ds.to(device)
    dt = dt.to(device)
    us = us.to(device)
    ut = ut.to(device)
    Data = dataset(train_anchor)
    Data = DataLoader(Data, batch_size=bs, shuffle=True)
    #pretrain
    for epoch in range(args.pre_epochs):
        S_Loss = 0
        for data in Data:
            i, j = data
            us_batch = us[i]
            ut_batch = ut[j]
            us_batch = us_batch.to(device)
            ut_batch = ut_batch.to(device)
            us1_batch = gst(us_batch)
            ut1_batch = gts(ut_batch)
            s_loss = torch.norm(us1_batch - ut_batch, p=1, dim=1).mean() + torch.norm(ut1_batch - us_batch, p=1, dim=1).mean()
            optimizer_g.zero_grad()
            s_loss.backward()
            optimizer_g.step()
            S_Loss += s_loss
        # if epoch % 10 == 0:
        #     p30 = evaluate(gst, us, ut, gt)
        #     print('Epoch: {:03d}, S_loss: {:.8f}, p30: {:.4f}'.format(epoch, S_Loss, p30))

    for epoch in range(epochs):
        for step in range(D_step):   
            dt_loss = dt(ut) + max(0, m - dt(gst(us)))
            # print(dt(ut_batch), dt(gst(us_batch)))
            ds_loss = ds(us) + max(0, m - ds(gts(ut)))
            d_loss = dt_loss + ds_loss
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

        for step in range(G_step):
            ganst_loss = dt(gst(us))
            gants_loss = ds(gts(ut))
            cyc_loss = torch.norm(gts(gst(us))-us, p=1, dim=1).mean() + torch.norm(gst(gts(ut))-ut, p=1,dim=1).mean()
            g_loss = ganst_loss + gants_loss + lamda * cyc_loss
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

        # if epoch % 10 == 0:
        #     p30 = evaluate(gst, us, ut, gt)
        #     print('Epoch: {:03d}, D_loss: {:.8f}, G_loss: {:.8f}, p30: {:.4f}'.format(epoch, d_loss, g_loss, p30))

    us1 = gst(us).detach().cpu()
    # ut1 = gts(ut).detach().cpu()
    return us1  #, ut1

@torch.no_grad()
def evaluate(gst, z1, z2, gt):
    z1_p = gst(z1)
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

        s_after_mapping = train(s_embedding, t_embedding, train_anchor, groundtruth_matrix, dim=args.dim, bs=args.bs, \
            lr_g=args.lr_g, lr_d=args.lr_d, epochs=args.epochs, D_step=args.d_step, G_step=args.g_step, m=args.m, lamda=args.lamda)
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
        
    print('\nCAMU')
    print(args)
    print('Average results:')
    for k, v in results.items():
        print('%s: %.4f' % (k, v))
