import torch
import numpy as np
from metrics import get_gt_matrix, get_statistics
from utils import get_adj, get_edgeindex
from model.node2vec import node2vec
from CCNE import CCNE, sample
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from time import time
import argparse
import networkx as nx

def parse_args():
    parser = argparse.ArgumentParser(description="ICCNE")
    parser.add_argument('--gt_path', default='./data/douban/anchor/node,split=0.2.test.dict')
    parser.add_argument('--train_path', default='./data/douban/anchor/node,split=0.2.train.dict')
    parser.add_argument('--out_path', default='./data/douban/anchor/embeddings')
    parser.add_argument('--s_edge', default='./data/douban/online/raw/edgelist')
    parser.add_argument('--t_edge', default='./data/douban/offline/raw/edgelist')
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--lamda', default=1, type=float)
    parser.add_argument('--margin', default=0.9, type=float)
    parser.add_argument('--neg', default=1, type=int)
    parser.add_argument('--lr1', default=0.0001, type=float)
    parser.add_argument('--epochs1', default=60, type=int)
    parser.add_argument('--n_add_anchors', default=10, type=int)
    return parser.parse_args()

def get_embedding(s_x, t_x, s_e, t_e, g_s, g_t,anchor, gt, dim=64, lr=0.01, lamda=1, margin=0, neg=1, epochs=100, paras=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s_x = s_x.to(device)
    t_x = t_x.to(device)
    s_e = s_e.to(device)
    t_e = t_e.to(device)
    s_input = s_x.shape[1]
    t_input = t_x.shape[1]
    model = CCNE(s_input, t_input, dim)
    model = model.to(device)
    if paras:
        model.load_state_dict(paras)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cosine_loss=nn.CosineEmbeddingLoss(margin=margin)
    in_a, in_b, anchor_label = sample(anchor, g_s, g_t, neg=neg)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # in_a, in_b, anchor_label = sample(anchor, g_s, g_t, neg=neg)

        zs = model.s_forward(s_x, s_e)
        zt = model.t_forward(t_x, t_e)

        intra_loss = model.intra_loss(zs, zt, s_e, t_e)
        anchor_label = anchor_label.view(-1).to(device)
        inter_loss = cosine_loss(zs[in_a], zt[in_b], anchor_label)
        loss = intra_loss + lamda * inter_loss
        loss.backward()
        optimizer.step()
        # if epoch % 100 == 0:
        #     p10 = evaluate(vs, vt, gt)
        #     print('Epoch: {:03d}, recon_loss: {:.8f}, cross_loss: {:.8f}, loss_train: {:.8f}, precision_10: {:.8f}'.format(epoch,\
        #         recon_loss, cross_loss, loss, p10))
       
    paras = model.state_dict()
    model.eval()
    s_embedding = model.s_forward(s_x, s_e)
    t_embedding = model.t_forward(t_x, t_e)
    s_embedding = s_embedding.detach().cpu()
    t_embedding = t_embedding.detach().cpu()
    return s_embedding, t_embedding, paras

def greedy_match(s_emb, t_emb, seeds, min_size):
    s_nodes = np.arange(s_emb.shape[0])
    s_unmatched = np.delete(s_nodes, seeds.T[0])
    
    t_nodes = np.arange(t_emb.shape[0])
    t_unmatched = np.delete(t_nodes, seeds.T[1])
    
    S = cosine_similarity(s_emb[s_unmatched], t_emb[t_unmatched])
    S = S.T
    m, n = S.shape
    x = S.T.flatten()
    used_rows = np.zeros((m))
    used_cols = np.zeros((n))
    max_list = np.zeros((min_size))
    row = np.zeros((min_size))  # target indexes
    col = np.zeros((min_size))  # source indexes
    ix = np.argsort(-x) + 1
    matched = 1
    index = 1
    while(matched <= min_size):
        ipos = ix[index-1]
        jc = int(np.ceil(ipos/m))
        ic = ipos - (jc-1)*m
        if ic == 0: ic = 1
        if (used_rows[ic-1] == 0 and used_cols[jc-1] == 0):
            row[matched-1] = ic - 1
            col[matched-1] = jc - 1
            max_list[matched-1] = x[index-1]
            used_rows[ic-1] = 1
            used_cols[jc-1] = 1
            matched += 1
        index += 1
    for i in range(len(row)):
        seeds = np.vstack((seeds,[s_unmatched[int(col[i])], t_unmatched[int(row[i])]]))
    return seeds

if __name__ == "__main__":
    results = dict.fromkeys(('Acc', 'MRR', 'AUC', 'Hit', 'Precision@1', 'Precision@5', 'Precision@10', 'Precision@15', \
        'Precision@20', 'Precision@25', 'Precision@30', 'time'), 0)
    N = 1 # repeat times for average, default: 1
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

        # generate graph for negative sampling
        s_edge = np.loadtxt(args.s_edge, dtype=int)
        t_edge = np.loadtxt(args.t_edge, dtype=int)
        g_s = nx.Graph()
        g_s.add_edges_from(s_edge)
        g_t = nx.Graph()
        g_t.add_edges_from(t_edge)

        print('Generate deepwalk embeddings as input X...')
        s_x = node2vec(s_adj, P=1, Q=1, WINDOW_SIZE=10, NUM_WALKS=10, WALK_LENGTH=80, DIMENSIONS=128, \
            WORKERS=8, ITER=5, verbose=1)
        t_x = node2vec(t_adj, P=1, Q=1, WINDOW_SIZE=10, NUM_WALKS=10, WALK_LENGTH=80, DIMENSIONS=128, \
            WORKERS=8, ITER=5, verbose=1)
        s_x = torch.FloatTensor(s_x)
        t_x = torch.FloatTensor(t_x)
        time1 = time()
        t1 = time1 - start_time
        print('Finished in %.4f s!'%(t1))

        # print('Expand edges...')
        # g_s, g_t, s_e, t_e = expand_edges(g_s, g_t, train_anchor, s_e, t_e)

        print('Generate embeddings...')
        seeds = train_anchor
        n_seeds = seeds.shape[0]
        k = args.n_add_anchors
        first = True
        iter = 0
        max_size = min((s_num, t_num))
        while n_seeds < max_size:
            if first:
                s_embedding, t_embedding, paras = get_embedding(s_x, t_x, s_e, t_e, g_s, g_t, seeds, groundtruth_matrix,\
                    dim=args.dim, lr=args.lr, epochs=args.epochs, lamda=args.lamda, margin=args.margin, neg=args.neg)
                # print(paras)
                first = False
            else:
                s_embedding, t_embedding, paras = get_embedding(s_x, t_x, s_e, t_e, g_s, g_t, seeds, groundtruth_matrix,\
                    dim=args.dim, lr=args.lr1, epochs=args.epochs1, lamda=args.lamda, margin=args.margin, neg=args.neg, paras=paras)
            if n_seeds + k > max_size:
                k = max_size - n_seeds
            seeds = greedy_match(s_embedding, t_embedding, seeds, k)
            n_seeds += k
            iter += 1
            # evaluate
            print('Iter: %d'%(iter))
            S = cosine_similarity(s_embedding, t_embedding)
            result = get_statistics(S, groundtruth_matrix)
            print(result)
        
        t2 = time() - time1
        print('Finished in %.4f s!'%(t2))

        # to_word2vec_format(s_embedding, args.out_path, 'source_emb')
        # to_word2vec_format(t_embedding, args.out_path, 'target_emb')

        print('Evaluating...')
        S = cosine_similarity(s_embedding, t_embedding)
        result = get_statistics(S, groundtruth_matrix)
        t3 = time() - start_time
        for k, v in result.items():
            print('%s: %.4f' % (k, v))
            results[k] += v

        results['time'] += t3
        print('Total runtime: %.4f s'%(t3))
    for k in results.keys():
        results[k] /= N
        
    print('\nICCNE')
    print(args)
    print('Average results:')
    for k, v in results.items():
        print('%s: %.4f' % (k, v))