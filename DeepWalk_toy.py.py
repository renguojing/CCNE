import numpy as np
import networkx as nx
from model.node2vec import node2vec
from metrics import get_gt_matrix, get_statistics
from sklearn.metrics.pairwise import cosine_similarity
from time import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="dwcross")
    parser.add_argument('--s_edge', default='./data/douban/online/raw/edgelist')
    parser.add_argument('--t_edge', default='./data/douban/offline/raw/edgelist')
    parser.add_argument('--gt_path', default='./data/douban/anchor/node,split=0.8.test.dict')
    parser.add_argument('--train_path', default='./data/douban/anchor/node,split=0.8.train.dict')
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    return parser.parse_args()

def expand_graph(g1, g2, seeds):
    '''
    g1, g2: edgelist, min node index is 0;
    seeds: for seed in seeds, seed[0] is in g1, seed[1] is in g2 
    '''
    n1 = np.max(g1) + 1
    n2 = np.max(g2) + 1
    g2 = g2 + n1
    seeds[:, 1] = seeds[:, 1] + n1
    edges = np.vstack((g1, g2))
    edges = np.vstack((edges, seeds))
    adj = np.zeros((n1 + n2, n1 + n2))
    for edge in edges:
        adj[edge[0], edge[1]] = 1
        adj[edge[1], edge[0]] = 1
    return adj

if __name__ == "__main__":
    results = dict.fromkeys(('Acc', 'MRR', 'AUC', 'Hit', 'Precision@1', 'Precision@5', 'Precision@10', 'Precision@15', \
        'Precision@20', 'Precision@25', 'Precision@30', 'time'), 0) # save results
    N = 10 # repeat times for average, default: 1
    for i in range(N):
        start_time = time()
        args = parse_args()

        print('Load data...')
        s_edges = np.loadtxt(args.s_edge, dtype=int)
        t_edges = np.loadtxt(args.t_edge, dtype=int)
        s_num = np.max(s_edges) + 1
        t_num = np.max(t_edges) + 1
        # load train anchor links
        train_anchor = np.loadtxt(args.train_path, dtype=int)
        # generate test anchor matrix for evaluation
        groundtruth_matrix = get_gt_matrix(args.gt_path, (s_num, t_num))

        adj_new = expand_graph(s_edges, t_edges, train_anchor)
        time1 = time()
        t1 = time1 - start_time
        print('Finished in %.4f s!'%(t1))

        print('Generate embeddings...')
        emb = node2vec(adj_new,P=1,Q=1,WINDOW_SIZE=10,NUM_WALKS=10,WALK_LENGTH=80,DIMENSIONS=args.dim,\
            WORKERS=8,ITER=5,verbose=args.verbose)
        s_embedding = emb[:s_num, :]
        t_embedding = emb[s_num:, :]
        t2 = time() - time1
        print('Finished in %.4f s!'%(t2))

        # to_word2vec_format(s_embedding, args.out_path, 'source_emb')
        # to_word2vec_format(t_embedding, args.out_path, 'target_emb')

        print('Evaluating...')
        scores = cosine_similarity(s_embedding, t_embedding)
        result = get_statistics(scores, groundtruth_matrix)
        t3 = time() - start_time
        for k, v in result.items():
            print('%s: %.4f' % (k, v))
            results[k] += v

        results['time'] += t2
        print('Total runtime: %.4f s'%(t3))
    for k in results.keys():
        results[k] /= N
        
    print('\ndwcross')
    print(args)
    print('Average results:')
    for k, v in results.items():
        print('%s: %.4f' % (k, v))