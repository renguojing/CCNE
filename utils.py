import networkx as nx
import torch
import numpy as np

def get_edgeindex(edgepath):
    edgelist=np.loadtxt(edgepath)
    edge=edgelist.T
    edge_index=torch.LongTensor(edge)
    edge_index_u=torch.vstack((edge_index[1],edge_index[0]))
    edge_index=torch.hstack((edge_index,edge_index_u))
    return edge_index

def get_adj(edgepath):
    g = nx.read_edgelist(edgepath,nodetype=int)
    # print(list(nx.selfloop_edges(g)))
    # g.remove_edges_from(nx.selfloop_edges(g))
    adjacency = np.zeros((len(g.nodes()), len(g.nodes())))
    for src_id, trg_id in g.edges():
        adjacency[src_id, trg_id] = 1
        adjacency[trg_id, src_id] = 1
    return adjacency

# def get_adj1(edgepath):
#     g = nx.read_edgelist(edgepath,nodetype=int)
#     # print(list(nx.selfloop_edges(g)))
#     # g.remove_edges_from(nx.selfloop_edges(g))
#     nodes = np.arange(len(g.nodes()))
#     adjacency = nx.to_numpy_matrix(g,nodelist=nodes)
#     return adjacency
      
def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    embedding=np.zeros((node_num, size))
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size+1
        embedding[int(vec[0])] = [float(x) for x in vec[1:]]
    fin.close()
    return embedding