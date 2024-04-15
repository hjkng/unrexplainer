import math
import numpy as np
import copy
import networkx as nx
import explainer.utils as ut
import explainer.args as args
import datetime
import pickle
import networkx as nx
import numpy as np
import matplotlib.pylab as plt
import scipy.io
import tqdm
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GraphSAGE, DeepGraphInfomax, SAGEConv




######################################
### 1. Load the dataset and trained model
######################################


args = args.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data, G = ut.load_dataset(args)
model,z = ut.load_model(args, data, device)
edges_lst = list(G.edges())
x, edge_index = data.x.to(device), data.edge_index.to(device)    

path_nm = './result/taxonomy/' + str(args.dataset).lower() +str(args.model).lower() 
mat_file = scipy.io.loadmat(path_nm + '_clusters.mat')
isleaf = scipy.io.loadmat(path_nm +'_isleaf.mat')


bf_top5_idx, bf_dist = ut.emb_dist_rank(z, args.neighbors_cnt)


######################################
### 2. Extract subgraphs from clusters
######################################

clusters = []
for i in range(0, len(isleaf['is_leaf'][0])):
    if isleaf['is_leaf'][0][i] == 1:
        clusters.append((mat_file['clusters'][0][i][0]-1).tolist())
        
num_nodes = G.number_of_nodes()
num_clusters = len(clusters)

from collections import defaultdict
nd_dict = defaultdict(int)
for node in range(0, num_nodes):
    nd_dict[G.nodes()[node]['label']] = node

clusters_new = []
for i in range(0, num_clusters):
    clusters_new.append([nd_dict[nd] for nd in clusters[i]]) 
    
cluster_G = []

for i in range(0, num_clusters):
    tmp = nx.Graph()
    tmp.add_nodes_from(clusters_new[i])
    for edge in edges_lst:
        if (edge[0] in clusters_new[i]) & (edge[1] in clusters_new[i]):
            tmp.add_edge(edge[0], edge[1])
    cluster_G.append(tmp)
    
cls_loc = []
idx_info = []
for node in range(0, num_nodes):
    for i in range(0, len(clusters)):
        for j in clusters_new[i]:
            if j == node:
                cls_loc.append(i)
                idx_info.append(j)
                

######################################
### 3. Evaluation
######################################

expl = dict({'subgraph':[], 'importance':[], 'size':[], 'idx':[]})


for nd,idx in enumerate(cls_loc):
    if nd in cluster_G[idx]:
        subgraph = nx.ego_graph(cluster_G[idx], nd, radius=2)
        try:
            subgraph = nx.ego_graph(cluster_G[idx], nd, radius=2)
            edges_to_perturb = list(subgraph.edges())
            importance = ut.importance(args, model, x,  edge_index, bf_top5_idx, bf_dist, edges_to_perturb, nd)

            expl['subgraph'].append(subgraph)
            expl['importance'].append(importance)
            expl['size'].append(subgraph.number_of_edges())    
            expl['idx'].append(nd)

        except ValueError:
            subgraph = nx.Graph()
            expl['subgraph'].append(subgraph)
            expl['importance'].append(0)
            expl['size'].append(0)   
            expl['idx'].append(nd)
    else:
            subgraph = nx.Graph()
            expl['subgraph'].append(subgraph)
            expl['importance'].append(0)
            expl['size'].append(0)   
            expl['idx'].append(nd)
    
    path_nm = './result/'+str(args.dataset).lower()+'_'+str(args.model)+'_'+str(args.task)+'_tx'
    f = open(path_nm,"wb")
    pickle.dump(expl,f)
    f.close() 

    print('node: ', int(nd), ' | generatedG: ', expl['importance'][-1])
     
print('----- total result-----')
print('generatedG: ', round(np.mean(expl['importance']),3))
    
    
