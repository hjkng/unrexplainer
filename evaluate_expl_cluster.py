import pickle
import os.path as osp
import datetime
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GraphSAGE, DeepGraphInfomax, SAGEConv
import explainer.args as args
import explainer.utils as ut
import explainer.unrexplainer as unr
from tqdm import tqdm
args = args.parse_args()

###################################
########### load model
###################################

data, G = ut.load_dataset(args)
model, z = ut.load_model(args, data, device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.expansion_num = max(int(round(np.mean([G.degree[i] for i in list(G.nodes())]))), 3)

x, edge_index = data.x.to(device), data.edge_index.to(device)    


z = z.to('cpu').detach().numpy()
bf_top5_idx, bf_dist = ut.emb_dist_rank(z, args.neighbors_cnt)


###################################
########### cal cluster
###################################

from sklearn.cluster import KMeans


cluster_num = 20


kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(z)
clusters = kmeans.predict(z)

match_lst = []
for i in range(0, len(z)):
    cluster = clusters[i]
    matchv = []
    for k in range(0, args.neighbors_cnt):
        if cluster == clusters[bf_top5_idx[i][k]]:
            matchv.append(1)
        else:
            matchv.append(0)
    match_lst.append(matchv)
    
ori = np.sum(match_lst, axis=1)/5
print(np.mean(ori), np.std(ori))

def fidelity_cluster(result, emb):

    fidelity_lst = []     
    subgraph_list = result['subgraph']
    
    for idx, subgraph in enumerate(tqdm(subgraph_list)):
        new_emb = ut.newemb_graphsage(args, model, x, edge_index, list(subgraph.edges()))
        kmeans_new = KMeans(n_clusters=cluster_num, random_state=0).fit(new_emb)
        clusters_new = kmeans_new.predict(new_emb)
        matchv_new = []
        for k in range(0, 5):
            if clusters_new[idx] == clusters_new[bf_top5_idx[idx][k]]:
                matchv_new.append(1)
            else:
                matchv_new.append(0)
        fidelity_lst.append(matchv_new)
    return np.sum(fidelity_lst, axis=1)/5


###################################
########### result
###################################    

    
f = open(path_nm +'_baseline',"rb")
result = pickle.load(f)

print(f'model -  vld  |  imp  |  PN  | size |')  
print('--------------------------------------')
n2 = fidelity_cluster(result['n2'], z)
r3 = fidelity_cluster(result['n3'], z)
knn = fidelity_cluster(result['knn'], z)
rw = fidelity_cluster(result['rw'], z)
rwr = fidelity_cluster(result['rwr'], z)

f = open(path_nm +'_tx',"rb")
tx = pickle.load(f)
tx = evaluate_subgraph_syn(tx, z)

f = open(path_nm +'_tage',"rb")
tage = pickle.load(f)
tage = evaluate_subgraph_syn(tage, z)

f = open(path_nm,"rb")
result = pickle.load(f)
unr = evaluate_subgraph(args, result, z, z)


fid_lst =[np.std(ori), np.std(n2), np.std(n3), np.std(knn),
          np.std(rw), np.std(rwr), np.std(tx), np.std(tage),
          np.std(unr)]

std_lst =[np.std(ori), np.std(n2), np.std(n3), np.std(knn),
          np.std(rw), np.std(rwr), np.std(tx), np.std(tage),
          np.std(unr)]


import pandas as pd
path_nm = str(args.dataset).lower()+'_'+str(args.model).lower()+'_'+str(args.task).lower()+'_clust.csv'

pd.DataFrame({'model':['ori','n2', 'n3', 'knn', 'rw', 'rwr', 'tx', 'tage','unr'],
              'fidelity':fid_lst,'std':std_lst}).to_csv(path_nm, index=False)


