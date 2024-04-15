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

args = args.parse_args()
device = torch.device('cuda:'+ str(args.gpu)if torch.cuda.is_available() else 'cpu')
       
data, G = ut.load_dataset(args)  
if args.task == 'link':
    test_data = data[1]; data = data[0]
model,z = ut.load_model(args, data, device)

args.expansion_num = max(int(round(np.mean([G.degree[i] for i in list(G.nodes())]))), 3)
emb_info = ut.emb_dist_rank(z, args.neighbors_cnt)

path_nm = str(args.path)+str(args.dataset.lower())+'_'+str(args.model)+'_'+str(args.task)
expl = dict({'idx':[], 'subgraph':[], 'importance':[], 'size':[], 'time':[]})     

node_lst = list(G.nodes)                   
for node in node_lst:
    
    start_time = datetime.datetime.now()
    subgraph, importance = unr.explainer(args, model, G, data, emb_info, node, device)

    expl['idx'].append(node)
    expl['time'].append((datetime.datetime.now()-start_time).seconds)
    expl['subgraph'].append(subgraph)
    expl['importance'].append(importance)
    expl['size'].append(subgraph.number_of_edges())
    
    f = open(path_nm, "wb")
    pickle.dump(expl,f)
    f.close() 

    print(f'Explanation of node {node} is generated with impt as {importance}')