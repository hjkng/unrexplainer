from tqdm import tqdm
import pickle
import numpy as np
import networkx as nx
import copy
import torch
from torch_geometric.utils import to_networkx
from gensim.models import Word2Vec
import explainer.utils as ut
import explainer.args as args

args = args.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path_nm = str(args.dataset).lower()+'_'+str(args.model) +'_'+str(args.task)
f = open('./result/' + path_nm,"rb")
expl = pickle.load(f)

data, G = ut.load_dataset(args) 
if args.task == 'link':
    test_data = data[1]; data = data[0] 
model,z = ut.load_model(args, data, device)
x, edge_index = data.x.to(device), data.edge_index.to(device)        
bf_top5_idx, bf_dist = ut.emb_dist_rank(z, args.neighbors_cnt)

rw1 = {'idx': [], 'subgraph':[], 'importance':[], 'size':[]};  # number of nodes = 2 
rw2 = {'idx': [], 'subgraph':[], 'importance':[], 'size':[]};  # number of nodes = 3 
rw3 = {'idx': [], 'subgraph':[], 'importance':[], 'size':[]};  # knn graph 
rw4 = {'idx': [], 'subgraph':[], 'importance':[], 'size':[]};  # random walk based graph
rw5 = {'idx': [], 'subgraph':[], 'importance':[], 'size':[]};  # random walk with restart
  
node_lst = list(G.nodes)      
for idx, node in enumerate(tqdm(node_lst)):
    
    neighbors = [n for n in G.neighbors(node)]
    if len(neighbors) >= 2:
        nd_lst = np.random.choice(neighbors, 2, replace=False)    
        G_3N = nx.Graph()
        G_3N.add_edges_from([(nd_lst[0], node), (nd_lst[1], node)])
        
        # number of nodes = 3
        rw2['subgraph'].append(G_3N)
        rw2['idx'].append(node)
        rw2['size'].append(G_3N.number_of_edges())
        edges_to_perturb = list(G_3N.edges())
        rw2['importance'].append(ut.importance(args, model, x, edge_index, bf_top5_idx, bf_dist, edges_to_perturb, node))
        
        nd_lst = np.random.choice(nd_lst, 1, replace=False)    
        G_2N = nx.Graph()
        G_2N.add_edges_from([(nd_lst[0], node)])
        
        # number of nodes = 2
        rw1['subgraph'].append(G_2N)
        rw1['idx'].append(node)
        rw1['size'].append(G_2N.number_of_edges())
        edges_to_perturb = list(G_2N.edges())
        rw1['importance'].append(ut.importance(args, model, x, edge_index, bf_top5_idx, bf_dist, edges_to_perturb, node))
        
    elif len(neighbors) == 1:
        # number of nodes = 2 for all
        
        rw_hop = nx.Graph()
        rw_hop.add_edges_from([(neighbors[0], node)])
        
        rw2['subgraph'].append(rw_hop)
        rw2['idx'].append(node)
        rw2['size'].append(rw_hop.number_of_edges())
        edges_to_perturb = list(rw_hop.edges())
        rw2['importance'].append(ut.importance(args, model, x, edge_index, bf_top5_idx, bf_dist, edges_to_perturb, node))
        
        rw1['subgraph'].append(rw_hop)
        rw1['idx'].append(node)
        rw1['size'].append(rw_hop.number_of_edges())
        edges_to_perturb = list(rw_hop.edges())
        rw1['importance'].append(ut.importance(args, model, x, edge_index, bf_top5_idx, bf_dist, edges_to_perturb, node))
    
    else:
        rw_hop = nx.Graph()
        rw2['subgraph'].append(rw_hop)
        rw2['idx'].append(node)
        rw2['size'].append(0)
        rw2['importance'].append(0)
        
        rw1['subgraph'].append(rw_hop)
        rw1['idx'].append(node)
        rw1['size'].append(0)
        rw1['importance'].append(0)
        
        
    # knn graph
    rw_hop = ut.generate_rgs_all(G, node, bf_top5_idx)
    rw3['subgraph'].append(rw_hop)
    rw3['idx'].append(node)
    rw3['size'].append(rw_hop.number_of_edges())
    edges_to_perturb = list(rw_hop.edges())
    rw3['importance'].append(ut.importance(args, model, x, edge_index, bf_top5_idx, bf_dist, edges_to_perturb, node))
                
    num_nodes = expl['subgraph'][idx].number_of_nodes()
    num_edges = expl['subgraph'][idx].number_of_edges()
    # random walk based graph
    rw_hop = ut.generate_randomG(G, node, num_nodes, num_edges)
    rw4['subgraph'].append(rw_hop)
    rw4['idx'].append(node)
    rw4['size'].append(rw_hop.number_of_edges())
    edges_to_perturb = list(rw_hop.edges())
    rw4['importance'].append(ut.importance(args, model, x, edge_index, bf_top5_idx, bf_dist, edges_to_perturb, node))
         
    # random walk with restart
    rw_hop = ut.generate_rwr(G, node, num_edges)
    rw5['subgraph'].append(rw_hop)
    rw5['idx'].append(node)
    rw5['size'].append(rw_hop.number_of_edges())
    edges_to_perturb = list(rw_hop.edges())
    rw5['importance'].append(ut.importance(args, model, x, edge_index, bf_top5_idx, bf_dist, edges_to_perturb, node))
               
rws = {'n2':rw1, 'n3':rw2, 'knn':rw3, 'rw':rw4, 'rwr':rw5}

with open('./result/'+path_nm+'_baseline', 'wb') as f:
    pickle.dump(rws,f)