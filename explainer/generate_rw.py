from tqdm import tqdm
import pickle
import numpy as np
import networkx as nx
import copy
import torch
from torch_geometric.utils import to_networkx
from gensim.models import Word2Vec
from torch_geometric.nn import GraphSAGE
import explainer.dgi as dgi
from torch_geometric.nn import DeepGraphInfomax, SAGEConv
from torch_geometric.nn import GraphSAGE
import explainer.utils as ut
import explainer.args as args

args = args.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rw1 = [] # number of nodes = 2 
rw2 = [] # number of nodes = 3 
rw3 = [] # knn graph # number of nodes = 2
rw4 = [] # random walk based graph
rw5 = [] # random walk with restart

path_nm = str(args.dataset).lower()+'_'+str(args.model) +'_'+str(args.task)
f = open('./result/' + path_nm,"rb")
expl = pickle.load(f)

data, G = ut.load_dataset(args)  
model, z = ut.load_model(args, data, device)
bf_top5_idx, bf_dist = ut.emb_dist_rank(z, args.neighbors_cnt)

if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    node_lst = list(G.nodes)
else:
    label = data.y.numpy()
    node_lst = [i for i, v in enumerate(label) if v >0]
    
for idx, node in enumerate(tqdm(list(G.nodes))):

    rw_hop= nx.ego_graph(G, node, radius=1)
    if rw_hop.number_of_nodes() > 3:
        rw_hop = copy.deepcopy(rw_hop)
        while rw_hop.number_of_nodes() > 3:
            nd_lst = list(rw_hop.nodes())
            nd_lst.remove(node)
            rw_hop.remove_node(np.random.choice(nd_lst, 1)[0])  
        # number of nodes = 3
        rw2.append(rw_hop)
        
        rw_hop = copy.deepcopy(rw_hop)
        nd_lst = list(rw_hop.nodes())
        nd_lst.remove(node)
        rw_hop.remove_node(np.random.choice(nd_lst, 1)[0])  
        # number of nodes = 2
        rw1.append(rw_hop)
        
    elif rw_hop.number_of_nodes() == 3:
        # number of nodes = 3
        rw2.append(rw_hop)
        
        rw_hop = copy.deepcopy(rw_hop)
        nd_lst = list(rw_hop.nodes())
        nd_lst.remove(node)
        rw_hop.remove_node(np.random.choice(nd_lst, 1)[0])  
        # number of nodes = 2
        rw1.append(rw_hop)
        
    elif rw_hop.number_of_nodes() < 3:
        # number of nodes = 2
        rw2.append(rw_hop)
        # number of nodes = 3
        rw1.append(rw_hop)        

    
    # knn graph
    rw3.append(ut.generate_rgs_all(G, node, bf_top5_idx))
        
    num_nodes = expl['subgraph'][idx].number_of_nodes()
    num_edges = expl['subgraph'][idx].number_of_edges()
    # random walk based graph
    rw4.append(ut.generate_randomG(G, node, num_nodes, num_edges))
    # random walk with restart
    rw5.append(ut.generate_rwr(G, node, num_edges))
        
    rws = {'n2':rw1, 'n3':rw2, 'knn':rw3, 'rw':rw4, 'rwr':rw4,}

    with open('./result/'+path_nm+'_baseline', 'wb') as f:
        pickle.dump(rws,f)

        
       
        
        
        
        
        
        
        