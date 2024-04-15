import pickle
import numpy as np
from itertools import combinations
import networkx as nx
import torch
import explainer.node2vec as nv
from scipy.spatial import distance
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GraphSAGE, DeepGraphInfomax, GCNConv

def load_dataset(args):
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        if args.task =='node':
            path = './dataset'
            dataset = Planetoid(path, name=args.dataset, transform=T.NormalizeFeatures())
            data = dataset[0]
            G = to_networkx(data, to_undirected=True)
            G = labelling_graph(G) 
            return data, G
            
        else:    
            path_nm = './dataset/' + str(args.dataset).lower() + '_'
            with open(path_nm + 'train_data' ,'rb') as fw:
                data = pickle.load(fw)
                G = to_networkx(data, to_undirected=True)
                G = labelling_graph(G)
            with open(path_nm + 'test_data' ,'rb') as fw:
                test_data = pickle.load(fw)  
            return (data,test_data), G
            
    elif args.dataset in ['syn1', 'syn2', 'syn3', 'syn4']:
        
        dataset_pth = './dataset/'+ args.dataset+ '.pkl'
        with open(dataset_pth, 'rb') as fin:
                adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix  = pickle.load(fin)
        x = torch.from_numpy(features).float()
        adj = torch.from_numpy(adj)
        edge_index = adj.nonzero().t().contiguous()
        y = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
        node_label = torch.from_numpy(np.where(y)[1])
        dataset = Data(x=x, edge_index=edge_index, y=node_label)
        G = to_networkx(dataset, to_undirected=True)
        G = labelling_graph(G) 
        return dataset, G
    else:
        print('Currently, the dataset is not implemented.')
        exit()    

def load_model(args, data, device):
    path_nm = './model/'+str(args.dataset.lower())+'_'+str(args.model) + '_'+ str(args.task)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    
    if args.model == 'graphsage':
        model = GraphSAGE(data.num_node_features, hidden_channels=args.hidden_dim, num_layers=2).to(device)
        model.load_state_dict(torch.load(path_nm))    
        with torch.no_grad():
            model.eval()
            z = model(x, edge_index)
    elif args.model == 'dgi':   
        
        # class Encoder(torch.nn.Module):
        #     def __init__(self, in_channels, hidden_channels):
        #         super().__init__()
        #         self.conv = GCNConv(in_channels, hidden_channels)
        #         self.prelu = torch.nn.PReLU(hidden_channels)

        #     def forward(self, x, edge_index, edge_weight= None):
        #         x = self.conv(x, edge_index, edge_weight)
        #         x = self.prelu(x)
        #         return x
        
        def corruption(x, edge_index, edge_weight=None):
            return x[torch.randperm(x.size(0))], edge_index

        model = DeepGraphInfomax(
                hidden_channels=512, encoder=GraphSAGE(data.x.shape[1], hidden_channels=512, num_layers=2),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)
    
    
        
        model.load_state_dict(torch.load(path_nm))    
        
        with torch.no_grad():
            model.eval()
            z = model.encoder(x, edge_index)
    
    z = z.to('cpu').detach().numpy()
    return model, z
        
def labelling_graph(graph):
    for idx, node in enumerate(graph.nodes()):
        graph.nodes[node]['label'] = idx    
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['weight'] = 1
    graph = graph.to_undirected()
    return graph

def emb_dist_rank(base_emb, neighbors_cnt):
    bf_dist = distance.cdist(base_emb, base_emb, 'euclidean')
    bf_dist_rank = bf_dist.argsort()[:,1:(1+neighbors_cnt)]
    return  bf_dist_rank, bf_dist

def cal_impt(bf_dist_rank, bf_dist, new_emb, neighbors_cnt, nd_idx):
    bf_top5_idx = bf_dist_rank[nd_idx]
    bf_dist = bf_dist[nd_idx][bf_top5_idx]
    if type(new_emb) == list:
        new_emb = np.array(new_emb)
    af_dist = distance.cdist(new_emb[nd_idx].reshape(1,-1), new_emb, 'euclidean')[0] 
    af_top5_idx = af_dist.argsort()[1:(neighbors_cnt+1)]
    importance = round(1-(len(np.intersect1d(bf_top5_idx, af_top5_idx))/neighbors_cnt), 5)
    return importance

def perturb_emb(args, model, x, edge_index, edges_to_perturb):
    
    edge_index_cpu = edge_index.cpu().tolist()
    edge_index_tuple = [tuple([edge_index_cpu[0][i], edge_index_cpu[1][i]]) for i in range(len(edge_index_cpu[0]))]
    perturb_idx_lst = [[edge_index_tuple.index((edge[0], edge[1])), edge_index_tuple.index((edge[1], edge[0]))] for edge in edges_to_perturb]
    perturb_idx_lst = sum(perturb_idx_lst, [])
    edge_index_perturbed = torch.LongTensor([[edge_index_cpu[i][j] for j in range(len(edge_index_cpu[i])) if j not in perturb_idx_lst] for i in range(len(edge_index_cpu))]).cuda('cuda:'+str(args.gpu))
    
    with torch.no_grad():
        model.eval()
        if args.model == 'graphsage':
            new_emb = model(x, edge_index_perturbed)
        elif args.model == 'dgi':
            new_emb = model.encoder(x, edge_index_perturbed)
    
    return new_emb.cpu()
    
def importance(args, model, x, edge_index, bf_top5_idx, bf_dist, edges_to_perturb, nd_idx):
    new_emb = perturb_emb(args, model, x, edge_index, edges_to_perturb)
    return cal_impt(bf_top5_idx, bf_dist, new_emb.to('cpu').detach().numpy(), args.neighbors_cnt, nd_idx)

def generate_randomG(nx_G, initial_node, num_nodes, num_edges):
    G = nv.Graph(nx_G, False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(3, num_edges*4) # number of walks, length
    walks_slt_idx = [i for i in range(len(walks)) if walks[i][0] == initial_node]
    walks_1 = walks[np.random.choice(walks_slt_idx)]
    random_graph = nx.Graph()
    random_graph.add_node(initial_node)

    len_walk = 0
    for i in range(len(walks_1)-1):
        random_graph.add_node(walks_1[(i+1)])
        random_graph.add_edge(walks_1[i], walks_1[(i+1)])
        if random_graph.number_of_nodes() == num_nodes:
            break
            
    if random_graph.number_of_edges() == num_edges:
        return random_graph
    elif random_graph.number_of_edges() > num_edges:
        while random_graph.number_of_edges() > num_edges:
            graph_degree = [degree[1] for degree in list(random_graph.degree)]
            graph_idx = [degree[0] for degree in list(random_graph.degree)]
            nd1 = graph_idx[np.argmax(graph_degree)]
            nd2 = [n for n in random_graph.neighbors(nd1) if len([m for m in random_graph.neighbors(n)]) >=2][0]
            random_graph.remove_edge(nd1, nd2)
            return random_graph
    elif random_graph.number_of_edges() < num_edges:
        comb = list(combinations(list(random_graph.nodes()), 2))
        candidate = [item for item in comb if (item not in list(random_graph.edges())) and (nx_G.has_edge(item[0], item[1])==True)]
        if len(candidate) <= num_edges-random_graph.number_of_edges():
            add_edge_idx = 0
            while len(candidate) < add_edge_idx:
                random_graph.add_edge(candidate[add_edge_idx][0], candidate[add_edge_idx][1])
                add_edge_idx += 1
            return random_graph
        else:
            add_edge_idx = 0
            while random_graph.number_of_edges() < num_edges:
                random_graph.add_edge(candidate[add_edge_idx][0], candidate[add_edge_idx][1])
                add_edge_idx += 1
            return random_graph
        
def generate_rwr(nx_G, initial_node, num_edges):
    rwr_g = nx.Graph()
    rwr_g.add_node(initial_node)
    cur_node = initial_node
    while rwr_g.number_of_edges() < num_edges:
        nx_node = np.random.choice([n for n in nx_G.neighbors(cur_node)])
        rwr_g.add_edge(cur_node, nx_node)
        if np.random.rand() < 0.2:
            cur_node = initial_node
        else:
            cur_node = nx_node    
    return rwr_g

def generate_rgs(nx_G, initial_node, bf_top5_idx):
    topk_nodes = bf_top5_idx[initial_node]
    topk_g = nx.Graph()
    for n in topk_nodes:
        if nx_G.has_edge(n, initial_node):
            topk_g.add_edge(n, initial_node)
    return topk_g

def generate_rgs_all(nx_G, initial_node, bf_top5_idx):
    topk_nodes = bf_top5_idx[initial_node].tolist()
    topk_nodes.append(initial_node)
    topk_g = nx.Graph()
    for n in topk_nodes:
        for t in topk_nodes:
            if n != t:
                if nx_G.has_edge(n, t):
                    topk_g.add_edge(n, t)
    return topk_g