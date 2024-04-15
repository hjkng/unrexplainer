import os.path as osp
import os
import torch
from torch_geometric.data import DataLoader, Batch, Data
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import sort_edge_index
from explainer.tagexplainer import TAGExplainer, MLPExplainer

import explainer.args as args
import explainer.utils as ut

import pickle
import os.path as osp
import datetime
import numpy as np
import torch
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GraphSAGE, DeepGraphInfomax, SAGEConv
from torch_geometric.data import Dataset


args = args.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if args.dataset in ['Cora', 'CiteSeer']:
    path_nm = './dataset/' + str(args.dataset).lower() + '_'
    with open(path_nm + 'train_data' ,'rb') as fw:
        data = pickle.load(fw)
    G = to_networkx(data, to_undirected=True)
    G = ut.labelling_graph(G) 
    with open(path_nm + 'test_data' ,'rb') as fw:
        test_data = pickle.load(fw)

elif args.dataset == 'PubMed':
    path = './dataset'
    dataset = Planetoid(path, name=args.dataset, transform=T.NormalizeFeatures(), split='full')
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    G = ut.labelling_graph(G) 
else:
    dataset_pth = './dataset/'+ args.dataset+ '.pkl'
    with open(dataset_pth, 'rb') as fin:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix  = pickle.load(fin)

    x = torch.from_numpy(features).float()
    adj = torch.from_numpy(adj)
    edge_index = adj.nonzero().t().contiguous()

    y = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
    node_label = torch.from_numpy(np.where(y)[1])

    data = Data(x=x, edge_index=edge_index, y=node_label)
    data.train_mask = torch.Tensor(train_mask)
    data.val_mask = torch.Tensor(val_mask)
    data.test_mask = torch.Tensor(test_mask)
    G = to_networkx(data, to_undirected=True)    
    G = ut.labelling_graph(G) 
        

if args.model == 'graphsage':
    model = GraphSAGE(data.num_node_features, hidden_channels=args.hidden_dim, num_layers=2).to(device)

elif args.model == 'dgi':
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    model = DeepGraphInfomax(
        hidden_channels=512, encoder=GraphSAGE(data.x.shape[1], hidden_channels=512, num_layers=2),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)

    
path_nm = './model/'+str(args.dataset.lower())+'_'+str(args.model)+'_'+str(args.task)
model.load_state_dict(torch.load(path_nm))       
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

x, edge_index = data.x.to(device), data.edge_index.to(device)

if args.model == 'graphsage':
    with torch.no_grad():
        model.eval()
        z = model(x, edge_index)
elif args.model == 'dgi':      
    with torch.no_grad():
        model.eval()
        z, _, _ = model(x, edge_index)
    
z = z.to('cpu').detach().numpy()
bf_top5_idx, bf_dist = ut.emb_dist_rank(z, args.neighbors_cnt)

enc_explainer = TAGExplainer(model, embed_dim=z.shape[1], device=device, explain_graph=False, 
                              grad_scale=0.1, coff_size=0.05, coff_ent=0.002, loss_type='JSE')


enc_explainer.train_explainer_node(data.to(device), lr=0.001, epochs=100)

# enc_explainer.load_state_dict(torch.load('./result/pubmed_dgi_tage_explainer'))

expl = dict({'subgraph':[], 'importance':[],'size':[], 'idx':[], 
             'time':[]})

for nd in torch.tensor(list(G.nodes())):

    start_time = datetime.datetime.now()
    
    subgraph, pred_mask, masked_embed, subset, impt_data = enc_explainer(data, node_idx=nd, top_k=10)
    expl['time'].append((datetime.datetime.now()-start_time).seconds)
    expl['idx'].append(nd)
    
    try:
        e1, e2 = impt_data.edge_index
        newG = nx.Graph()
        newG.add_edges_from([(subset[e1[i]], subset[e2[i]])for i in range(0, len(e1))])
        expl['subgraph'].append(newG)

        edges_to_perturb =  list(newG.edges())
        importance = ut.importance(args, model, x, edge_index, bf_top5_idx, bf_dist, edges_to_perturb, nd)
        
        expl['importance'].append(importance)
        expl['size'].append(newG.number_of_edges())    
        
    except ValueError:
        newG = nx.Graph()
        expl['subgraph'].append(newG)
        expl['importance'].append(0)
        expl['size'].append(0)

    path_nm = './result/'+str(args.dataset.lower())+'_'+str(args.model)+'_'+str(args.task)+'_tage'
    f = open(path_nm,"wb")
    pickle.dump(expl,f)
    f.close() 
    
    print( expl['time'][-1], 'mins is consumed...')   
    print('node: ', int(nd), ' | generatedG: ', expl['importance'][-1])
     
print('----- total result-----')
print('TAGE: ', round(np.mean(expl['importance']),3))

