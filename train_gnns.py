from sys import exit
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, silhouette_score
from torch_geometric.loader import LinkNeighborLoader, NeighborSampler, NeighborLoader
from torch_geometric.nn import GraphSAGE, DeepGraphInfomax, SAGEConv, GCNConv
import explainer.args as args

args = args.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########################
##### 1. load a dataset
#########################

if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    if args.task =='node':
        path = './dataset'
        dataset = Planetoid(path, name=args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        
    else:
        
        transform = T.Compose([
        T.NormalizeFeatures(),
        T.RandomLinkSplit(num_val=0, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False)])
        path = './dataset'
        dataset = Planetoid(path, name=args.dataset, transform=transform)
        data, _, test_data = dataset[0]
        
        # path_nm = './dataset/' + str(args.dataset).lower() + '_'
        # f = open(path_nm + 'train_data',"wb")
        # pickle.dump(data,f)
        # f.close()     

        # f = open(path_nm + 'test_data',"wb")
        # pickle.dump(test_data,f)
        # f.close() 
        
elif args.dataset in ['syn1', 'syn2', 'syn3', 'syn4']:
    
    dataset_pth = './dataset/'+ args.dataset+ '.pkl'
    with open(dataset_pth, 'rb') as fin:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix  = pickle.load(fin)

    x = torch.from_numpy(features).float()
    adj = torch.from_numpy(adj)
    edge_index = adj.nonzero().t().contiguous()

    y = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
    node_label = torch.from_numpy(np.where(y)[1])

    data = Data(x=x, edge_index=edge_index, y=node_label)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

else:
    print('Currently, the dataset is not implemented.')
    exit()

#########################
##### 2. load the model
#########################

if args.model == 'dgi':
    
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    model = DeepGraphInfomax(
        hidden_channels=512, encoder=GraphSAGE(data.x.shape[1], hidden_channels=512, num_layers=2),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)
else:
    train_loader = LinkNeighborLoader(data, batch_size=256, shuffle=True,
                                  neg_sampling_ratio=1.0,num_neighbors=[10, 10])

    model = GraphSAGE(data.x.shape[1], hidden_channels=args.hidden_dim, num_layers=2).to(device)
      
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#########################
##### 3. train the model
#########################

def train_dgi():
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary = model(x, edge_index)
    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
    optimizer.step()
    return loss.item()
    
def train_gs():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        h = model(batch.x, batch.edge_index)
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)

    return total_loss / data.num_nodes

if args.model == 'dgi':
    x, edge_index, y = data.x.to(device),  data.edge_index.to(device), data.y.to(device)
    for epoch in range(0, 100):
        loss = train_dgi()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
else:
    for epoch in range(0, 300):
        loss = train_gs()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

#########################
##### 4. test the model
#########################

def link_pred_auc(data, z):
    out_lk = (z[data.edge_label_index[0]] * z[data.edge_label_index[1]])
    X_train, X_test, y_train, y_test = train_test_split(
    out_lk , data.edge_label.cpu().numpy(), test_size=0.3, random_state=42, stratify= data.edge_label.cpu().numpy())
    clf = LogisticRegression(max_iter=1000) 
    clf.fit(X_train, y_train)
    try:
        return roc_auc_score(clf.predict(X_test), y_test)
    except ValueError:
        return 0

@torch.no_grad()
def test():

    X = out.cpu().detach().numpy()
    # node classification
    if args.dataset in ['syn1', 'syn2', 'syn3', 'syn4'] or args.task == 'node':
        X_train, X_test, y_train, y_test = train_test_split(X, data.y, test_size=0.2, random_state=42, shuffle=True)
        clf = LogisticRegression(random_state=42, max_iter=300).fit(X_train, y_train)
        score = clf.score(X_test, y_test) # accuracy
    
    # link prediction
    else:   
        score = link_pred_auc(test_data, out) # auc

    kmeans = KMeans(n_clusters=len(torch.unique(data.y)), random_state=0, n_init=10).fit(X)
    clst = kmeans.labels_
    silh = silhouette_score(X, clst)
    hs = homogeneity_score(data.y.tolist(), clst)

    return score, silh, hs

with torch.no_grad():
    model.eval()
    if args.model == 'dgi':
        out, _, _ = model(x, edge_index)
        out = out.cpu()
    else:
        out = model(data.x.to(device), data.edge_index.to(device)).cpu()
       
score, dbs, hs = test()    
print(f'score: {score:.4f},'
    f'silhouette_score: {dbs:.4f}, Homogeneity: {hs:.4f}') 
     
#########################
##### 5. save the model
#########################
        
def save_model():
    if args.task == 'link':
        model_nm = str(args.dataset).lower() + '_' + str(args.model) + '_link'
        model_path = './model/'+model_nm
        torch.save(model.state_dict(), model_path)
        np.save(model_path+'_emb',out)
    else:
        model_nm = str(args.dataset).lower() + '_' + str(args.model) + '_node'
        model_path = './model/'+model_nm
        torch.save(model.state_dict(), model_path) # model
        np.save(model_path+'_emb',out) # learned emb
        
save_model()