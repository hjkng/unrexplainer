import pickle
import os.path as osp
import datetime
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from torch_geometric.nn import DeepGraphInfomax, SAGEConv
from gnn.GraphSAGE import GraphSAGE
import explainer.args as args
import explainer.utils as ut
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import explainer.utils as ut

args = args.parse_args()
device = torch.device('cuda:'+ str(args.gpu) if torch.cuda.is_available() else 'cpu')    
data, G = ut.load_dataset(args)
if args.task == 'link':
    test_data = data[1]; data = data[0]
    
x, edge_index = data.x.to(device), data.edge_index.to(device)
model,z = ut.load_model(args, data, device)
emb_info = ut.emb_dist_rank(z, args.neighbors_cnt)
    
def PN_topk_link(subgraph_list, bf_top5_idx):
    
    score = 0; num = 0
    new_score = 0; new_num = 0
    label_lst = test_data.edge_label.cpu().tolist()
    
    for i,label in enumerate(tqdm(label_lst)):
        
        nd1 = int(test_data.edge_label_index[0][i])
        nd2 = int(test_data.edge_label_index[1][i])

        if label == 1:
            if nd2 in bf_top5_idx[nd1]:
                score +=1; num +=1
            else:
                num +=1
            if nd1 in bf_top5_idx[nd2]:
                score +=1; num +=1
            else:
                num +=1
        else:
            if nd2 not in bf_top5_idx[nd1]:
                score +=1; num +=1
            else:
                num+=1
            if nd1 not in bf_top5_idx[nd2]:
                score +=1; num +=1
            else:
                num+=1               
        subgraph = subgraph_list[nd1]
        new_emb = ut.perturb_emb(args, model, x, edge_index, list(subgraph.edges()))
        new_bf_top5_idx, _ = ut.emb_dist_rank(new_emb, args.neighbors_cnt)

        if label == 1:
            if nd2 in new_bf_top5_idx[nd1]:
                new_score +=1; new_num +=1
            else:
                new_num +=1

        else:
            if nd2 not in new_bf_top5_idx[nd1]:
                new_score +=1; new_num +=1
            else:
                new_num+=1


        subgraph = subgraph_list[nd2]
        new_emb = ut.perturb_emb(args, model, x, edge_index, list(subgraph.edges()))
        new_bf_top5_idx, _ = ut.emb_dist_rank(new_emb, args.neighbors_cnt)

        if label_lst[i] == 1:
            if nd1 in new_bf_top5_idx[nd2]:
                new_score +=1; new_num +=1
            else:
                new_num +=1
        else:
            if nd1 not in new_bf_top5_idx[nd2]:
                new_score +=1; new_num +=1
            else:
                new_num+=1         

    return abs(score/num-new_score/new_num)


def PN_node(subgraph_list, z):
        
    PN_lst = []     
    X_train, X_test, y_train, y_test = train_test_split(
    z, data.y, test_size=0.2, random_state=42, stratify= data.y.cpu().numpy())
    clf = LogisticRegression(max_iter=1000) # node classifier
    clf.fit(X_train, y_train)  
    
    for idx, subgraph in enumerate(tqdm(subgraph_list)):

        new_z = ut.perturb_emb(args, model, x, edge_index, list(subgraph.edges()))
        prediction = int(clf.predict(z[idx].reshape(1,-1))[0])
        out_bf = clf.predict_proba(z[idx].reshape(1,-1))[0][prediction]
        out_af = clf.predict_proba(np.array(new_z[idx]).reshape(1,-1))[0][prediction]
        PN_lst.append(out_bf - out_af)

    return PN_lst

def evaluate_subgraph(args, result, z, nm):

    if args.task == 'link':
        score = PN_topk_link(result['subgraph'], z)
    else:
        score = PN_node(result['subgraph'], z)
        
    vld = np.mean(np.where(np.array(result['importance'])>=1, 1, 0))
    impt = np.mean(result['importance'])
    sz = np.mean(result['size'])
    
    print(f'{nm:<5s} - {vld:.3f} | {impt:.3f} | {score:.3f} |  {sz:.1f} | ')   
    return  vld, impt, score, sz    
    
path_nm = './result/'+str(args.dataset.lower())+'_'+str(args.model)+'_'+str(args.task)    

f = open(path_nm +'_baseline',"rb")
result = pickle.load(f)

if args.task == 'link':
    z = emb_info[0]

print(f'model -  vld  |  imp  |  PN  | size |')  
print('--------------------------------------')
r1 = evaluate_subgraph(args, result['n2'], z, 'n2')
r2 = evaluate_subgraph(args, result['n3'], z, 'n3')
r3 = evaluate_subgraph(args, result['knn'], z, 'knn')
r4 = evaluate_subgraph(args, result['rw'], z, 'rw')
r5 = evaluate_subgraph(args, result['rwr'], z, 'rwr')

f = open(path_nm +'_tx',"rb")
tx = pickle.load(f)
tx = evaluate_subgraph(args, tx, z,'tx')

f = open(path_nm +'_tage',"rb")
tage = pickle.load(f)
tage = evaluate_subgraph(args, tage, z, 'tage')

f = open(path_nm,"rb")
result = pickle.load(f)
r6 = evaluate_subgraph(args, result, z, 'unr')

# import pandas as pd
# pd.DataFrame({'model': ['n2', 'n3', 'knn', 'rw', 'rwr', 'unr'],
#             'vld':[r1[0], r2[0], r3[0], r4[0], r5[0], r6[0]], 
#             'imp':[r1[1], r2[1], r3[1], r4[1], r5[1], r6[1]], 
#             'PN':[r1[2], r2[2], r3[2], r4[2], r5[2], r6[2]], 
#             'size':[r1[3], r2[3], r3[3], r4[3], r5[3], r6[3]]}).to_csv('./result/'+data_nm+'_'+str(args.model)+'_'+str(args.task)+'_eval.csv', index=False)


import pandas as pd
pd.DataFrame({'model': ['n2', 'n3', 'knn', 'rw', 'rwr', 'tx', 'tage','unr'],
            'vld':[r1[0], r2[0], r3[0], r4[0], r5[0], tx[0], tage[0], r6[0]], 
            'imp':[r1[1], r2[1], r3[1], r4[1], r5[1], tx[1], tage[1], r6[1]], 
            'PN':[r1[2], r2[2], r3[2], r4[2], r5[2], tx[2], tage[2], r6[2]], 
            'size':[r1[3], r2[3], r3[3], r4[3], r5[3], tx[3], tage[3], r6[3]]}).to_csv('./result/'+data_nm+'_'+str(args.model)+'_'+str(args.task)+'_eval.csv', index=False)
