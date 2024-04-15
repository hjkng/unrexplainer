import pickle
import numpy as np
import networkx as nx
import random
import explainer.args as args
from explainer.groundtruth_utils import *
from explainer.utils import load_dataset

args = args.parse_args()
data_nm = str(args.dataset).lower()
data, G = load_dataset(args)
label = data.y.numpy()
node_lst = [i for i, v in enumerate(data.y.numpy()) if v >0]

def evaluate_subgraph_syn(result, nm):
    precision=[]; recall=[]

    v1, v2 = evaluation(data_nm, G, label, result)
    precision.append(v1); recall.append(v2)

    impt = np.mean([result['importance'][i] for i in range(0, len(result['importance'])) if i in node_lst])
    prc = np.mean(precision)
    rcl = np.mean(recall)
    sz = np.mean([result['size'][i] for i in range(0, len(result['size'])) if i in node_lst])
    print(f'{nm:<5s} - {prc:.3f} | {rcl:.3f} | {impt:.3f} |  {sz:.1f} |')   
    return prc, rcl, impt, sz
    
    
path_nm = './result/'+data_nm+'_'+str(args.model)+'_'+str(args.task)  

f = open(path_nm +'_baseline',"rb")
result = pickle.load(f)

print(f'model -  prc  |  rcl  |  imp  | size |')  
print('--------------------------------------')
r1 = evaluate_subgraph_syn(result['n2'], 'n2')
r2 = evaluate_subgraph_syn(result['n3'], 'n3')
r3 = evaluate_subgraph_syn(result['knn'], 'knn')
r4 = evaluate_subgraph_syn(result['rw'], 'rw')
r5 = evaluate_subgraph_syn(result['rwr'], 'rwr')

f = open(path_nm +'_tx',"rb")
tx = pickle.load(f)
tx = evaluate_subgraph_syn(tx, 'tx')

f = open(path_nm +'_tage',"rb")
tage = pickle.load(f)
tage = evaluate_subgraph_syn(tage, 'tage')

f = open(path_nm,"rb")
result = pickle.load(f)
r6 = evaluate_subgraph_syn(result, 'unr')

# import pandas as pd
# pd.DataFrame({'model': ['n2', 'n3', 'knn', 'rw', 'rwr', 'unr'],
#             'prc':[r1[0], r2[0], r3[0], r4[0], r5[0], r6[0]], 
#             'rcl':[r1[1], r2[1], r3[1], r4[1], r5[1], r6[1]], 
#             'imp':[r1[2], r2[2], r3[2], r4[2], r5[2], r6[2]], 
#             'size':[r1[3], r2[3], r3[3], r4[3], r5[3], r6[3]]}).to_csv('./result/'+data_nm+'_'+str(args.model)+'_'+str(args.task)+'_eval.csv', index=False)
import pandas as pd
pd.DataFrame({'model': ['n2', 'n3', 'knn', 'rw', 'rwr', 'tx', 'tage','unr'],
            'prc':[r1[0], r2[0], r3[0], r4[0], r5[0], tx[0], tage[0], r6[0]], 
            'rcl':[r1[1], r2[1], r3[1], r4[1], r5[1], tx[1], tage[1], r6[1]], 
            'imp':[r1[2], r2[2], r3[2], r4[2], r5[2], tx[2], tage[2], r6[2]], 
            'size':[r1[3], r2[3], r3[3], r4[3], r5[3], tx[3], tage[3], r6[3]]}).to_csv('./result/'+data_nm+'_'+str(args.model)+'_'+str(args.task)+'_eval.csv', index=False)
