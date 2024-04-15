import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, GraphConv
import pandas as pd
import networkx as nx
from collections import deque
import numpy as np

def plot_nhop(graph, node= 0, hop = 3,label=None):
    hop_graph = nx.Graph()
    colerdict = {0:'black', 1:'blue', 2:'orange',3:'red',4:'green', 5:'yellow', 6:'purple', 7:'grey'}
    neighbor_nodes = set([node])
    for i in range(hop):
        hop_neighbor = set()
        for j in neighbor_nodes:
            hop_neighbor = hop_neighbor|set(graph.neighbors(j))
        neighbor_nodes = neighbor_nodes|hop_neighbor
    neighbor_nodes = list(neighbor_nodes)
    len_neighbor = len(neighbor_nodes)
    for i in range(0, len_neighbor-1):
        for j in range(i, len_neighbor):
            node1 = neighbor_nodes[i]
            node2 = neighbor_nodes[j]
            if graph.has_edge(node1, node2):
                hop_graph.add_edge(node1, node2)
    color_map = [colerdict[label[i].item()] for i in hop_graph.nodes]

    nx.draw(hop_graph, pos=nx.kamada_kawai_layout(hop_graph), node_color = color_map, with_labels=True, node_size=100)
    plt.title(node)
    plt.show()

def find_baco_gt(graph, node, label):
    if label[node] == 0 or label[node] == 4:
        return
    if label[node] in [5, 7, 6]:
        ground_dict = {5:0, 6:0, 7:1}
    else:
        ground_dict = {1:0, 2:0, 3:1}
    neighbor_nodes = deque([node])
    ground_nodes = []
    while True:
        now_node = neighbor_nodes.popleft()

        now_label = label[now_node].item()

        if now_label in ground_dict.keys():
            if ground_dict[now_label] < 2:
                ground_nodes.append(now_node)
                ground_dict[now_label] += 1
                neighbor_nodes += graph.neighbors(now_node)
            if sum(ground_dict.values()) == 6:
                return ground_nodes

def find_cycle(graph, node, label):
    if label[node] == 0 or label[node] == 4:
        return
    if label[node] in [5, 7, 6]:
        ground_dict = {5:0, 6:0, 7:1}
    else:
        ground_dict = {1:0, 2:0, 3:1}
    neighbor_nodes = deque([node])
    ground_nodes = []
    exist_nodes = []
    while neighbor_nodes:
        now_node = neighbor_nodes.popleft()
        if abs(now_node - node) >10:
            continue
        if now_node in exist_nodes:
            continue
        exist_nodes.append(now_node)
        now_label = label[now_node].item()
        if now_label != 0:
            ground_nodes.append(now_node)
            neighbor_nodes += graph.neighbors(now_node)
    return ground_nodes


def f1(a, b):
    return 2/(1/a+1/b)

def evaluation(data_type, graph, label, expl):
    
    node_idx = 0
    total = 0
    precisoin = 0
    acc = 0
    recall = 0
    
    for i, l in enumerate(label):
        answer = 0
        answer2 = 0
        if l == 0 or node_idx>=229:
            continue

        sub_graph = expl['subgraph'][i]
        node_idx += 1

        if data_type == 'syn3' or data_type == 'syn4' or data_type == 'syn5':
            ground_nodes = find_cycle(graph, i, label)
        else:
            ground_nodes = find_baco_gt(graph, i, label)

        total += 1
        for node1, node2 in sub_graph.edges():
            if node1 in ground_nodes and node2 in ground_nodes:
                answer += 1
        if len(sub_graph.edges())<1:
            pass
        else:
            precisoin += answer/len(sub_graph.edges())
        if data_type == 'syn5':
            num_ground_edge = 12
        else:
            num_ground_edge = 6
        recall += answer/num_ground_edge
        answer2 = (answer+len(graph.edges())-num_ground_edge)
        if len(sub_graph.edges())>num_ground_edge:
            answer2 -= (len(sub_graph.edges())-num_ground_edge)
        if len(graph.edges())<1:
            pass
        else:
            acc += answer2/len(graph.edges())
            
    acc = acc/total
    prc =precisoin/total
    rc = recall/total 
    f1_score = f1(prc, rc)

    return prc, rc