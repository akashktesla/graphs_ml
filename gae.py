import torch
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges

dataset = Planetoid("datasets","CiteSeer")
print(dataset.data)
print(dataset.data.edge_index)
ei_x = dataset.data.edge_index[0]
ei_y = dataset.data.edge_index[1]
len_ei = len(edge_index_y)
for i in range(len_ei): 
    print(ei_x[i],ei_y[i])

# G = nx.Graph(dataset.data)




