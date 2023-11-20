import torch
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges

dataset = Planetoid("datasets","CiteSeer")

G = nx.Graph(dataset.data)



