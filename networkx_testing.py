import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
edge_list = [(1,2),(2,4),(3,4),(7,8),(9,1)]
G.add_edges_from(edge_list)
# G.add_edge(1,2, weight = 0.6)
nx.draw_spring(G,with_labels = True)

# G = nx.MultiGrap()
# G = nx.MultiDiGrap()
