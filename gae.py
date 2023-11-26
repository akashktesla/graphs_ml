import torch
from torch import nn
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv,GAE
from torch_geometric.utils import train_test_split_edges

def main():
    dataset = Planetoid("datasets","CiteSeer")
    data = dataset[0]
    data.train_mask = data.val_mask = data.test_mask = None
    data = train_test_split_edges(data)

    #model params
    in_channels = data.num_features
    out_channels = 2
    lr = 0.01
    epochs = 100

    #initializing the model
    encoder = GCNEncoder(in_channels,out_channels)
    model = GAE(encoder) #it has a default decoder
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)

    #training
    for epoch in range(1, epochs + 1):
        loss = train(model,optimizer,x,train_pos_edge_index)
        auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    

def train(model,optimizer,x,train_pos_edge_index):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x,train_pos_edge_index)
    loss = model.recon_loss(z,train_pos_edge_index) #calculate reconstruction loss
    loss.backward()
    optimizer.step()
    return float(loss)

def test():
    model.eval()
    with torch.no_grad():
        z = model.encode(x,train_pos_index)
    return model.test(z,pos_edge_index,neg_edge_index)


#Encoder
class GCNEncoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(GCNEncoder,self).__init__()
        self.conv1 = GCNConv(in_channels,2*in_channels)
        self.conv2 = GCNConv(2*in_channels,2*out_channels)
    def forward(x,edge_index):
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = self.conv2(x,edge_index)
        return x


def visualize(dataset):
    ei_x = dataset.data.edge_index[0]
    ei_y = dataset.data.edge_index[1]
    len_ei = len(ei_x)
    edge_list = []
    for i in range(len_ei): 
        # print(ei_x[i],ei_y[i])
        edge_list.append((ei_x[i].item(),ei_y[i].item()))
    print(edge_list)
    visualization 
    G = nx.Graph(dataset.data)
    G.add_edges_from(edge_list)
    nx.draw_spring(G,with_labels = True)
    plt.show()

if __name__ == "__main__":
    main()
