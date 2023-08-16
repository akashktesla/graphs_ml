import torch

def create_adj(size):
    a = torch.rand(size,size)
    a[a>0.5] = 1
    a[a<=0.5] = 0
    # for illustration i set the diagonal elemtns to zero
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i==j:
                a[i,j] = 0
    return a

def degree_matrix(a):
    return torch.diag(a.sum(dim=-1))

def graph_lapl(a):
    return degree_matrix(a)-a



