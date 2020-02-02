import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

class MultiLayerModel(nn.Module):
    def __init__(self, ls):
        super().__init__()
        self.model = nn.Sequential(*[nn.Sequential(nn.Linear(*n),
                                                   nn.ReLU()) for n in ls])
        
    def forward(self, X):
        return torch.sigmoid(self.model(X))

def get_layers(ni, hs, nc, step):
    lse = [*list(range(hs, nc, -step)), nc]
    return list(zip([ni,*lse[:]], [*lse[:], nc]))[:-1]
