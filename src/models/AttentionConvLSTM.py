import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)

def get_layers(ni, hs, nc, step):
    lse = [*list(range(hs, nc, -step)), nc]
    return list(zip([ni,*lse[:]], [*lse[:], nc]))[:-1]
    
class AttentionConvLSTMNet(nn.Module):
    def __init__(self, ni, nf, nh, hs, ls):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(ni,nf,3, padding = 1),
                                  nn.BatchNorm1d(nf),
                                  nn.ReLU(),
                                  nn.Conv1d(nf,nh,3, padding = 1))
        
        self.key = nn.Conv1d(nh,nh,3,padding=1)
        self.value = nn.Conv1d(nh,nh,3,padding=1)
        self.query = nn.Conv1d(nh,nh,3,padding=1)
        
        self.lstm = nn.LSTM(nh, hs)
        self.model = nn.Sequential(*[nn.Sequential(nn.Linear(*n), nn.ReLU()) for n in ls])
        
    def forward(self, X):
        fm = self.conv(X)
        
        attnmap = torch.bmm(self.key(fm), self.query(fm))
        atfm = torch.softmax(attnmap,dim =1) * self.value(fm)
        
        enc = self.lstm(atfm.transpose(0,1))[0].squeeze()
        return torch.sigmoid(self.model(enc)).squeeze()