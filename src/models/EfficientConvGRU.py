import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from efficientnet_pytorch import EfficientNet 

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim = 1, num_layers =1,
                 dropout = 0, nf = 25):
        super().__init__()
        self.conv = EfficientNet.from_pretrained('efficientnet-b7')
        self.gru = nn.GRU(2560 * 4 * 4, hidden_dim, num_layers, dropout = dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_dim*nf,512),
                                nn.Linear(512,1))
        self.nf = nf
        
    def forward(self, X, verbose = False):
        init = True
        for seq in X:
            fm = self.conv.extract_features(seq)
            enc = torch.flatten(fm, start_dim = 1).unsqueeze(0) if init else torch.cat([enc, torch.flatten(fm, start_dim = 1).unsqueeze(0)], dim = 0)
            init = False
        enc = self.gru(enc.transpose(0,1))[0].transpose(0,1)
        x = torch.flatten(enc, start_dim=1)
        return torch.sigmoid(self.fc(x))
