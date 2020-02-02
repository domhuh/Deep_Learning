import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from efficientnet_pytorch import EfficientNet 

class ConvCGRUCell(nn.Module):
    def __init__(self, nh):
        super().__init__()
        self.nh = nh
        self.conv = EfficientNet.from_pretrained('efficientnet-b7')
        self.czr = nn.Conv2d(2560 + self.nh, nh*2, 3, padding = 1)
        self.chh = nn.Conv2d(2560 + self.nh, nh, 3, padding = 1)
        self.init = True
        
    def forward(self, x, h=None):
        if self.init: 
            self._build(x.shape[0])
            self.init = False
        if h is not None: self.h = h
        fm = self.conv.extract_features(x)
        fmzr = torch.cat((fm,self.h), dim=1)
        enc = self.czr(fmzr)
        z, r = torch.split(enc, self.nh, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        fmh = torch.cat((fm,torch.mul(self.h,r)), dim=1)
        hhat = torch.tanh(self.chh(fmh))
        self.h = torch.mul(hhat,1-z) + torch.mul(self.h,z)
        return self.h
    
    def _build(self, batch_size):
        self.h = torch.normal(0,1,(batch_size,self.nh,4,4))