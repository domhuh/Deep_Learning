import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ./ConvCGRUCell import *

class ConvCGRU(nn.Module):
    def __init__(self, nh, dim=4):
        super().__init__()
        self.cgru = ConvCGRUCell(nh)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(nh, 1)
        self.losses, self.accuracy = [], []

    def forward(self,x):
        self.cgru._build(1)
        for img in x:
            self.cgru(img[None,:])
        pfm = self.pool(self.cgru.h)
        return torch.sigmoid(self.fc(torch.flatten(pfm,start_dim=1)))