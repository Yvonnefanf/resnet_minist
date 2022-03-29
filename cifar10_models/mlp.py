import os

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["mlp3"]

class MLP(nn.Module):
    def __init__(self, hidden_layer):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer,hidden_layer)
        self.fc3 = nn.Linear(hidden_layer,10)
        self.droput = nn.Dropout(0.2)
        
    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.droput(x)
        x = F.relu(self.fc2(x))
        x = self.droput(x)
        x = self.fc3(x)
        return x


def mlp3():
    return MLP(512)