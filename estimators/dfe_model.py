import torch
import torch.nn.functional as F
from torch import nn

class DFE_model(nn.Module):
    def __init__(self, A, V, F, W):
        super(DFE_model, self).__init__()
        self.A = A
        self.V = V
        self.F = F
        self.W = nn.Parameter(W, requires_grad=True)

    def dist(self, X):
        D = torch.zeros(X.size()[0], self.A.size()[0], self.A.size()[1])
        idc = self.A.indices().transpose(0,1)
        for i, idc in enumerate(idc):
            j = idc[0]
            k = idc[1]
            D[:,j,k] = self.A.values()[i]*X[:,k//2] - self.V[j,k]
        D = torch.nn.functional.relu(D)
        return D

    def delta(self, dist):
        # j leaves, k conditions, m samples
        return torch.einsum('mjk,mjk->mj', -(dist*self.W), dist).exp()

    def h(self, delta, s):
        return (delta*s).sum(dim=-1)

    def forward(self, X, training=False):
        D = self.dist(X)
        H = torch.zeros(D.size()[0])
        Delta = self.delta(D)
        H = self.h(Delta, self.F)
        return H

class BigModel(nn.Module):
    def __init__(self, models):
        super(BigModel, self).__init__()
        self.m = nn.ModuleList(models)
        
    def forward(self, X, training=True):
        outs = [m(X) for i, m in enumerate(self.m)]
        out = torch.stack(outs)
        out = out.transpose(0,1)
        return out