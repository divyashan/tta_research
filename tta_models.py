from torch import nn, optim
import torch

class ClassTTA(nn.Module):
    def __init__(self, n_augs, n_classes, temp_scale=1):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn((n_augs, n_classes), requires_grad=True, dtype=torch.float))
        self.coeffs.data.fill_(1.0/n_augs) 
        self.temperature = temp_scale
    
    def forward(self, x):
        x = x/self.temperature
        mult = self.coeffs * x
        return mult.sum(axis=1)

class AugTTA(nn.Module):
    
    def __init__(self, n_augs, n_classes, temp_scale=1):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn((n_augs,1 ), requires_grad=True, dtype=torch.float))
        self.temperature = temp_scale
        self.coeffs.data.fill_(1.0/n_augs) 
        
    def forward(self, x):
        x = x/self.temperature
        mult = torch.matmul(x.transpose(1, 2), self.coeffs / torch.sum(self.coeffs, axis=0))
        return mult.squeeze()
