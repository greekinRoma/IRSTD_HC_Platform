from torch import nn
from ..NSLayer import NSLayer
import torch
class DecLayer(nn.Module):
    def __init__(self, num_sample, channel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_sample = num_sample
        self.ns_layer = NSLayer(channel=channel,kernel=num_sample)
    def forward(self,origin,basis):
        basisT = torch.transpose(basis,dim0=2,dim1=3)
        weight = torch.matmul(origin,basisT)
        output = torch.matmul(weight,basis)
        return output
        