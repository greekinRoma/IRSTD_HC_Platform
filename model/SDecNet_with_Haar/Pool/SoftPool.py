from torch.nn.functional import unfold
from torch import nn
import torch
class SoftPool(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self,inp):
        out = torch.stack([inp[:,:,0::2,0::2],inp[:,:,1::2,0::2],inp[:,:,0::2,1::2],inp[:,:,1::2,1::2]],dim=2)
        out = torch.nn.functional.normalize(out,p=2,dim=2)*out
        out = torch.sum(out,dim=2)
        return out