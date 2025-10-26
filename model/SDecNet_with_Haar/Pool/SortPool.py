import torch
from torch import nn
class SortPool(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self,inp):
        out = torch.stack([inp[:,:,0::2,0::2],inp[:,:,1::2,0::2],inp[:,:,0::2,1::2],inp[:,:,1::2,1::2]],dim=-1)
        out = torch.sort(out,dim=-1).values
        out1 , out2, out3, out4 = torch.chunk(out,dim=1,chunks=4)
        out = torch.concat([out1[...,0],out2[...,1],out3[...,2],out4[...,3]],dim=1)
        return out