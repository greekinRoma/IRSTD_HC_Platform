from torch.nn.functional import unfold
from torch import nn
import torch 
class SortConv(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unfold_layer = nn.Unfold(kernel_size=2,stride=2)
    def forward(self,inp):
        b,c ,h,w = inp.shape
        out = self.unfold_layer(inp).view(b,c,4,-1)
        out = torch.sort(out,dim=2).values.view(b,4*c,h//2,w//2)
        return out