import torch 
from torch import nn
from .SoftPool import SoftPool
from ..FDecM.SDecD import SDecD
class InceptionPool(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=2,stride=2):
        super().__init__()
        # self.sdem = SDecD(in_channels=in_channel,out_channels=in_channel,kernel_size=1,shifts=[1])
        self.max_pool = nn.MaxPool2d((2,2))
        self.avg_pool = nn.AvgPool2d((2,2))
        self.sort_conv = SDecD(dim=in_channel//4)
    def forward(self,inp):
        # out = self.sdem(inp)
        out1,out2,out3,out4 = torch.chunk(inp,dim=1,chunks=4)
        out1 = self.max_pool(out1)
        out2 = self.avg_pool(out2)
        out3 = -self.max_pool(-out3)
        out4 = self.sort_conv(out4)
        out = torch.concat([out1,out2,out3,out4],dim=1)
        return out