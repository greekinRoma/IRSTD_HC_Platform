import numpy as np
import torch
from torch import nn
from .SortConv import SortConv
class SDecP(nn.Module):
    def __init__(self,dim,ratio=1):
        super().__init__()
        #The hyper parameters settting
        self.hidden_channels = dim//ratio
        self.in_channels = dim
        self.convs_list=nn.ModuleList()
        self.kernel_size = ratio
        kernel=np.array([[[1, -1], [1, -1]],
                         [[1, 1],[-1, -1]],
                         [[1, -1,], [-1, 1]],
                         ])
        self.num_layer = 3
        self.max_pool = nn.MaxPool2d((2,2))
        self.avg_pool = nn.AvgPool2d((2,2))
        self.kernel = torch.from_numpy(kernel).float().cuda().view(-1,1,2,2)
        self.kernels = self.kernel.repeat(self.hidden_channels,1,1,1)
        self.origin_conv = nn.Sequential(
            nn.AvgPool2d((2,2)),
            nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=1,stride=1)
        )
    def Extract_layer(self,cen,b,w,h):
        edge = torch.nn.functional.conv2d(weight=self.kernels.to(cen.device),stride=2,input=cen,groups=self.hidden_channels).view(b,self.hidden_channels,self.num_layer,-1)
        max1 = self.max_pool(cen)
        max = max1.view(b,self.hidden_channels,1,-1)
        basis = torch.concat([max,edge],dim=2)
        Basis1 = torch.nn.functional.normalize(basis,dim=-1)
        Basis2 = Basis1.transpose(-2,-1)
        origins = self.origin_conv(cen)
        origins = origins.view(b,self.hidden_channels,1,-1)
        weight_score = torch.matmul(origins,Basis2)
        out = torch.matmul(weight_score,Basis1).view(b,self.hidden_channels,w//2,h//2)
        return out
    def forward(self,cen):
        b,_,w,h= cen.shape
        out = self.Extract_layer(cen,b,w,h)
        return out