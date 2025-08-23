import numpy as np
import torch
from torch import nn
from .NSLayer import NSLayer
class SD2D(nn.Module):
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
        self.kernel = torch.from_numpy(kernel).float().cuda().view(-1,1,2,2)
        self.kernels = self.kernel.repeat(self.hidden_channels,1,1,1)
        self.origin_conv = nn.Sequential(
            nn.AvgPool2d((2,2)),
            nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1,bias=False),
            nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1),
        )
        self.NSs = nn.Sequential(NSLayer(channel=self.hidden_channels,kernel=4),
                                 NSLayer(channel=self.hidden_channels,kernel=4),)
    def Extract_layer(self,cen,b,w,h):
        edge = torch.nn.functional.conv2d(weight=self.kernels,stride=2,input=cen,groups=self.hidden_channels).view(b,self.hidden_channels,self.num_layer,-1)
        max1 = self.max_pool(cen)
        max1 = max1.view(b,self.hidden_channels,1,-1)
        basis = torch.concat([max1,edge],dim=2)
        basis = torch.nn.functional.normalize(basis,dim=-1)/2
        basis1 = self.NSs(basis)
        basis1 = torch.nn.functional.normalize(basis1,dim=-1)
        basis2 = basis1.transpose(-2,-1)
        origin = self.origin_conv(cen)
        origin = origin.view(b,self.hidden_channels,1,-1)
        weight_score = torch.matmul(origin,basis2)
        out = torch.matmul(weight_score,basis1).view(b,self.hidden_channels,w//2,h//2)
        return out
    def forward(self,cen):
        b,_,w,h= cen.shape
        out = self.Extract_layer(cen,b,w,h)
        return out