import numpy as np
import torch
from torch import nn
from .NSLayer import NSLayer
import math
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
                        [[1/4,1/4],[1/4,1/4]]
                         ])
        self.num_layer = 3
        self.kernel = torch.from_numpy(kernel).float().cuda().view(-1,1,2,2)
        self.kernels = self.kernel.repeat(self.hidden_channels,1,1,1)
        self.origin_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.hidden_channels,kernel_size=2,stride=2),
            nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1),
            nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1),
        )
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.params = nn.Parameter(torch.ones(1,1,1,1)/2,requires_grad=True)
        self.kernel = 4
        self.NSs = NSLayer(kernel=self.kernel,channel=self.hidden_channels)
        self.trans_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.hidden_channels*self.kernel,kernel_size=(4,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden_channels*self.kernel,out_channels=self.hidden_channels*self.kernel,kernel_size=1),
        )
    def Extract_layer(self,cen,b,w,h):
        basis = torch.nn.functional.conv2d(weight=self.kernels,stride=2,input=cen,groups=self.hidden_channels).view(b,self.hidden_channels,self.num_layer,-1)
        max_value = self.max_pool(cen).view(b,self.hidden_channels,1,-1)
        basis = torch.concat([basis,max_value],dim=2)
        basis = self.trans_layer(basis).view(b,self.hidden_channels,self.kernel,-1)
        basis = torch.nn.functional.normalize(basis,dim=-1,p=2)/2
        basis1 = self.NSs(basis) 
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