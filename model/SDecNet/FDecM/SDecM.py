import torch
from torch import nn
import numpy as np
from torch import nn
import math
from .NSLayer import NSLayer
class SD2M(nn.Module):
    def __init__(self,in_channels,out_channels,shifts,kernel_size,use_norm=True):
        super().__init__()
        #The hyper parameters settting
        self.hidden_channels = in_channels//kernel_size
        self.in_channels = in_channels
        self.convs_list=nn.ModuleList()
        self.shifts = shifts
        self.kernel_size = kernel_size
        self.num_shift = len(self.shifts)
        delta1=np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, 0]],
                         [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
                         [[0, 0, -1], [0, 1, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 1, -1], [0, 0, 0]]])
        delta1=delta1.reshape(4,1,3,3)
        delta2=delta1[:,:,::-1,::-1].copy()
        kernel=np.concatenate([delta1,delta2],axis=0)
        self.kernel = torch.from_numpy(kernel).float().cuda()
        self.kernels = self.kernel.repeat(self.hidden_channels,1,1,1)
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.in_channels,kernel_size=1,stride=1))
        self.basis_convs = nn.ModuleList()
        self.origin_convs = nn.ModuleList()
        self.num_layer = 8
        self.down_layer = nn.Conv2d(in_channels=self.in_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1)
        self.origin_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1),
            nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1),
        )
        self.kernel = 8
        self.NSs = NSLayer(kernel=self.kernel,channel=self.hidden_channels)
        self.trans_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.hidden_channels*self.kernel,kernel_size=(self.num_layer*2,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden_channels*self.kernel,out_channels=self.hidden_channels*self.kernel,kernel_size=1),
            )
    def Extract_layer(self,cen,b,w,h):
        basises = []
        for i in range(len(self.shifts)):
            basis = torch.nn.functional.conv2d(weight=self.kernels,stride=1,padding="same",input=cen,groups=self.hidden_channels,dilation=self.shifts[i]).view(b,self.hidden_channels,self.num_layer,-1)
            basises.append(basis)
        basis = torch.concat(basises,dim=2)
        basis = self.trans_layer(basis).view(b,self.hidden_channels,self.kernel,-1)
        basis = torch.nn.functional.normalize(basis,dim=-1,p=2)/4
        basis2 = self.NSs(basis)
        basis1 = basis2.transpose(-2,-1)
        origin = self.origin_conv(cen)
        origin = origin.view(b,self.hidden_channels,1,-1)
        weight_score = torch.matmul(origin,basis1)
        out = torch.matmul(weight_score,basis2).view(b,self.hidden_channels,w,h)
        return out
    def forward(self,cen):
        b,_,w,h= cen.shape
        cen = self.down_layer(cen)
        out = self.Extract_layer(cen,b,w,h)
        out = self.out_conv(out)
        return out