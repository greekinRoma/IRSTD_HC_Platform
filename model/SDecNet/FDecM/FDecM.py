import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import nn
import math
from ..AttentionModule.nonlocal_module import _NonLocalBlockND
from .contrast_and_atrous import AttnContrastLayer
class SDecM(nn.Module):
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
        self.origin_conv = nn.Conv2d(in_channels=self.in_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1,bias=False)
        self.out_conv = nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False)
    def Extract_layer(self,cen,b,w,h):
        basises = []
        origin = self.origin_conv(cen)
        for i in range(len(self.shifts)):
            basis = torch.nn.functional.conv2d(weight=self.kernels,stride=1,padding="same",input=cen,groups=self.hidden_channels,dilation=self.shifts[i])
            basises.append(basis)
        origin=origin.view(b,self.hidden_channels,1,-1)
        basis1 = torch.stack(basises,dim=2)
        basis2 = torch.nn.functional.normalize(basis1.view(b,self.hidden_channels,self.num_layer*self.num_shift,-1),dim=-1)
        basis1 = basis2.transpose(-2,-1)
        weight_score = torch.matmul(origin,basis1)
        out = torch.matmul(weight_score,basis2).view(b,self.hidden_channels,w,h)
        return out
    def forward(self,cen):
        b,_,w,h= cen.shape
        out = self.Extract_layer(cen,b,w,h)
        return out
class SDecD(nn.Module):
    def __init__(self,in_channels,out_channels,shifts,kernel_size,use_norm=True):
        super().__init__()
        #The hyper parameters settting
        self.hidden_channels = in_channels//kernel_size
        self.in_channels = in_channels
        self.convs_list=nn.ModuleList()
        self.shifts = shifts
        self.kernel_size = kernel_size
        self.num_shift = len(self.shifts)
        kernel=np.array([[[1, -1], [-1, 1]],
                         [[1, -1,], [1, -1]],
                         [[1, 1,], [-1, -1]]])/2
        self.num_layer = 3
        self.kernel = torch.from_numpy(kernel).float().cuda().view(-1,1,2,2)
        self.kernels = self.kernel.repeat(self.hidden_channels,1,1,1)
        self.origin_conv = nn.Sequential(
            nn.AvgPool2d((2,2)),
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1)
        )
    def Extract_layer(self,cen,b,w,h):
        origins = self.origin_conv(cen)
        origins = origins.view(b,self.hidden_channels,1,-1)
        Basis = torch.nn.functional.conv2d(weight=self.kernels,stride=2,input=cen,groups=self.hidden_channels).view(b,self.hidden_channels,self.num_layer,-1)
        Basis1 = torch.nn.functional.normalize(Basis,dim=-1)
        Basis2 = Basis1.transpose(-2,-1)
        weight_score = torch.matmul(origins,Basis2)
        out = torch.matmul(weight_score,Basis1).view(b,self.hidden_channels,w//2,h//2)
        return out
    def forward(self,cen):
        b,_,w,h= cen.shape
        out = self.Extract_layer(cen,b,w,h)
        return out