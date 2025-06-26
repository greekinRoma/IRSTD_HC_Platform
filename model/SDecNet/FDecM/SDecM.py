import torch
from torch import nn
import numpy as np
from torch import nn
import math
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
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1),
                                      nn.BatchNorm2d(self.hidden_channels),
                                      nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.in_channels,kernel_size=1,stride=1))
        self.basis_convs = nn.ModuleList()
        self.origin_convs = nn.ModuleList()
        self.num_layer = 8
        self.down_layer = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1),
                                        nn.BatchNorm2d(self.hidden_channels))
        self.origin_conv = nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1,bias=False)
        self.trans_conv = nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.params = nn.Parameter(torch.ones(1,1,1,1),requires_grad=True).cuda()
    def Extract_layer(self,cen,b,w,h):
        basises = [(self.max_pool(cen)-self.avg_pool(cen)).view(b,self.hidden_channels,1,-1)]
        for i in range(len(self.shifts)):
            basis = torch.nn.functional.conv2d(weight=self.kernels,stride=1,padding="same",input=cen,groups=self.hidden_channels,dilation=self.shifts[i]).view(b,self.hidden_channels,self.num_layer,-1)
            basises.append(basis)
        origin = self.origin_conv(cen)
        origin = origin + torch.rand_like(origin)
        origin = self.trans_conv(origin)
        origin=origin.view(b,self.hidden_channels,1,-1)
        basis1 = torch.concat(basises,dim=2)
        basis2 = torch.nn.functional.normalize(basis1,dim=-1)
        basis1 = basis2.transpose(-2,-1)
        weight_score = torch.matmul(origin,basis1)
        out = torch.matmul(weight_score,basis2).view(b,self.hidden_channels,w,h)
        return out
    def forward(self,cen):
        b,_,w,h= cen.shape
        cen = self.down_layer(cen)
        out = self.Extract_layer(cen,b,w,h)
        out = self.out_conv(out)
        return out