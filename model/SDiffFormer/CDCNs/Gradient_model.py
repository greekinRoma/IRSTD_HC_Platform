import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import nn
import math
from ..AttentionModule.nonlocal_module import _NonLocalBlockND
from .contrast_and_atrous import AttnContrastLayer
class ExpansionContrastModule(nn.Module):
    def __init__(self,in_channels,out_channels,shifts,kernel_size,use_norm=True):
        super().__init__()
        #The hyper parameters settting
        self.hidden_channels = in_channels//2
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
        sumkel=(np.array([[[1.,1.,1.],[1.,8.,1.],[1.,1.,1.]]])/8).reshape(1,1,3,3)
        kernel=np.concatenate([delta1,delta2,sumkel],axis=0)
        self.kernel = torch.from_numpy(kernel).float().cuda()
        self.kernels = self.kernel.repeat(self.hidden_channels,1,1,1)
        if use_norm == True:
            self.out_conv = nn.Sequential(nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False),
                                    nn.BatchNorm2d(self.in_channels),
                                    nn.ReLU())
        else: 
            self.out_conv = nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False)
        self.value_convs = nn.ModuleList()
        self.key_convs = nn.ModuleList()
        self.query_conv = nn.Conv2d(in_channels=self.in_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1,bias=False)
        self.num_layer = 9
        for _ in range(len(self.shifts)):
            self.value_convs.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1,bias=False))
            self.key_convs.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1,bias=False))
    def Extract_layer(self,cen,b,w,h):
        keys = []
        values = []
        for i in range(len(self.shifts)):
            value = self.value_convs[i](cen)
            key = self.key_convs[i](cen)
            value = torch.nn.functional.conv2d(weight=self.kernels,stride=1,padding="same",input=value,groups=self.hidden_channels,dilation=self.shifts[i])
            key = torch.nn.functional.conv2d(weight=self.kernels, stride=1, padding="same", input=key,groups=self.hidden_channels,dilation=self.shifts[i])
            keys.append(key)
            values.append(value)
        querys = self.query_conv(cen)
        keys = torch.stack(keys,dim=2)
        values = torch.stack(values,dim=2)
        keys = torch.nn.functional.normalize(keys.view(b,self.hidden_channels,self.num_layer*self.num_shift,-1),dim=-1).transpose(-2,-1)
        querys = querys.view(b,self.hidden_channels,1,-1)
        values = values.view(b,self.hidden_channels,self.num_layer*self.num_shift,-1)
        weight_score = torch.matmul(querys,keys)
        out = torch.matmul(weight_score,values).view(b,self.hidden_channels,w,h)
        return out
    def forward(self,cen):
        b,_,w,h= cen.shape
        out = self.Extract_layer(cen,b,w,h)
        out = self.out_conv(out)
        return out