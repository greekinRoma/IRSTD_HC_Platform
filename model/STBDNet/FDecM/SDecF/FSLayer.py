import torch
from torch import nn
import numpy as np
from torch import nn
import math
from .. import NSLayer
class FSLayer(nn.Module):
    def __init__(self,channel,kernel_size,shifts):
        super().__init__()
        #The hyper parameters settting
        self.hidden_channels = channel
        self.convs_list=nn.ModuleList()
        self.shifts = shifts
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
        self.num_layer = 8
        self.kernel = kernel_size
        self.trans_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_channels,out_channels=self.hidden_channels*self.kernel,kernel_size=kernel, stride=kernel),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden_channels*self.kernel,out_channels=self.hidden_channels*self.kernel,kernel_size=1),
            )
    def forward(self,cen):
        b,c,h,w = cen.shape
        basises = []
        for i in range(len(self.shifts)):
            basis = torch.nn.functional.conv2d(weight=self.kernels,stride=1,padding="same",input=cen,groups=self.hidden_channels,dilation=self.shifts[i])
            basises.append(basis)
        basis = torch.concat(basises,dim=1)
        basis = self.trans_layer(basis)
        return basis