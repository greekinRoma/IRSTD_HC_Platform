from .FSLayer import FSLayer
from .DecLayer import DecLayer
from torch import nn
import math
import torch
class SDecF(nn.Module):
    def __init__(self, n_channels=[4,8,16,32],channel_ratios =[2,4,8,16], down_ratios =[8,4,2,1],kernel_size = 4,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_channels = n_channels
        self.channel_ratios = channel_ratios
        self.kernel_size = kernel_size
        self.sample_num = 4
        self.CFLayers = nn.ModuleList()
        self.OTLayers = nn.ModuleList()
        self.ORLayers = nn.ModuleList()
        self.CTLayers = nn.ModuleList()
        self.DeLayers = nn.ModuleList()
        self.hidden_channels = []
        for n_channel,down_ratio,channel_ratio in zip(self.n_channels,down_ratios,self.channel_ratios):
            hidden_channel = n_channel//channel_ratio
            self.hidden_channels.append(hidden_channel)
            self.CFLayers.add_module(
                nn.Sequential(
                nn.Conv2d(in_channels=n_channel,out_channels=hidden_channel,kernel_size=1,stride=1),
                FSLayer(channel=n_channel,kernel=self.sample_num,shifts=[1]),
                ))
            self.ORLayers.add_module(nn.Sequential(
                nn.Conv2d(in_channels=hidden_channel,out_channels=hidden_channel,kernel_size=1,stride=1),
                nn.Conv2d(in_channels=hidden_channel,out_channels=hidden_channel,kernel_size=1,stride=1),
            ))
            self.CTLayers.add_module(nn.ConvTranspose2d(in_channels=hidden_channel,out_channels=hidden_channel,kernel_size=down_ratio,stride=down_ratio))
            self.OTLayers.add_module(nn.Conv2d(in_channels=hidden_channel,out_channels=n_channel,kernel_size=1,stride=1))
            self.DeLayers.add_module(DecLayer(num_sample=self.sample_num,channel=hidden_channel))
        self.sum_hidden_channels = sum(self.hidden_channels)
        self.fuse_layer = nn.Conv2d(in_channels=self.sum_hidden_channels,out_channels=self.sum_hidden_channels,kernel_size=1,stride=1)
    def forward(self,inp0,inp1,inp2,inp3):
        b = inp0.size(0)
        feature_map_sample_0 = self.CFLayers[0](inp0)
        feature_map_sample_1 = self.CFLayers[1](inp1)
        feature_map_sample_2 = self.CFLayers[2](inp2)
        feature_map_sample_3 = self.CFLayers[3](inp3)
        feature_map_sample = torch.concat([feature_map_sample_0,feature_map_sample_1,feature_map_sample_2,feature_map_sample_3],dim=1)
        feature_map_sample = self.fuse_layer(feature_map_sample)
        fps0,fps1,fps2,fps3 = torch.split(feature_map_sample,self.hidden_channels,dim=1)
        fps0 = self.CTLayers[0](fps0).view(b,self.hidden_channels[0],self.sample_num,-1)
        fps1 = self.CTLayers[1](fps1).view(b,self.hidden_channels[1],self.sample_num,-1)
        fps2 = self.CTLayers[2](fps2).view(b,self.hidden_channels[2],self.sample_num,-1)
        fps3 = self.CTLayers[3](fps3).view(b,self.hidden_channels[3],self.sample_num,-1)
        basis0 = torch.nn.functional.normalize(fps0,dim=-1)/math.sqrt(self.kernel_size)
        basis1 = torch.nn.functional.normalize(fps1,dim=-1)/math.sqrt(self.kernel_size)
        basis2 = torch.nn.functional.normalize(fps2,dim=-1)/math.sqrt(self.kernel_size)
        basis3 = torch.nn.functional.normalize(fps3,dim=-1)/math.sqrt(self.kernel_size)
        origin0 = self.ORLayers[0](inp0)
        origin1 = self.ORLayers[1](inp1)
        origin2 = self.ORLayers[2](inp2)
        origin3 = self.ORLayers[3](inp3)
        output0 = self.DeLayers[0](origin0,basis0)
        output1 = self.DeLayers[1](origin1,basis1)
        output2 = self.DeLayers[2](origin2,basis2)
        output3 = self.DeLayers[3](origin3,basis3)
        output0 = self.OTLayers[0](output0)
        output1 = self.OTLayers[1](output1)
        output2 = self.OTLayers[2](output2)
        output3 = self.OTLayers[3](output3)
        return output0,output1,output2,output3