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
        self.FSLayers = nn.ModuleList()
        self.OTLayers = nn.ModuleList()
        self.ORLayers = nn.ModuleList()
        self.CTLayers = nn.ModuleList()
        self.DeLayers = nn.ModuleList()
        self.DWLayers = nn.ModuleList()
        self.hidden_channels = []
        self.split_channels = []
        for n_channel,down_ratio,channel_ratio in zip(self.n_channels,down_ratios,self.channel_ratios):
            hidden_channel = n_channel//channel_ratio
            self.hidden_channels.append(hidden_channel)
            self.split_channels.append(hidden_channel*self.sample_num)
            self.DWLayers.append(
                nn.Conv2d(in_channels=n_channel,out_channels=hidden_channel,kernel_size=1,stride=1)
            )
            self.FSLayers.append(FSLayer(channel=hidden_channel,kernel_size=down_ratio,sample_num=self.sample_num,shifts=[1]))
            self.ORLayers.append(nn.Sequential(
                nn.Conv2d(in_channels=hidden_channel,out_channels=hidden_channel,kernel_size=1,stride=1),
                nn.Conv2d(in_channels=hidden_channel,out_channels=hidden_channel,kernel_size=1,stride=1),
            ))
            self.CTLayers.append(nn.ConvTranspose2d(in_channels=hidden_channel*self.sample_num,out_channels=hidden_channel*self.sample_num,kernel_size=down_ratio,stride=down_ratio))
            self.OTLayers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=n_channel,kernel_size=1,stride=1))
            self.DeLayers.append(DecLayer(num_sample=self.sample_num,channel=hidden_channel))
        self.sum_split_channels = sum(self.split_channels)
        self.fuse_layer = nn.Conv2d(in_channels=self.sum_split_channels,out_channels=self.sum_split_channels,kernel_size=1,stride=1)
    def forward(self,inp0,inp1,inp2,inp3):
        b,c,h,w = inp0.shape
        inp0 = self.DWLayers[0](inp0)
        inp1 = self.DWLayers[1](inp1)
        inp2 = self.DWLayers[2](inp2)
        inp3 = self.DWLayers[3](inp3)
        feature_map_sample_0 = self.FSLayers[0](inp0)
        feature_map_sample_1 = self.FSLayers[1](inp1)
        feature_map_sample_2 = self.FSLayers[2](inp2)
        feature_map_sample_3 = self.FSLayers[3](inp3)
        feature_map_sample = torch.concat([feature_map_sample_0,feature_map_sample_1,feature_map_sample_2,feature_map_sample_3],dim=1)
        feature_map_sample = self.fuse_layer(feature_map_sample)
        fps0,fps1,fps2,fps3 = torch.split(feature_map_sample,self.split_channels,dim=1)
        fps0 = self.CTLayers[0](fps0).view(b,self.hidden_channels[0],self.sample_num,-1)
        fps1 = self.CTLayers[1](fps1).view(b,self.hidden_channels[1],self.sample_num,-1)
        fps2 = self.CTLayers[2](fps2).view(b,self.hidden_channels[2],self.sample_num,-1)
        fps3 = self.CTLayers[3](fps3).view(b,self.hidden_channels[3],self.sample_num,-1)
        basis0 = torch.nn.functional.normalize(fps0,dim=-1)/math.sqrt(self.kernel_size)
        basis1 = torch.nn.functional.normalize(fps1,dim=-1)/math.sqrt(self.kernel_size)
        basis2 = torch.nn.functional.normalize(fps2,dim=-1)/math.sqrt(self.kernel_size)
        basis3 = torch.nn.functional.normalize(fps3,dim=-1)/math.sqrt(self.kernel_size)
        origin0 = self.ORLayers[0](inp0).view(b,self.hidden_channels[0],1,-1)
        origin1 = self.ORLayers[1](inp1).view(b,self.hidden_channels[1],1,-1)
        origin2 = self.ORLayers[2](inp2).view(b,self.hidden_channels[2],1,-1)
        origin3 = self.ORLayers[3](inp3).view(b,self.hidden_channels[3],1,-1)
        output0 = self.DeLayers[0](origin0,basis0).view(b,self.hidden_channels[0],h//1,w//1)
        output1 = self.DeLayers[1](origin1,basis1).view(b,self.hidden_channels[1],h//2,w//2)
        output2 = self.DeLayers[2](origin2,basis2).view(b,self.hidden_channels[2],h//4,w//4)
        output3 = self.DeLayers[3](origin3,basis3).view(b,self.hidden_channels[3],h//8,w//8)
        output0 = self.OTLayers[0](output0)
        output1 = self.OTLayers[1](output1)
        output2 = self.OTLayers[2](output2)
        output3 = self.OTLayers[3](output3)
        return output0,output1,output2,output3