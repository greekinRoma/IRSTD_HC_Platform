import torch 
from torch import nn
from torch.nn import functional as F
class GaussianFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = torch.tensor([[1,2,1],[2,4,2],[1,2,1]])/16
        self.kernel = self.kernel.unsqueeze(0).unsqueeze(0).cuda()  # Add batch and channel dimensions
        self.padding = nn.ReflectionPad2d((1,1,1,1))
    def forward(self, input_image):
        kernel = self.kernel.expand(input_image.shape[1], -1, -1, -1)  # Expand for all input channels
        input_image = self.padding(input_image)
        output_image = F.conv2d(input_image, kernel, groups=input_image.shape[1])
        return output_image
class OrdFilter(nn.Module):
    def __init__(self,size,K=9):
        super().__init__()
        self.sample = nn.Sequential(
            nn.ReflectionPad2d((size//2,size//2,size//2,size//2)),
            nn.Unfold(kernel_size=size,stride=1)
        )
        self.size = size
        self.K = K
    def forward(self,inp):
        b,c,h,w = inp.size()
        sample = self.sample(inp)
        sample = torch.sort(sample,dim=1,descending=True).values
        sampel = torch.mean(sample[:,:self.K],dim=1).view(b,c,h,w)
        return sampel
class WSLCM(nn.Module):
    def __init__(self,scales=[3,5,23,33]):
        super().__init__()
        self.guassian_conv = GaussianFilter()
        self.ordfilters = nn.ModuleList()
        self.samples = nn.ModuleList()
        self.avgpools = nn.ModuleList()
        self.scales = scales
        for scale in scales:
            self.ordfilters.append(OrdFilter(size=scale,K=9))
            self.samples.append(nn.Sequential(
                nn.ReflectionPad2d((scale,scale,scale,scale)),
                nn.Unfold(kernel_size=3,dilation=scale)
            ))
            self.avgpools.append(
                nn.Sequential(
                    nn.ReflectionPad2d((scale//2,scale//2,scale//2,scale//2)),
                    nn.AvgPool2d(kernel_size=scale,stride=1)
                )
            )
    def forward(self,inp):
        inp = inp*255.
        guassian_inp = self.guassian_conv(inp)+1
        WSLCMs = []
        b,c,h,w = inp.size()
        for i,scale in enumerate(self.scales):
            M = self.ordfilters[i](inp)
            BE = self.samples[i](M)
            BE = torch.max(BE[:,[0,1,2,3,5,6,7,8]],dim=1).values.view(b,c,h,w)
            SLCM = torch.clip((BE/guassian_inp-1.)*guassian_inp,min=0)
            IRIL = M - self.avgpools[i](inp)
            IRIL = self.samples[i](IRIL)
            WT = IRIL[:,4]
            WD = torch.clip(torch.min(WT - IRIL[:,[0,1,2,3,5,6,7,8]],dim=1).values,min=0)
            WB = torch.clip(torch.std(IRIL[:,[0,1,2,3,5,6,7,8]],dim=1),min=5)
            W = (WT*WD/WB).view(b,c,h,w)
            WSLCMs.append(SLCM*W)
        WSLCMs = torch.stack(WSLCMs,dim=0)
        output = torch.max(WSLCMs,dim=0).values
        return output

