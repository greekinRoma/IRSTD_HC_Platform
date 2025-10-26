from torch import nn
from .nonlocal_module import _NonLocalBlockND
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out 
        return self.sigmoid(out)
class GFEM(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.down = nn.MaxPool2d((2,2))
        self.up = nn.Upsample(scale_factor=2,mode='bilinear')
        self.ca = ChannelAttention(in_planes=channels)
        self.sp = _NonLocalBlockND(in_channels=channels,inter_channels=channels//8)
        self.sattn = nn.Sequential(nn.Conv2d(channels,1,kernel_size=1),
                                   nn.Sigmoid())
        self.tra_conv_1 = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1)
        self.tra_conv_2 = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=1,padding=1)
        self.out_conv = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=1,padding=1)
    def forward(self,inps):
        spat = self.sp(inps)
        down = self.down(inps)
        down = self.ca(spat)*down
        down = self.up(down)*self.sattn(inps)
        spat = self.tra_conv_1(spat)
        down = self.tra_conv_2(down)
        out = spat + down
        out = self.out_conv(out)
        return out