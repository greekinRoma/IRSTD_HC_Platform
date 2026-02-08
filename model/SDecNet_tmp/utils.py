from torch import nn
from .AttentionModule import *
from .AttentionModule.nonlocal_module import _NonLocalBlockND
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()
class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU',kernel_size=3):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding='same')
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)
def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)
class GFEM(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.down = nn.MaxPool2d((2,2))
        self.up = nn.Upsample(scale_factor=2,mode='bilinear')
        self.ca = ChannelAttention(in_planes=channels)
        self.sp = _NonLocalBlockND(in_channels=channels,inter_channels=channels)
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