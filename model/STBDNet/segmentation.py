import torch
import torch.nn as nn
from .FDecM.SDecF import SDecF
from .FDecM.SDecD import SD2D
from .AttentionModule import *
from .UNet_module.model_UIUNet import *
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
class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2,mode='bilinear')
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.sattn = nn.Sequential(
            nn.Conv2d(in_channels//2,in_channels//2,kernel_size=1),
            nn.Sigmoid())
    def forward(self,d,c,xin):
        d = self.up(d)+xin
        d = self.sattn(c)*d
        x = torch.cat([c, d], dim=1)
        x = self.nConvs(x)
        return x
class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.fca = FCA_Layer(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out
class Head(nn.Module):
    def __init__(self, inpChannel, oupChannel):
        super(Head, self).__init__()
        interChannel = inpChannel//4
        self.head = nn.Sequential(
            nn.Conv2d(inpChannel, interChannel,kernel_size=7, padding=3,groups=interChannel,bias=False),
        )
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels=interChannel+inpChannel,out_channels=oupChannel,kernel_size=1,stride=1,bias=False))
    def forward(self, x):
        return self.out_conv(torch.concat([self.head(x),x],dim=1))
class STBDNet(nn.Module):
    def __init__(self,  n_channels=1, n_classes=1, img_size=512, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = 8
        block = Res_block
        self.inc = RSU7(n_channels,in_channels,in_channels*2,dilation_ratio=1)
        self.down1 = SD2D(dim=in_channels*2)
        self.encoder1 = self._make_layer(block, in_channels * 2, in_channels * 2, 1) 
        self.down2 = SD2D(dim=in_channels*2)
        self.encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 1) 
        self.down3 = SD2D(dim=in_channels*4)
        self.encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)  
        self.down4 = SD2D(dim=in_channels*8)
        self.encoder4 = self._make_layer(block, in_channels * 8,  in_channels * 8, 1)  

        self.Contrast_Layer = SDecF(n_channels=[in_channels*2,in_channels*2,in_channels*4,in_channels*8],
                                    channel_ratios=[2,4,8,16],
                                    down_ratios=[8,4,2,1],
                                    kernel_size=8)

        self.decoder4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.decoder3 = UpBlock_attention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.decoder2 = UpBlock_attention(in_channels * 4, in_channels*2, nb_Conv=2)
        self.decoder1 = UpBlock_attention(in_channels * 4, in_channels*2, nb_Conv=2)
        self.outc = nn.Sequential(RSU7(in_channels*2,in_channels,in_channels,dilation_ratio=1),
                                  Head(inpChannel=in_channels,oupChannel=n_classes))
    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for _ in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        #encoder
        x1 = self.inc(x) 
        x2 = self.encoder1(self.down1(x1)) 
        x3 = self.encoder2(self.down2(x2)) 
        x4 = self.encoder3(self.down3(x3))  
        d5 = self.encoder4(self.down4(x4))  
        # Transfor_layer
        c1, c2, c3, c4 = self.Contrast_Layer(inp0=x1,inp1=x2,inp2=x3,inp3=x4)
        # decoder
        d4 = self.decoder4(d5, c4, x4)
        d3 = self.decoder3(d4, c3, x3)
        d2 = self.decoder2(d3, c2, x2)
        out = self.outc(self.decoder1(d2, c1, x1))
        return out.sigmoid()