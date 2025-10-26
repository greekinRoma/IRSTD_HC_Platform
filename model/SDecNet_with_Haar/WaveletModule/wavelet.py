from pytorch_wavelets import DWTForward
import torch
from torch import nn
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, out_ch, kernel_size=1, stride=1)
                                    ) 
    def forward(self, x):
        _, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([y_HL, y_LH, y_HH], dim=1)        
        x = self.conv_bn_relu(x)
        return x

