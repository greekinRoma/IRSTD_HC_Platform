from torch import nn
import torch
class NSLayer(nn.Module):
    def __init__(self,channel,kernel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = nn.Parameter(torch.ones(1,channel,1,1)*0.75,requires_grad=True).cuda()
        self.b = nn.Parameter(torch.ones(1,channel,1,1)*0.25,requires_grad=True).cuda()
    def forward(self,inp):
        A = torch.matmul(inp,torch.transpose(inp, 2, 3).contiguous())
        B = torch.matmul(A,A)
        Mat = -self.a*A + self.b*B
        out = inp + torch.matmul(Mat,inp)
        return out