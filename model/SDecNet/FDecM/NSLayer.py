from torch import nn
import torch
class NSLayer(nn.Module):
    def __init__(self, kernel=16, channel=8, bias=True):
        super(NSLayer, self).__init__()
        self.scale_layer_1 = nn.Sequential(
            nn.Linear(kernel, kernel*2, bias=bias),
            nn.BatchNorm1d(channel),
            nn.SiLU(),
            nn.Linear(kernel*2, 2, bias=bias),
            nn.Sigmoid()
        )
        self.mask = torch.eye(kernel, kernel,requires_grad=False).reshape(1,1,kernel,kernel)
        self.scale = torch.ones(1, channel, 1, 1)
    def forward(self, input):
        C = torch.matmul(input=input, other=torch.transpose(input,2,3))
        A = self.mask.to(C.device) - C
        B = torch.matmul(A,A)
        weight = self.scale_layer_1(torch.diagonal(C,dim1=2,dim2=3)).unsqueeze(-2)
        Mat = weight[...,0:1]*A + weight[...,1:2]*B
        out = input + torch.matmul(Mat, input)
        return out