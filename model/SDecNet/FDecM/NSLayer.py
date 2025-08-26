from torch import nn
import torch
class NSLayer(nn.Module):
    def __init__(self,channel,kernel=16):
        super(NSLayer, self).__init__()
        self.weight = nn.Parameter(torch.rand(9,1,channel,1,1), requires_grad=True)
        self.mask = nn.Parameter(torch.eye(kernel).reshape(1,1,kernel,kernel),requires_grad=False)
    def forward(self, input):
        A = (self.mask - torch.matmul(input, torch.transpose(input, 2, 3)))
        B = torch.matmul(A, A)
        C = torch.matmul(B, B)
        D = torch.matmul(C, C)
        # E = torch.matmul(D, D)
        # F = torch.matmul(E, E)
        # G = torch.matmul(F, F)
        # H = torch.matmul(G, G)
        weight = self.weight
        Mat = weight[0] * A+ weight[2]*C+weight[3]*D
        # + weight[4]*D + weight[5] * E+ weight[6]*F + weight[7]*G + weight[8] * H
        out = input + torch.matmul(Mat, input)
        return out