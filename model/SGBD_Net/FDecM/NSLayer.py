from torch import nn
import torch
class NSLayer(nn.Module):
    def __init__(self,channel,kernel=16):
        super(NSLayer, self).__init__()
        self.basis_scale = torch.randn(14,1,channel,1,1)
        self.weight = nn.Parameter(self.basis_scale, requires_grad=True)
        self.mask = nn.Parameter(torch.eye(kernel).reshape(1,1,kernel,kernel),requires_grad=False)
    def forward(self, input):
        A = (self.mask - torch.matmul(input, torch.transpose(input, 2, 3)))
        B = torch.matmul(A, A)#**2
        C = torch.matmul(B, B)#**4
        D = torch.matmul(C, C)#**8
        E = torch.matmul(D, D)#**16
        F = torch.matmul(E, E)#**32
        G = torch.matmul(F, F)#**64
        H = torch.matmul(G, G)#**128
        I = torch.matmul(H, H)#**256
        J = torch.matmul(I, I)#**512
        K = torch.matmul(J, J)#**1024
        L = torch.matmul(K, K)#**2048
        M = torch.matmul(L, L)#**4096
        N = torch.matmul(M, M)#**8192
        weight = torch.relu(self.weight)
        Mat =  weight[0]*A + weight[1] * B + weight[2]*C
        out = input + torch.matmul(Mat, input)
        return out