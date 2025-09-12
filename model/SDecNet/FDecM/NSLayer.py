from torch import nn
import torch
class NSLayer(nn.Module):
    def __init__(self,channel,kernel=16):
        super(NSLayer, self).__init__()
        self.basis_scale = torch.tensor([ 4.9234e-01,  4.5757e-01,  7.8983e-01, -1.0768e+00, -1.5491e+00,
        -2.1980e+00, -3.2622e+00, -3.6207e+00, -9.4038e+00, -1.1254e-03,
        -2.6745e+01,  1.7235e+01, -5.1393e-03, -7.1784e-03, -2.8206e-01]).view(15,1,1,1,1)
        self.weight = nn.Parameter(torch.rand(14,1,channel,1,1), requires_grad=False)
        # self.weight = nn.Parameter(torch.rand(14,1,channel,1,1), requires_grad=True)
        self.mask = nn.Parameter(torch.eye(kernel).reshape(1,1,kernel,kernel),requires_grad=False)
    def forward(self, input):
        # print(torch.diag(torch.matmul(input, torch.transpose(input, 2, 3))[0,0]))
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
        weight = torch.abs(self.weight)
        Mat =  weight[0]*A + weight[1] * B + weight[2]*C +weight[3]*D+  weight[4] * E+  weight[5] * F + weight[6] * G + weight[7] * H + weight[8] * I + weight[9] * J + weight[10] * K +weight[11] * L + weight[12] * M + weight[13] * N
        out = input + torch.matmul(Mat, input)
        return out