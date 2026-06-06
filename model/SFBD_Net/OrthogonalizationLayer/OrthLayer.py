import torch 
from torch import nn
import math
class OrthLayer(nn.Module):
    def __init__(self, width, height, channels, N, L=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = width
        self.height = height
        self.num_polar = 1
        self.N = N+self.num_polar
        self.L = L
        self.constant_polars = nn.Parameter(torch.randn([1,channels,self.num_polar,width*height]))
        self.weight = nn.Parameter(torch.ones([1,1,1,1])*math.exp(0.5)*(1-1/math.sqrt(2)))
        self.eye = torch.eye(self.N).unsqueeze(0).unsqueeze(0)
        self.total = math.exp(0.5)*(1-1/math.sqrt(2))
        self.act = torch.nn.GELU()

    def ortho(self, X):
        with torch.no_grad():
            I = self.eye.to(X.device)
            conf = math.sqrt(2)
            T = X @ X.transpose(-1, -2)
            X_curr = I - T / self.N
            Y = X_curr
            for i in range(self.L):
                X_curr = X_curr @ X_curr
                Y = Y + X_curr * conf
                conf = conf * math.sqrt(2)
        return Y @ X*torch.nn.functional.leaky_relu(self.weight) + X
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        inputs= torch.concat([inputs,self.constant_polars.expand(batch_size,-1,-1,-1)],dim=2)
        inputs = torch.nn.functional.normalize(inputs,dim=-1,p=2)
        outputs = self.ortho(inputs)[:,:,:(-self.num_polar)]
        outputs = torch.nn.functional.normalize(outputs,dim=-1,p=2)
        return outputs