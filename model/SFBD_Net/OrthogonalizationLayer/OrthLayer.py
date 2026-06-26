import torch 
from torch import nn
import math
class OrthLayer(nn.Module):
    def __init__(self, width, height, channels, N, L=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = width
        self.height = height
        self.num_polar = 1
        self.N = N+self.num_polar
        self.L = L
        self.constant_polars = nn.Parameter(self.get_positional_encoding_half(seq_len=channels,num_elements=width*height).unsqueeze(0).unsqueeze(2).expand(-1,-1,self.num_polar,-1),requires_grad=False)
        self.weight = nn.Parameter(torch.ones([1,1,1,1])*math.exp(0.5)*(math.sqrt(2)-1.))
        self.eye = torch.eye(self.N).unsqueeze(0).unsqueeze(0)
        self.total = math.exp(0.5)*(math.sqrt(2)-1.)
        self.act = torch.nn.GELU()
    
    def get_positional_encoding_half(self, seq_len, num_elements):
        d_pos = num_elements   # 位置编码维度
        pe = torch.zeros(seq_len, d_pos)
        position = torch.arange(0, d_pos).float().unsqueeze(0)
        position = position // self.width + position % self.width
        div_term = 2*math.pi*torch.exp(torch.arange(0, seq_len).float() *
                            -(math.log(10000.0)*2 / seq_len)).unsqueeze(1)
        # div_term = (1/10000.**(torch.arange(seq_len)*2/seq_len)).unsqueeze(1)
        # print(div_term)
        # pe[:, 0::2] = torch.sin(position * div_term)
        pe = torch.cos(position * div_term)
        return pe

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
        return Y @ X*self.total + X
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        inputs= torch.concat([inputs,self.constant_polars.expand(batch_size,-1,-1,-1)],dim=2)
        inputs = torch.nn.functional.normalize(inputs,dim=-1,p=2)
        outputs = self.ortho(inputs)[:,:,:self.N-self.num_polar,:]
        outputs = torch.nn.functional.normalize(outputs,dim=-1,p=2)
        # print(outputs@outputs.transpose(-1,-2))
        return outputs