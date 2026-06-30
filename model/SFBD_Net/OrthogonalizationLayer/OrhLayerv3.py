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
        d_pos = num_elements//2   # 位置编码维度
        pe = torch.zeros(seq_len, d_pos*2)
        position = torch.arange(0, d_pos).float().unsqueeze(0)
        position = position // self.width + position % self.width
        div_term = 2*math.pi*torch.exp(torch.arange(0, seq_len).float() *
                            -(math.log(10000.0)*2 / seq_len)).unsqueeze(1)
        # div_term = (1/10000.**(torch.arange(seq_len)*2/seq_len)).unsqueeze(1)
        # print(div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def ortho(self, X):
        B, C, N, D = X.shape
        with torch.no_grad():
            
            I_mat = self.eye.to(X.device)                     # [1, 1, N, N]
            I_exp = I_mat.expand(B, C, N, N)                  # [B, C, N, N]

            # ---- Gram matrix ----
            G = X @ X.transpose(-1, -2)                     # [B, C, N, N]

            # ---- Scale + regularise → eigenvalues in (ε, 1] ⊂ (0, 2) ----
            A = G/N                     # ridge for numerical safety

            # ---- Denman-Beavers iteration ----
            Y = A
            Z = I_exp.clone()

            for _ in range(self.L):
                T_mat = (3.0 * I_mat - Z @ Y) / 2.0
                Y_new = Y @ T_mat
                Z_new = T_mat @ Z

                # Numerical guard: break early if diverging
                if torch.isnan(Y_new).any() or torch.isinf(Y_new).any():
                    break
                Y, Z = Y_new, Z_new
            result = Z @ X
        return result
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        inputs= torch.concat([inputs,self.constant_polars.expand(batch_size,-1,-1,-1)],dim=2)
        inputs = torch.nn.functional.normalize(inputs,dim=-1,p=2)
        outputs = self.ortho(inputs)
        # print(outputs.shape)
        outputs = torch.nn.functional.normalize(outputs,dim=-1,p=2)
        return outputs