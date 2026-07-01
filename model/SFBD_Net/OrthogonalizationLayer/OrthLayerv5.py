import torch
from torch import nn
import math
class OrthLayer(nn.Module):
    def __init__(self, width, height, channels, N, L=10, num_polar=1, num_groups=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = width
        self.height = height
        self.num_polar = num_polar
        self.num_groups = num_groups
        self.group_size = N // num_groups
        assert N % num_groups == 0, f"N ({N}) must be divisible by num_groups ({num_groups})"
        self.N = N + self.num_polar                     # total N (kept for reference)
        self.N_group = self.group_size + self.num_polar # per-group N after adding polars
        self.L = L
        self.constant_polars = nn.Parameter(torch.randn([1, 1, channels, self.num_polar, width*height]))
        self.weight = nn.Parameter(torch.ones([1, 1, 1, 1]) * math.exp(0.5) * (1 - 1/math.sqrt(2)))
        self.total = math.exp(0.5) * (1 - 1/math.sqrt(2))
        self.act = torch.nn.GELU()

    def ortho(self, X):
        B, C, N, D = X.shape
        with torch.no_grad():
            
            I_mat =  torch.eye(N, device=X.device).unsqueeze(0).unsqueeze(0).to(X.device)                     # [1, 1, N, N]
            I_exp = I_mat.expand(B, C, N, N)                  # [B, C, N, N]

            # ---- Gram matrix ----
            G = X @ X.transpose(-1, -2)                     # [B, C, N, N]

            # ---- Scale + regularise → eigenvalues in (ε, 1] ⊂ (0, 2) ----
            A = G/self.group_size                    # ridge for numerical safety

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
        """Forward pass — orthogonalizes each dilation group independently via reshape.

        Args:
            inputs: (B, C, N_total, D) where N_total = num_groups * group_size
        Returns:
            (B, C, N_total, D)
        """
        B, C, _, D = inputs.shape
        # (B, C, num_groups * group_size, D) -> (B * num_groups, C, group_size, D)
        x = inputs.view(B, C, self.num_groups, self.group_size, D) \
                  .permute(0, 2, 1, 3, 4) \
                  .reshape(B * self.num_groups, C, self.group_size, D)
        # Add per-group polar vectors (different polars for each group)
        polars = self.constant_polars.expand(B, self.num_groups, -1, -1, -1).reshape(B * self.num_groups, C, self.num_polar, D)
        x = torch.concat([x, polars], dim=2)
        x = torch.nn.functional.normalize(x, dim=-1, p=2)
        x = self.ortho(x)
        # Strip polar, normalize
        x = x[:, :, :-self.num_polar]
        x = torch.nn.functional.normalize(x, dim=-1, p=2)
        # (B * num_groups, C, group_size, D) -> (B, C, num_groups * group_size, D)
        x = x.view(B, self.num_groups, C, self.group_size, D) \
             .permute(0, 2, 1, 3, 4) \
             .reshape(B, C, -1, D)
        # print(x@x.transpose(-1, -2))
        return x