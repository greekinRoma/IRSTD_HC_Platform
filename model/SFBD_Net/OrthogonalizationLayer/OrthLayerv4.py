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
        self.constant_polars = nn.Parameter(torch.randn([1, self.num_groups, channels, self.num_polar, width*height]))
        self.weight = nn.Parameter(torch.ones([1, 1, 1, 1]) * math.exp(0.5) * (1 - 1/math.sqrt(2)))
        self.total = math.exp(0.5) * (1 - 1/math.sqrt(2))
        self.act = torch.nn.GELU()

    def ortho(self, X):
        """Orthogonalize a single group of vectors.

        Args:
            X: (B, C, N_group, D) where N_group = group_size + num_polar
        Returns:
            (B, C, N_group, D) with orthogonalized rows
        """
        with torch.no_grad():
            N = X.shape[2]
            I = torch.eye(N, device=X.device).unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
            conf = math.sqrt(2)
            T = X @ X.transpose(-1, -2)                                   # [B, C, N, N]
            X_curr = I - T / N *2
            Y = X_curr
            for i in range(self.L):
                X_curr = X_curr @ X_curr
                Y = Y + X_curr * conf
                conf = conf * math.sqrt(2)
        return Y @ X 
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
        polars = self.constant_polars.repeat(B, 1, 1, 1, 1).reshape(B * self.num_groups, C, self.num_polar, D)
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
        print(x@x.transpose(-1, -2))
        return x