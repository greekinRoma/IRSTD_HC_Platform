from torch import nn
import torch
from torch.nn import Conv2d
class CELayer(nn.Module):
    def __init__(self,n_channel,down_ratio,feature_size):
        super().__init__()
        n_patch_len = (feature_size//down_ratio)
        self.patch_embeddings = Conv2d(in_channels=n_channel,
                                       out_channels=n_channel,
                                       kernel_size=down_ratio,
                                       stride=down_ratio)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_channel, n_patch_len, n_patch_len))

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        embeddings = x + self.position_embeddings
        return embeddings