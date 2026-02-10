import torch
import torch.nn as nn
import torch.nn.functional as F
class TopHat(nn.Module):
    def __init__(self):
        super(TopHat, self).__init__()

    def forward(self, input_tensor):
        if input_tensor.dim() != 4:
            raise ValueError("Input tensor must be 4D: (batch_size, channels, height, width)")
        
        
        dilated = F.max_pool2d(input_tensor, kernel_size=3, stride=1, padding=1)
        
        eroded = -F.max_pool2d(-input_tensor, kernel_size=3, stride=1, padding=1)
        
        top_hat_output = dilated - eroded
        top_hat_output = top_hat_output/top_hat_output.max()
        return top_hat_output