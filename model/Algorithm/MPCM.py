import torch
import torch.nn as nn
import torch.nn.functional as F
class MPCM(nn.Module):
    def __init__(self):
        super(MPCM, self).__init__()
    def localmean(self, img, mask):
        """
        Applies local mean filter using a given mask.
        The input is a PyTorch tensor of shape (b, c, w, h).
        Returns a tensor of the same shape after applying the filter.
        """
        # For simplicity, assume the image tensor is already in (b, c, w, h) format
        b, c, w, h = img.shape
        # The local mean operation can be done using convolution with the mask
        return F.conv2d(img, mask.unsqueeze(0).unsqueeze(0), padding=(mask.shape[0] // 2, mask.shape[1] // 2))

    def MPCM_fun_2(self, img):
        """
        Rewriting the MPCM algorithm to use the 8 filters mentioned in the paper.
        The input is assumed to be a PyTorch tensor of shape (b, c, w, h).
        The output will be of shape (b, c, h, w).
        """

        # Get the batch size, channels, width, and height
        b, c, w, h = img.shape

        # Create masks of sizes 3, 5, 7, and 9
        mask3 = torch.ones((3, 3)).cuda()/9
        mask5 = torch.ones((5, 5)).cuda()/25
        mask7 = torch.ones((7, 7)).cuda()/49
        mask9 = torch.ones((9, 9)).cuda()/81

        # Step 1: Perform local mean filtering using different masks
        l3 = self.localmean(img, mask3)
        l5 = self.localmean(img, mask5)
        l7 = self.localmean(img, mask7)
        l9 = self.localmean(img, mask9)

        # Step 2: Create the 8 filters and apply them to the mean images
        # Create the filters (m31, m32, ..., m38)
        filters = [
            torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 0]]).cuda().float(),
            torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().float(),
            torch.tensor([[0, 0, -1], [0, 1, 0], [0, 0, 0]]).cuda().float(),
            torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().float(),
            torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().float(),
            torch.tensor([[0, 0, 0], [0, 1, 0], [-1, 0, 0]]).cuda().float(),
            torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().float(),
            torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, -1]]).cuda().float()
        ]
        
        PCM3 = torch.zeros((b, c, w, h, 8))
        PCM5 = torch.zeros((b, c, w, h, 8))
        PCM7 = torch.zeros((b, c, w, h, 8))
        PCM9 = torch.zeros((b, c, w, h, 8))
        
        for i in range(8):
            PCM3[..., i] = F.conv2d(l3, filters[i].unsqueeze(0).unsqueeze(0), padding=3,dilation=3)
            PCM5[..., i] = F.conv2d(l5, filters[i].unsqueeze(0).unsqueeze(0), padding=5,dilation=5)
            PCM7[..., i] = F.conv2d(l7, filters[i].unsqueeze(0).unsqueeze(0), padding=7,dilation=7)
            PCM9[..., i] = F.conv2d(l9, filters[i].unsqueeze(0).unsqueeze(0), padding=9,dilation=9)

        # Apply element-wise multiplication for PCM3
        temp3 = PCM3[..., [[0, 4], [1, 5], [2, 6], [3, 7]]].prod(dim=-1)
        temp5 = PCM5[..., [[0, 4], [1, 5], [2, 6], [3, 7]]].prod(dim=-1)
        temp7 = PCM7[..., [[0, 4], [1, 5], [2, 6], [3, 7]]].prod(dim=-1)
        temp9 = PCM9[..., [[0, 4], [1, 5], [2, 6], [3, 7]]].prod(dim=-1)
        out3 = temp3.min(dim=-1)[0]
        out5 = temp5.min(dim=-1)[0]
        out7 = temp7.min(dim=-1)[0]
        out9 = temp9.min(dim=-1)[0]

        # Combine results from all filter sizes (3, 5, 7, 9)
        temp = torch.stack([out3, out5, out7, out9], dim=-1)
        out = temp.max(dim=-1)[0]

        # The output tensor is of shape (b, c, h, w)
        return out
    # Local mean function to be used with different masks

    def forward(self, input_tensor):
        if input_tensor.dim() != 4:
            raise ValueError("Input tensor must be 4D: (batch_size, channels, height, width)")
        
        
        outputs = self.MPCM_fun_2(input_tensor)
        batch_size = outputs.size(0)
        thresholded_outputs = []
        for i in range(batch_size):
            out = outputs[i]  # Shape: (h, w)
            avg = out.mean().item()
            std = out.std().item()
            kth = 3
            th = avg + kth * std
            # Convert to binary using the threshold
            bw = (out > th).float()
            thresholded_outputs.append(bw)
        thresholded_outputs = torch.stack(thresholded_outputs,dim=0)
        return thresholded_outputs