import torch
from torch import nn
from torch import functional as F
import math
def mat2gray(input_tensor):
    min_val = torch.min(input_tensor)
    max_val = torch.max(input_tensor)
    
    # 添加一个小常数，避免除以零
    epsilon = torch.finfo(input_tensor.dtype).eps
    range_val = max_val - min_val + epsilon
    
    output_tensor = (input_tensor - min_val) / range_val
    return output_tensor
class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=3, sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self._get_gaussian_kernel(kernel_size, sigma)
        self.kernel = self.kernel.unsqueeze(0).unsqueeze(0).cuda()  # Add batch and channel dimensions
        self.padding = nn.ReflectionPad2d((kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2))
    
    def _get_gaussian_kernel(self, kernel_size, sigma):
        """Generates a Gaussian kernel using the MATLAB-style formula."""
        kernel = torch.arange(kernel_size).float() - kernel_size // 2
        x, y = torch.meshgrid(kernel, kernel)
        
        # MATLAB-style Gaussian function
        gaussian_kernel = (-1 / (math.pi * sigma)) * torch.exp(-(x**2 + y**2) / sigma)
        
        # Normalize the kernel
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        
        return gaussian_kernel

    def forward(self, input_image):
        kernel = self.kernel.expand(input_image.shape[1], -1, -1, -1)  # Expand for all input channels
        input_image = self.padding(input_image)
        output_image = F.conv2d(input_image, kernel, groups=input_image.shape[1])
        return output_image
class gen_patch_ten(nn.Module):
    def __init__(self,patch_size,slideStep):
        super().__init__()
        self.patch_size = patch_size
        self.slideStep = slideStep
        self.lambdaL = 0.7
    def forward(self,inp):
        _,_,imgHei,imgWid = inp.size()
        rowPatchNum = math.ceil((imgHei - self.patch_size) / self.slideStep) + 1
        colPatchNum = math.ceil((imgWid - self.patch_size) / self.slideStep) + 1
        pad_h = rowPatchNum * self.patch_size - imgHei
        pad_w = colPatchNum * self.patch_size - imgWid
        unfolding_output = torch.nn.functional.unfold(input=inp,kernel_size=self.patch_size,stride=self.slideStep,padding=(pad_h//2,pad_w//2,pad_h-pad_h//2,pad_w-pad_w//2))
        return unfolding_output
class PSTNN(nn.Module):
    def __init__(self,patch_size=40,slideStep=20):
        super().__init__()
        self.patch_size = patch_size
        self.slideStep = slideStep
        self.guassian_conv_1 = GaussianFilter(kernel_size=patch_size,sigma=2.)
        self.guassian_conv_2 = GaussianFilter(kernel_size=patch_size,sigma=9.)
        self.gen_patch_ten = gen_patch_ten(patch_size=patch_size,slideStep=slideStep)
        self.lambdaL = 0.7
    def get_gradient(self,img):
        Gx = (img[:,:,:,2:] - img[:,:,:,:-2])/2
        Gx = torch.concat([img[:,:,:,1:2]-img[:,:,:,0:1],Gx,img[:,:,:,-1:]-img[:,:,:,-2:-1]],dim=-1)
        Gy = (img[:,:,2:,:] - img[:,:,:-2,:])/2
        Gy = torch.concat([img[:,:,1:2,:]-img[:,:,0:1,:],Gy,img[:,:,-1:,:]-img[:,:,-2:-1,:]],dim=-2)
        return Gx, Gy
    def structure_tensor_lambda(self,img):
        img = self.guassian_conv_1(img)
        Gx,Gy = self.get_gradient(img=img)
        J_11 = self.guassian_conv_2(Gx*Gx)
        J_12 = self.guassian_conv_2(Gx*Gy)
        J_22 = self.guassian_conv_2(Gy*Gy)
        sqrt_delta = torch.sqrt((J_11 - J_22)^2 + 4*J_12^2)
        lambda_1 = 0.5*(J_11 + J_22 + sqrt_delta)
        lambda_2 = 0.5*(J_11 + J_22 - sqrt_delta)
        return lambda_1,lambda_2
    def forward(self,inp):
        b,c,h,w = inp.size()
        tenD = gen_patch_ten(inp)
        lambda_1,lambda_2 = self.structure_tensor_lambda(inp)
        cornerStrength = (((lambda_1*lambda_2)/(lambda_1 + lambda_2 + 1e-5)))
        maxValue = (torch.maximum(lambda_1,lambda_2))
        priorWeight = mat2gray(cornerStrength * maxValue)
        tenW = self.gen_patch_ten(priorWeight)
        
        return inp