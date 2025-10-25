import argparse
from net import Net
import os
import time
from thop import profile
import torch
def calculate_fps(model, num_frames=100, input_size=(1, 1, 256, 256)):
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create a random input tensor with the specified input size
    inputs = torch.randn(*input_size).to(device)

    # Measure time for processing 'num_frames' frames
    start_time = time.time()
    for _ in range(num_frames):
        with torch.no_grad():
            _ = model(inputs)
    
    end_time = time.time()
    
    # Calculate the FPS
    total_time = end_time - start_time
    fps = num_frames / total_time
    
    return fps
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
model_name = 'DRPCANet'
input_img = torch.rand(1,1,256,256).cuda()
net = Net(model_name, mode='test').cuda()


net.eval()
output = net(input_img)
# 获取输出图像大小
output_size = output.size()
print('Output Image Size:', output_size)
fps = calculate_fps(net)
print(f"FPS: {fps:.2f}")

flops, params = profile(net, inputs=(input_img, ))
print(model_name)
print('Params: %2fM' % (params/1e6))
print('FLOPs: %2fGFLOPs' % (flops/1e9))