from torch import nn
import torch
import os
from loss import SoftIoULoss, ISNetLoss
from model import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Net(nn.Module):
    def __init__(self, model_name, mode='test',size=256):
        super(Net, self).__init__()

        self.model_name = model_name
        self.softiou_loss = SoftIoULoss()
        self.mse_loss = torch.nn.MSELoss()
        self.model = Algorithms()
        self.is_alg =False
        self.model_name = model_name
        if self.model.detect(model_name):
            self.model.set_algorithm(model_name)
            self.is_alg = True
        elif model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'AGPCNet':
            self.model = AGPCNet()
        elif model_name == 'UIUNet':
            if mode == 'train':
                self.model = UIUNet(mode='train')
            else:
                self.model = UIUNet(mode='test')
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN()
        elif model_name == 'ISTDU_Net':
            self.model = ISTDU_Net()
        elif model_name == 'DATransNet':
            self.model = DATransNet(img_size=size)
        elif model_name == 'SDiffFormer':
            self.model = SDiffFormer(img_size=size)
        elif model_name == 'res_UNet':
            self.model = res_UNet()
        elif model_name == 'L2SKNet':
            self.model = L2SKNet_UNet()
        elif model_name == 'MSHNet':
            self.model = MSHNet(input_channels=1)
        elif model_name == 'SDecNet':
            self.model = SDecNet()
        elif model_name == 'SCTransNet':
            self.model = SCTransNet()
        elif model_name == "HDNet":
            self.model = HDNet(input_channels=1)
        elif model_name == "RPCANet":
            self.model = RPCANet()
    def forward(self, img, mode='train'):
        if hasattr(self.model,'mode'):
            return self.model(img, mode=mode)
        else:
            return self.model(img)

    def loss(self, pred, gt_mask, image):
        if "RPCANet" == self.model_name:
            D, T = pred
            loss =  self.mse(D, image) + self.softiou_loss(T,gt_mask)
        else:
            loss = self.softiou_loss(pred, gt_mask)
        return loss
