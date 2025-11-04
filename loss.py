from torch import nn
import torch
import os
from loss import SoftIoULoss, ISNetLoss, DiceLoss
from model import *
from utils.loss.IRSAM_loss import SigmoidMetric, SamplewiseSigmoidMetric
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Net(nn.Module):
    def __init__(self, model_name, mode='test',size=256):
        super(Net, self).__init__()

        self.model_name = model_name
        self.softiou_loss = SoftIoULoss()
        self.dice_loss = DiceLoss(reduction='mean')
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.bce_loss = torch.nn.BCELoss(reduction='mean')
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
        elif model_name == "DRPCANet":
            self.model = DRPCANet()
        elif model_name =="RPCANet_plus":
            self.model = RPCANet_LSTM()
        elif model_name == "LRPCANet":
            self.model = LRPCANet()
        elif model_name == "SDecNet_DHPF":
            self.model = SDecNet_DHPF()
        elif model_name == "SDecNet_Haar":
            self.model  = SDecNet_Haar()
        elif model_name == "MiM":
            self.model = MiM([2]*3,[8, 16, 32, 64, 128])
        elif model_name == "VMamba":
            self.model = VMambaSeg()
        elif model_name == "LocalMamba":
            self.model = build_seg_model()
        elif model_name == "IRSAM":
            self.model = build_sam_IRSAM(image_size=size)
    def forward(self, imgs, mode='train'):
        if self.model_name in ["RPCANet", "DRPCANet", "RPCANet_plus", "LRPCANet"]:
            return self.model(imgs, mode=mode)
        elif self.model_name == "IRSAM":
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = imgs[b_i].to(self.model.device)
                dict_input['image'] = input_image
                dict_input['original_size'] = imgs[b_i].shape[2:]
                batched_input.append(dict_input)
            if mode == "train":
                masks, edges = self.model(batched_input)
                return edges.sigmoid(), masks.sigmoid()
            else:
                masks, edges = self.model(batched_input)
                return masks.sigmoid()
        else:
            return self.model(imgs)

    def loss(self, pred, gt_mask, image):
        if "RPCANet" == self.model_name:
            D, T = pred
            loss =  self.mse_loss(D, image) * 0.01 + self.softiou_loss(T,gt_mask)
        elif self.model_name == "DRPCANet":
            D, T = pred
            loss =  self.mse_loss(D, image) * 0.1 + self.softiou_loss(T,gt_mask)
        elif self.model_name == "RPCANet_plus":
            D, T = pred
            loss =  self.mse_loss(D, image) * 0.1 + self.softiou_loss(T,gt_mask)
        elif self.model_name == "LRPCANet":
            D, T = pred
            loss =  self.mse_loss(D, image) * 0.1 + self.softiou_loss(T,gt_mask)
        elif self.model_name == "IRSAM":
            edges, masks = pred
            # print( self.bce_loss(edges, gt_mask))
            # print( self.dice_loss(preds=masks, gt_masks=gt_mask))
            loss = self.bce_loss(edges, gt_mask) * 10. + self.dice_loss(preds=masks, gt_masks=gt_mask)
        else:
            loss = self.softiou_loss(pred, gt_mask)
        return loss
