import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import Get_gradient_nopadding

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1. - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss


class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()

    def forward(self, preds, gt_masks):
        edge_gt = self.grad(gt_masks.clone())

        ### img loss
        loss_img = self.softiou(preds[0], gt_masks)

        ### edge loss
        loss_edge = 10 * self.bce(preds[1], edge_gt) + self.softiou(preds[1].sigmoid(), edge_gt)

        return loss_img + loss_edge

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, reduction='mean', ignore_index=None):
        """
        Dice Loss 优化版
        
        Args:
            smooth: 平滑系数，防止除零
            reduction: 损失缩减方式 'mean' | 'sum' | 'none'
            ignore_index: 需要忽略的标签索引
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, preds, gt_masks):
        # 处理多尺度输出
        if isinstance(preds, (list, tuple)):
            losses = []
            for pred in preds:
                loss = self._compute_dice_loss(pred, gt_masks)
                losses.append(loss)
            
            total_loss = torch.stack(losses).mean()
            return total_loss
        
        # 单尺度输出
        return self._compute_dice_loss(preds, gt_masks)

    def _compute_dice_loss(self, pred, target):
        """
        计算单个预测的 Dice Loss
        """
        # 输入验证
        if pred.shape != target.shape:
            raise ValueError(f"预测值和真实值形状不匹配: pred {pred.shape}, target {target.shape}")
        
        # 处理忽略的索引
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred * mask
            target = target * mask
        
        # 确保预测值在合理范围内（防止数值不稳定）
        pred = torch.clamp(pred, 1e-7, 1.0)
        
        # 展张量以便计算
        if pred.dim() > 2:
            pred = pred.contiguous().view(pred.size(0), -1)
            target = target.contiguous().view(target.size(0), -1)
        
        # 计算交集和并集
        intersection = (pred * target).sum(dim=1)
        pred_sum = pred.sum(dim=1)
        target_sum = target.sum(dim=1)
        
        # 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # 计算损失
        loss = 1.0 - dice
        
        # 应用缩减
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"不支持的缩减方式: {self.reduction}")