import torch
import torch.nn as nn
import torch.nn.functional as F

# 请把这行换成你本地的导入路径：
# from segmentation.models.backbones.vmamba import VSSM   # 可能的路径/类名
from model.VMamba.vmamba import VSSM, Backbone_VSSM
class FPNConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.proj(x)))

class SimpleUPerHead(nn.Module):
    """极简 UPer 风格：P2,P3,P4,P5 -> 融合 -> 分割Logits"""
    def __init__(self, in_channels, ppm_channels=256, num_classes=150):
        super().__init__()
        assert len(in_channels) == 4
        self.lateral = nn.ModuleList([FPNConv(c, ppm_channels) for c in in_channels])

        # 简化版的 Pyramid Pooling（四种池化尺寸）
        self.ppm_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveAvgPool2d(2),
            nn.AdaptiveAvgPool2d(3),
            nn.AdaptiveAvgPool2d(6),
        ])
        self.ppm_convs = nn.ModuleList([nn.Conv2d(in_channels[-1], ppm_channels, 1, bias=False)
                                        for _ in range(len(self.ppm_pools))])

        self.fuse = nn.Conv2d(ppm_channels * (len(self.ppm_pools) + 4), ppm_channels, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(ppm_channels)
        self.act  = nn.ReLU(inplace=True)
        self.cls  = nn.Conv2d(ppm_channels, num_classes, 1)

    def forward(self, feats):
        # feats: [C2, C3, C4, C5] 分辨率依次 /4,/8,/16,/32
        C2, C3, C4, C5 = feats
        h, w = C2.shape[-2:]
        laterals = [F.interpolate(self.lateral[i](f), size=(h, w), mode="bilinear", align_corners=False)
                    for i, f in enumerate([C2, C3, C4, C5])]

        # PPM 在 C5 上做
        ppm_list = []
        for pool, conv in zip(self.ppm_pools, self.ppm_convs):
            p = pool(C5)
            p = conv(p)
            p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
            ppm_list.append(p)

        x = torch.cat(laterals + ppm_list, dim=1)
        x = self.act(self.bn(self.fuse(x)))
        return self.cls(x)  # [B, num_classes, H/4, W/4]

class VMambaSeg(nn.Module):
    def __init__(self, num_classes=1, variant="tiny"):
        super().__init__()
        # 典型四阶段通道数（按常见设定给默认值，可根据仓库配置改）
        variant_dims = {
            "tiny":  [64, 128, 256, 512],
            "small": [96, 192, 384, 768],
            "base":  [128, 256, 512, 1024],
        }
        dims = variant_dims.get(variant, variant_dims["tiny"])

        # 假设 VSSM/VMamba 主干支持返回四层金字塔特征
        self.backbone = Backbone_VSSM()
        self.decode_head = SimpleUPerHead(in_channels=[64, 128, 256, 512], num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)          # List[4]: [B, C2, H/4,W/4], ... [B, C5, H/32,W/32]
        logits = self.decode_head(feats)  # [B, num_classes, H/4, W/4]
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits.sigmoid()

def build_vmamba_seg(num_classes=150, variant="tiny"):
    return VMambaSeg(num_classes=num_classes, variant=variant)

if __name__ == "__main__":
    model = build_vmamba_seg(num_classes=19, variant="tiny")
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(y.shape)  # -> [2, 19, 512, 512]
