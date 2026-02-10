# localmamba_seg.py
import sys
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from model.LocalMamba.backbone import Backbone_LocalVSSM
import os
def _make_conv_bn_act(in_ch, out_ch, k=1, s=1, p=0):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class FPN(nn.Module):
    """
    标准 FPN：把 [C2,C3,C4,C5] 自顶向下融合到同一通道数，再输出多尺度拼接/逐级融合特征
    """
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        assert len(in_channels) == 4
        self.lateral = nn.ModuleList([_make_conv_bn_act(c, out_channels, 1, 1, 0) for c in in_channels])
        self.out_conv = nn.ModuleList([_make_conv_bn_act(out_channels, out_channels, 3, 1, 1) for _ in in_channels])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # feats: [C2, C3, C4, C5]
        c2, c3, c4, c5 = feats
        p5 = self.lateral[3](c5)
        p4 = self.lateral[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        p2 = self.out_conv[0](p2)
        p3 = self.out_conv[1](p3)
        p4 = self.out_conv[2](p4)
        p5 = self.out_conv[3](p5)
        return [p2, p3, p4, p5]


class SegHead(nn.Module):
    """
    简洁语义分割 Head：用 FPN 的最高分辨率 p2 预测，辅以多尺度特征聚合
    """
    def __init__(self, fpn_channels=256, num_classes=19, aggr="sum"):
        super().__init__()
        self.aggr = aggr
        self.fuse_p3 = _make_conv_bn_act(256, 256, 3, 1, 1)
        self.fuse_p4 = _make_conv_bn_act(256, 256, 3, 1, 1)
        self.fuse_p5 = _make_conv_bn_act(256, 256, 3, 1, 1)
        self.pred = nn.Conv2d(256, num_classes, 1, 1, 0)

    def forward(self, ps: List[torch.Tensor]) -> torch.Tensor:
        # ps: [p2, p3, p4, p5] with strides [4,8,16,32] (示意)
        p2, p3, p4, p5 = ps
        H, W = p2.shape[-2:]
        p3u = F.interpolate(self.fuse_p3(p3), size=(H, W), mode="bilinear", align_corners=False)
        p4u = F.interpolate(self.fuse_p4(p4), size=(H, W), mode="bilinear", align_corners=False)
        p5u = F.interpolate(self.fuse_p5(p5), size=(H, W), mode="bilinear", align_corners=False)

        if self.aggr == "sum":
            x = p2 + p3u + p4u + p5u
        else:
            x = torch.cat([p2, p3u, p4u, p5u], dim=1)
            x = _make_conv_bn_act(x.shape[1], 256, 3, 1, 1)(x)

        return self.pred(x)


class LocalMambaFPN(nn.Module):
    """
    将 LocalVMamba/LocalVim 作为 backbone（来自 LocalMamba 仓库注册到 timm 的模型）
    然后接 FPN + 简洁 Head，输出语义分割结果。
    """
    def __init__(
        self,
        backbone_name: str = "timm_local_vmamba_small",  # 例如：timm_local_vmamba_tiny / timm_local_vim_tiny
        num_classes: int = 19,
        in_chans: int = 3,
        pretrained: bool = False,
        fpn_out: int = 256,
        localmamba_repo_root: str = "LocalMamba",  # 仓库根目录
    ):
        super().__init__()

        # 1) 把 LocalMamba 的 classification/lib 加入 PYTHONPATH，完成 timm 注册
        lib_dir = Path(localmamba_repo_root) / "classification" / "lib"
        if lib_dir.exists():
            sys.path.insert(0, str(lib_dir))
            # 这一行会触发其 __init__ 里的 timm 注册（不同版本命名可能略有差异）
            try:
                import models  # noqa: F401
            except Exception:
                pass

        # 2) timm 拉取 backbone 的多尺度特征
        # out_indices=(1,2,3,4) 或 (0,1,2,3) 取决于注册实现，下面先尝试常见的 4 个尺度输出
        self.backbone = Backbone_LocalVSSM()

        # 读取各层通道数
        in_channels = [96, 192, 384, 768]
        assert len(in_channels) == 4, f"期望4个尺度特征，实际拿到: {in_channels}"

        # 3) FPN + Head
        self.fpn = FPN(in_channels, out_channels=fpn_out)
        self.head = SegHead(fpn_channels=fpn_out, num_classes=num_classes, aggr="sum")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)              # list of 4 feature maps
        ps = self.fpn(feats)                  # [p2,p3,p4,p5] (p2 分辨率最高)
        logit = self.head(ps)                 # [B, C, H/4, W/4]（示意）
        # 最终上采样回输入大小
        logit = F.interpolate(logit, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logit


def build_seg_model(
    num_classes: int = 1,
    variant: str = "vmamba_s",          # "vmamba_t", "vim_t", "vim_s" 等
    pretrained: bool = False,
    in_chans: int = 1,
    repo_root: str = "LocalMamba",
) -> nn.Module:
    """
    工厂函数：根据 variant 选择具体 backbone 名称，返回可训练的分割模型
    """
    name_map = {
        "vmamba_t": "timm_local_vmamba_tiny",
        "vmamba_s": "timm_local_vmamba_small",
        "vim_t":    "timm_local_vim_tiny",
        "vim_s":    "timm_local_vim_small",
    }
    backbone_name = name_map.get(variant.lower(), "timm_local_vmamba_small")
    return LocalMambaFPN(
        backbone_name=backbone_name,
        num_classes=num_classes,
        in_chans=in_chans,
        pretrained=pretrained,
        localmamba_repo_root=repo_root,
    )


if __name__ == "__main__":
    # 简单自测
    model = build_seg_model(num_classes=6, variant="vmamba_s", pretrained=False, repo_root="LocalMamba")
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print("out:", y.shape)  # [1, 6, 512, 512]
