import torch
from torch import nn
from functools import partial
from .modeling import MaskDecoder, PromptEncoder, TwoWayTransformer, sam
from .modeling.IRSAM_decoder import MaskDecoder as EdgeDecoder
from .modeling.IRSAM_encoder import TinyViT as EdgeEncoder
from .modeling.IRSAM_edge import Sam as EdgeIRSAM
class IRSAM(nn.Module):
    def __init__(self, prompt_embed_dim=256, image_size=1024, vit_patch_size=16):
        super().__init__()
        self.prompt_embed_dim = prompt_embed_dim
        self.img_size = image_size
        self.vit_patch_size = vit_patch_size
        self.image_embedding_size = image_size//vit_patch_size
        self.mobile_sam = EdgeIRSAM(
            image_encoder=EdgeEncoder(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=EdgeDecoder(
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )