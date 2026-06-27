from torch import nn
import torch 
class MLP(nn.Module):
    def __init__(self,input_dim=2048,emded_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim,emded_dim)
    def forward(self,x):
        x = x.flatten(2).transpose(1,2)
        x = self.proj(x)
        return x
# class SegFormerHead(nn.Module):
#     def __init__(self,in_channels,embedding_dim,img_size,num_classes) -> None:
#         super(SegFormerHead,self).__init__()
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.img_size = img_size
        
#         self.linear_fuse = nn.Sequential(*[
#             nn.Conv2d(in_channels=embedding_dim*4,out_channels=embedding_dim,kernel_size=1,stride=1),
#             nn.BatchNorm2d(embedding_dim),
#             nn.ReLU()
#         ])
        
#         self.linear_pred = nn.Conv2d(embedding_dim,self.num_classes,kernel_size=1,stride=1)
#     def forward(self,inputs):
#         c1,c2,c3,c4 = inputs
#         n,_,w,h = c4.shape

#         _c = self.linear_fuse(torch.cat([c4, c3, c2, c1], dim=1))

#         x = self.linear_pred(_c)

#         return x
class SegFormerHead(nn.Module):
    def __init__(self,in_channels,embedding_dim,img_size,num_classes) -> None:
        super(SegFormerHead,self).__init__()
        self.num_classes = num_classes
        self.up1 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.up3 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.up0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels[0],out_channels=embedding_dim,kernel_size=1,stride=1)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(in_channels=in_channels[1],out_channels=embedding_dim,kernel_size=1,stride=1)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=4,mode='bilinear'),
            nn.Conv2d(in_channels=in_channels[2],out_channels=embedding_dim,kernel_size=1,stride=1)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(in_channels=in_channels[3],out_channels=embedding_dim,kernel_size=1,stride=1)
        )
        self.linear_fuse = nn.Sequential(*[
            nn.Conv2d(in_channels=embedding_dim*4,out_channels=embedding_dim,kernel_size=1,stride=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU()
        ])
        
        self.linear_pred = nn.Conv2d(embedding_dim,self.num_classes,kernel_size=1,stride=1)
    def forward(self,x0,x1,x2,x3):
        c0 = self.up0(x0)
        c1 = self.up1(x1)
        c2 = self.up2(x2)
        c3 = self.up3(x3)
        _c = self.linear_fuse(torch.cat([ c3, c2, c1,c0], dim=1))

        x = self.linear_pred(_c)

        return x