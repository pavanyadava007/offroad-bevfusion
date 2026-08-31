"""Lift-Splat-Shoot camera branch. Backbone -> depth distribution (D) x context (C) -> scatter to BEV via
precomputed frustum->BEV indices (computed in the data pipeline; see obf/data/lss_geometry.py)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CameraEncoder(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, D=44, C=80):
        super().__init__()
        rn = getattr(torchvision.models, backbone)(weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1, rn.layer2)  # stride 8
        self.l3, self.l4 = rn.layer3, rn.layer4  # stride 16, 32
        c3, c4 = (256, 512) if backbone in ("resnet18", "resnet34") else (1024, 2048)
        self.up = nn.Sequential(nn.Conv2d(c3 + c4, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
                                nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True))
        self.depthnet = nn.Conv2d(256, D + C, 1)
        self.D, self.C = D, C

    def forward(self, x):  # [BN,3,H,W] -> depth [BN,D,fH,fW] (softmax), ctx [BN,C,fH,fW]
        x = self.stem(x)
        x3 = self.l3(x)
        x4 = F.interpolate(self.l4(x3), size=x3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.depthnet(self.up(torch.cat([x3, x4], 1)))
        return x[:, : self.D].softmax(1), x[:, self.D:]


class LSSViewTransform(nn.Module):
    def __init__(self, bev_size):
        super().__init__()
        self.Y, self.X = bev_size

    def forward(self, depth, ctx, bev_idx, valid):
        """depth [B,N,D,fH,fW], ctx [B,N,C,fH,fW], bev_idx [B,M] long, valid [B,M] bool/float -> [B,C,Y,X]"""
        B, N, D, fH, fW = depth.shape
        C = ctx.shape[2]
        vol = depth.unsqueeze(2) * ctx.unsqueeze(3)  # [B,N,C,D,fH,fW]
        vol = vol.permute(0, 1, 3, 4, 5, 2).reshape(B, N * D * fH * fW, C)
        vol = vol * valid.unsqueeze(-1).to(vol.dtype)
        canvas = vol.new_zeros(B, self.Y * self.X, C)
        canvas = canvas.scatter_add(1, bev_idx.unsqueeze(-1).expand(-1, -1, C), vol)
        return canvas.view(B, self.Y, self.X, C).permute(0, 3, 1, 2).contiguous()
