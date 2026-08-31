import torch.nn as nn
import torch.nn.functional as F

from ..layers import conv_bn_relu


class BEVSegHead(nn.Module):
    """Multi-label (nuScenes: drivable/vehicle/pedestrian, BCE+Dice) or single-label (GOOSE, CE) BEV segmentation."""

    def __init__(self, in_ch, classes, multilabel=True):
        super().__init__()
        self.multilabel = multilabel
        self.net = nn.Sequential(conv_bn_relu(in_ch, 128), conv_bn_relu(128, 64), nn.Conv2d(64, len(classes), 1))

    def forward(self, x):
        return {"seg": self.net(x)}

    def loss(self, logit, target):
        logit = logit.float()
        if not self.multilabel:
            return {"seg": F.cross_entropy(logit, target.long(), ignore_index=255)}
        bce = F.binary_cross_entropy_with_logits(logit, target)
        p = logit.sigmoid()
        dice = 1 - (2 * (p * target).sum((0, 2, 3)) + 1) / ((p + target).sum((0, 2, 3)) + 1)
        return {"seg": bce + dice.mean()}


class OccHead(nn.Module):
    """Dense 3D occupancy from BEV: predicts Z*C logits per cell -> [B,C,Z,Y,X] (Occ3D-nuScenes grid, 18 classes)."""

    def __init__(self, in_ch, classes, Z):
        super().__init__()
        self.C, self.Z = classes, Z
        self.net = nn.Sequential(conv_bn_relu(in_ch, 256), conv_bn_relu(256, 256), nn.Conv2d(256, Z * classes, 1))

    def forward(self, x):
        B, _, Y, X = x.shape
        o = self.net(x).view(B, self.Z, self.C, Y, X).permute(0, 2, 1, 3, 4)
        return {"occ": o}

    def loss(self, logit, target):  # target [B,Y,X,Z] long, 255 = ignore
        return {"occ": F.cross_entropy(logit.float(), target.permute(0, 3, 1, 2), ignore_index=255)}
