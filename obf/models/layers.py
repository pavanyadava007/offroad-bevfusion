import torch
import torch.nn as nn


def conv_bn_relu(i, o, k=3, s=1, p=None):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2 if p is None else p, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class BEVBackbone(nn.Module):
    """SECOND-style 2-stage 2D backbone; output stride 2 w.r.t. the input canvas (400 -> 200)."""

    def __init__(self, in_ch, out_ch=128, chs=(64, 128), n=3):
        super().__init__()
        self.s1 = nn.Sequential(conv_bn_relu(in_ch, chs[0], s=2), *[conv_bn_relu(chs[0], chs[0]) for _ in range(n)])
        self.s2 = nn.Sequential(conv_bn_relu(chs[0], chs[1], s=2), *[conv_bn_relu(chs[1], chs[1]) for _ in range(n)])
        self.up2 = nn.Sequential(nn.ConvTranspose2d(chs[1], chs[1], 2, 2, bias=False), nn.BatchNorm2d(chs[1]), nn.ReLU(True))
        self.out = conv_bn_relu(chs[0] + chs[1], out_ch, k=1)

    def forward(self, x):
        a = self.s1(x)
        b = self.up2(self.s2(a))
        if b.shape[-2:] != a.shape[-2:]:  # odd canvas sizes
            b = nn.functional.pad(b, (0, a.shape[-1] - b.shape[-1], 0, a.shape[-2] - b.shape[-2]))
        return self.out(torch.cat([a, b], 1))
