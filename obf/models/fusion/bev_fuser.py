import torch.nn as nn

from ..layers import conv_bn_relu


class ConvFuser(nn.Module):
    """Channel-concat of modality BEV maps -> 3 conv layers (BEVFusion-style)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(conv_bn_relu(in_ch, out_ch), conv_bn_relu(out_ch, out_ch), conv_bn_relu(out_ch, out_ch))

    def forward(self, x):
        return self.net(x)
