"""Deformable-attention BEV encoder (single-scale MSDeformAttn implemented with F.grid_sample -> ONNX opset>=16,
TensorRT native; no custom CUDA op)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformAttn(nn.Module):
    def __init__(self, ch, heads=8, points=4):
        super().__init__()
        assert ch % heads == 0
        self.h, self.p, self.c = heads, points, ch
        self.offset = nn.Linear(ch, heads * points * 2)
        self.attn = nn.Linear(ch, heads * points)
        self.value = nn.Linear(ch, ch)
        self.out = nn.Linear(ch, ch)
        self._init()

    def _init(self):
        nn.init.zeros_(self.offset.weight)
        th = torch.arange(self.h, dtype=torch.float32) * (2 * math.pi / self.h)
        g = torch.stack([th.cos(), th.sin()], -1)
        g = (g / g.abs().max(-1, keepdim=True).values).view(self.h, 1, 2).repeat(1, self.p, 1)
        for i in range(self.p):
            g[:, i] *= i + 1
        with torch.no_grad():
            self.offset.bias.copy_(g.reshape(-1))
        nn.init.zeros_(self.attn.weight); nn.init.zeros_(self.attn.bias)
        nn.init.xavier_uniform_(self.value.weight); nn.init.zeros_(self.value.bias)
        nn.init.xavier_uniform_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, q, ref, v2d):
        """q [B,Q,C], ref [B,Q,2] in [0,1] (x,y), v2d [B,C,H,W] -> [B,Q,C]"""
        B, Q, C = q.shape
        H, W = v2d.shape[-2:]
        h, p, d = self.h, self.p, C // self.h
        v = self.value(v2d.flatten(2).transpose(1, 2)).view(B, H * W, h, d)
        v = v.permute(0, 2, 3, 1).reshape(B * h, d, H, W)
        off = self.offset(q).view(B, Q, h, p, 2) / q.new_tensor([W, H])
        a = self.attn(q).view(B, Q, h, p).softmax(-1)
        loc = ref[:, :, None, None, :] + off
        grid = (2 * loc - 1).permute(0, 2, 1, 3, 4).reshape(B * h, Q, p, 2)
        s = F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros", align_corners=False)  # [Bh,d,Q,p]
        s = (s * a.permute(0, 2, 1, 3).reshape(B * h, 1, Q, p)).sum(-1)
        s = s.view(B, h, d, Q).permute(0, 3, 1, 2).reshape(B, Q, C)
        return self.out(s)


class DeformLayer(nn.Module):
    def __init__(self, ch, heads, points, ffn):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(ch), nn.LayerNorm(ch)
        self.attn = DeformAttn(ch, heads, points)
        self.ffn = nn.Sequential(nn.Linear(ch, ffn), nn.GELU(), nn.Linear(ffn, ch))

    def forward(self, x, ref, H, W):
        B, Q, C = x.shape
        q = self.n1(x)
        x = x + self.attn(q, ref, q.transpose(1, 2).reshape(B, C, H, W))
        return x + self.ffn(self.n2(x))


class DeformBEVEncoder(nn.Module):
    def __init__(self, ch, bev_size, layers=2, heads=8, points=4, ffn=256):
        super().__init__()
        Y, X = bev_size
        self.Y, self.X = Y, X
        self.layers = nn.ModuleList([DeformLayer(ch, heads, points, ffn) for _ in range(layers)])
        self.pos = nn.Parameter(torch.zeros(1, ch, Y, X))
        nn.init.trunc_normal_(self.pos, std=0.02)
        ys, xs = torch.meshgrid(torch.arange(Y), torch.arange(X), indexing="ij")
        ref = torch.stack([(xs.flatten() + 0.5) / X, (ys.flatten() + 0.5) / Y], -1)
        self.register_buffer("ref", ref[None], persistent=False)

    def forward(self, bev):
        if len(self.layers) == 0:
            return bev
        B, C = bev.shape[:2]
        x = (bev + self.pos).flatten(2).transpose(1, 2)
        ref = self.ref.expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, ref, self.Y, self.X)
        return x.transpose(1, 2).reshape(B, C, self.Y, self.X)
