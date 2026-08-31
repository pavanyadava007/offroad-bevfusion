"""CenterPoint-style anchor-free 3D detection head on the shared 200x200 BEV (single task group, 10 classes).
reg channels: [dx, dy, z, log w, log l, log h, sin yaw, cos yaw, vx, vy]."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import conv_bn_relu

CODE_WEIGHTS = (1.0,) * 8 + (0.2, 0.2)


def gaussian_radius(h, w, min_overlap=0.1):
    a1, b1, c1 = 1, h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(hm, cx, cy, radius):
    d = 2 * radius + 1
    sigma = d / 6
    m, n = [(s - 1.0) / 2 for s in (d, d)]
    y, x = np.ogrid[-m: m + 1, -n: n + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    H, W = hm.shape
    l, r = min(cx, radius), min(W - cx, radius + 1)
    t, b = min(cy, radius), min(H - cy, radius + 1)
    if r > 0 and b > 0:
        np.maximum(hm[cy - t: cy + b, cx - l: cx + r], g[radius - t: radius + b, radius - l: radius + r],
                   out=hm[cy - t: cy + b, cx - l: cx + r])


def focal_loss(pred, gt):
    pred = pred.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_l = (torch.log(pred) * (1 - pred) ** 2 * pos).sum()
    neg_l = (torch.log(1 - pred) * pred ** 2 * (1 - gt) ** 4 * neg).sum()
    return -(pos_l + neg_l) / pos.sum().clamp(min=1)


class CenterHead(nn.Module):
    def __init__(self, in_ch, classes, pc_range, bev_size, head_ch=64, score_thr=0.1, max_dets=500,
                 min_radius=2, overlap=0.1, max_objs=500):
        super().__init__()
        self.classes = list(classes)
        self.K = len(classes)
        self.pc_range, self.Y, self.X = pc_range, bev_size[0], bev_size[1]
        self.dx = (pc_range[3] - pc_range[0]) / self.X
        self.dy = (pc_range[4] - pc_range[1]) / self.Y
        self.score_thr, self.max_dets, self.min_radius, self.overlap, self.max_objs = score_thr, max_dets, min_radius, overlap, max_objs
        self.shared = conv_bn_relu(in_ch, head_ch)
        self.hm = nn.Conv2d(head_ch, self.K, 3, padding=1)
        self.reg = nn.Conv2d(head_ch, 10, 3, padding=1)
        nn.init.constant_(self.hm.bias, -2.19)
        self.register_buffer("code_w", torch.tensor(CODE_WEIGHTS), persistent=False)

    def forward(self, x):
        x = self.shared(x)
        return {"hm": self.hm(x), "reg": self.reg(x)}

    # ---- targets / loss ---------------------------------------------------------------------------------------
    @torch.no_grad()
    def targets(self, gt_boxes, gt_labels, device):
        B, M = len(gt_boxes), self.max_objs
        hm = np.zeros((B, self.K, self.Y, self.X), np.float32)
        ind = np.zeros((B, M), np.int64); mask = np.zeros((B, M), np.float32); reg = np.zeros((B, M, 10), np.float32)
        x0, y0 = self.pc_range[0], self.pc_range[1]
        for b in range(B):
            boxes, labels = gt_boxes[b].cpu().numpy(), gt_labels[b].cpu().numpy()
            k = 0
            for bx, lb in zip(boxes, labels):
                x, y, z, w, l, h, yaw, vx, vy = bx[:9]
                cx, cy = (x - x0) / self.dx, (y - y0) / self.dy
                ix, iy = int(cx), int(cy)
                if not (0 <= ix < self.X and 0 <= iy < self.Y) or k >= M or min(w, l, h) <= 0:
                    continue
                r = max(self.min_radius, int(gaussian_radius(l / self.dy, w / self.dx, self.overlap)))
                draw_gaussian(hm[b, lb], ix, iy, r)
                ind[b, k] = iy * self.X + ix; mask[b, k] = 1
                reg[b, k] = [cx - ix, cy - iy, z, np.log(w), np.log(l), np.log(h), np.sin(yaw), np.cos(yaw), vx, vy]
                k += 1
        t = lambda a: torch.from_numpy(a).to(device)
        return t(hm), t(ind), t(mask), t(reg)

    def loss(self, out, gt_boxes, gt_labels):
        hm_t, ind, mask, reg_t = self.targets(gt_boxes, gt_labels, out["hm"].device)
        l_hm = focal_loss(out["hm"].float(), hm_t)
        B = out["reg"].shape[0]
        pred = out["reg"].float().view(B, 10, -1).gather(2, ind.unsqueeze(1).expand(-1, 10, -1)).permute(0, 2, 1)
        l_reg = (F.l1_loss(pred, reg_t, reduction="none") * self.code_w * mask[..., None]).sum() / mask.sum().clamp(min=1)
        return {"det_hm": l_hm, "det_reg": l_reg}

    # ---- decode -----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, out):
        hm = out["hm"].float().sigmoid()
        keep = (F.max_pool2d(hm, 3, 1, 1) == hm).float()
        hm = hm * keep
        B = hm.shape[0]
        scores, inds = hm.view(B, -1).topk(self.max_dets)
        cls = inds // (self.Y * self.X)
        pix = inds % (self.Y * self.X)
        iy, ix = pix // self.X, pix % self.X
        reg = out["reg"].float().view(B, 10, -1).gather(2, pix.unsqueeze(1).expand(-1, 10, -1)).permute(0, 2, 1)
        x = (ix.float() + reg[..., 0]) * self.dx + self.pc_range[0]
        y = (iy.float() + reg[..., 1]) * self.dy + self.pc_range[1]
        dims = reg[..., 3:6].clamp(max=6).exp()
        yaw = torch.atan2(reg[..., 6], reg[..., 7])
        boxes = torch.stack([x, y, reg[..., 2], dims[..., 0], dims[..., 1], dims[..., 2], yaw, reg[..., 8], reg[..., 9]], -1)
        res = []
        for b in range(B):
            m = scores[b] > self.score_thr
            res.append({"boxes": boxes[b][m], "scores": scores[b][m], "labels": cls[b][m]})
        return res
