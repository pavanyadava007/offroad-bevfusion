"""LSS frustum -> BEV index computed in the data pipeline (numpy/torch, CPU), so the network graph contains no
calibration math and exports to ONNX with static shapes."""
import torch


def make_frustum(d_bound, img_size, feat_size):
    ds = torch.arange(*d_bound, dtype=torch.float32)
    D = ds.numel()
    H, W = img_size
    fH, fW = feat_size
    us = torch.linspace(0, W - 1, fW)
    vs = torch.linspace(0, H - 1, fH)
    d = ds.view(D, 1, 1).expand(D, fH, fW)
    u = us.view(1, 1, fW).expand(D, fH, fW)
    v = vs.view(1, fH, 1).expand(D, fH, fW)
    return torch.stack([u, v, d], -1)  # [D,fH,fW,3] = (u, v, depth)


def cam_to_bev_index(frustum, cam2ego, intrins, post_rot, post_tran, pc_range, bev_size):
    """frustum [D,fH,fW,3]; cam2ego [N,4,4]; intrins [N,3,3]; post_rot [N,3,3]; post_tran [N,3]
    -> idx [N*D*fH*fW] long (flattened BEV, row=y col=x), valid [N*D*fH*fW] bool. Flatten order (N,D,fH,fW)."""
    pts = frustum[None] - post_tran[:, None, None, None, :]
    pts = torch.inverse(post_rot)[:, None, None, None] @ pts.unsqueeze(-1)
    pts = torch.cat([pts[..., :2, :] * pts[..., 2:3, :], pts[..., 2:3, :]], 4)
    R = cam2ego[:, :3, :3] @ torch.inverse(intrins)
    pts = (R[:, None, None, None] @ pts + cam2ego[:, None, None, None, :3, 3:4]).squeeze(-1)
    x0, y0, z0, x1, y1, z1 = pc_range
    Y, X = bev_size
    dx, dy = (x1 - x0) / X, (y1 - y0) / Y
    ix = ((pts[..., 0] - x0) / dx).floor().long()
    iy = ((pts[..., 1] - y0) / dy).floor().long()
    valid = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (pts[..., 2] >= z0) & (pts[..., 2] < z1)
    idx = torch.where(valid, (iy * X + ix).clamp(0, Y * X - 1), torch.zeros_like(ix))
    return idx.reshape(-1), valid.reshape(-1)
