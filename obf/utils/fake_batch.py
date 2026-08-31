"""Synthetic batch with the exact tensor layout of NuScenesDataset.collate — used by tests, CI and export smoke checks."""
import torch


def fake_batch(cfg, B=1, device="cpu", n_boxes=3):
    g = cfg.grid
    Y, X = g.bev_size
    x0, y0, z0, x1, y1, z1 = g.pc_range
    H, W = cfg.data.img_size
    N = len(cfg.data.cams)
    D = torch.arange(*cfg.model.cam.d_bound).numel()
    fH, fW = H // 16, W // 16
    M = N * D * fH * fW
    b = {
        "imgs": torch.randn(B, N, 3, H, W),
        "cam_bev_idx": torch.randint(0, Y * X, (B, M)),
        "cam_valid": torch.rand(B, M) > 0.3,
    }

    def pillars(mc, F):
        P, Np = mc.max_pillars, mc.max_points
        cw = int(round((x1 - x0) / mc.pillar))
        feats = torch.rand(B, P, Np, F)
        feats[..., 0] = feats[..., 0] * (x1 - x0) + x0
        feats[..., 1] = feats[..., 1] * (y1 - y0) + y0
        feats[..., 2] = feats[..., 2] * (z1 - z0) + z0
        num = torch.randint(1, Np + 1, (B, P))
        coors = torch.randint(0, cw, (B, P, 2))
        return feats, num, coors

    if "lidar" in cfg.model.modalities:
        b["lidar_feats"], b["lidar_num"], b["lidar_coors"] = pillars(cfg.model.lidar, cfg.model.lidar.in_ch)
    if "radar" in cfg.model.modalities:
        b["radar_feats"], b["radar_num"], b["radar_coors"] = pillars(cfg.model.radar, cfg.model.radar.in_ch)
    boxes = []
    for _ in range(B):
        bx = torch.rand(n_boxes, 9)
        bx[:, 0] = bx[:, 0] * (x1 - x0) * 0.8 + x0 * 0.8
        bx[:, 1] = bx[:, 1] * (y1 - y0) * 0.8 + y0 * 0.8
        bx[:, 3:6] = bx[:, 3:6] * 2 + 1
        boxes.append(bx)
    b["gt_boxes"] = boxes
    b["gt_labels"] = [torch.randint(0, len(cfg.heads.det.classes), (n_boxes,)) for _ in range(B)]
    b["seg"] = (torch.rand(B, len(cfg.heads.seg.classes), Y, X) > 0.5).float()
    b["occ"] = torch.randint(0, cfg.heads.occ.classes, (B, Y, X, g.occ_z))
    b["token"] = [f"fake{i}" for i in range(B)]
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
