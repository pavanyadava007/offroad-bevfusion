import numpy as np
import torch

from obf.data.common import box_corners_bev, rasterize_polys
from obf.data.lss_geometry import cam_to_bev_index, make_frustum
from obf.data.voxelize import voxelize


def test_voxelize_static_shapes_and_content():
    pts = np.array([[0.1, 0.1, 0.0, 1.0], [0.15, 0.12, 0.0, 2.0], [-3.0, 2.0, 0.0, 3.0], [100.0, 0, 0, 4.0]], np.float32)
    f, n, c = voxelize(pts, 0.2, [-6.4, -6.4, -1, 6.4, 6.4, 5.4], max_pillars=10, max_points=3)
    assert f.shape == (10, 3, 4) and n.shape == (10,) and c.shape == (10, 2)
    assert n.sum() == 3 and (c[n > 0] >= 0).all() and (c[n == 0] == -1).all()
    p = np.where(n == 2)[0][0]
    assert c[p, 1] == int((0.1 + 6.4) / 0.2) and c[p, 0] == int((0.1 + 6.4) / 0.2)


def test_voxelize_overflow():
    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.uniform(-6, 6, (5000, 3)), rng.uniform(0, 1, (5000, 1))], 1).astype(np.float32)
    f, n, c = voxelize(pts, 0.2, [-6.4, -6.4, -1, 6.4, 6.4, 5.4], max_pillars=50, max_points=4)
    assert (n > 0).sum() == 50 and n.max() <= 4


def test_voxelize_float_boundary_stays_in_canvas():
    """Regression (GOOSE aying_hills__0112): y = 39.999996f passes the < 40 filter but (y+40)/0.2 rounds to exactly
    400.0 in float32 -> pillar index == canvas width -> CUDA scatter assert. Indices must be clamped."""
    y = np.float32(39.999996)
    pts = np.array([[0.0, y, 0.0, 1.0], [y, 0.0, 0.0, 1.0]], np.float32)
    f, n, c = voxelize(pts, 0.2, [-40, -40, -1, 40, 40, 5.4], max_pillars=10, max_points=3)
    W = 400
    occupied = c[n > 0]
    assert (occupied >= 0).all() and (occupied < W).all(), occupied.tolist()
    assert n.sum() == 2


def test_lss_index_forward_point_lands_ahead():
    fr = make_frustum([1.0, 9.0, 1.0], (64, 128), (4, 8))
    K = torch.tensor([[[100.0, 0, 64], [0, 100.0, 32], [0, 0, 1]]])
    cam2ego = torch.tensor([[[0, 0, 1, 0.0], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]]])  # optical -> ego (x fwd)
    idx, valid = cam_to_bev_index(fr, cam2ego, K, torch.eye(3)[None], torch.zeros(1, 3), [-6.4, -6.4, -1, 6.4, 6.4, 5.4], (32, 32))
    assert idx.shape == valid.shape == (8 * 4 * 8,)
    xs = idx[valid] % 32
    assert (xs.float().mean() > 16) and valid.any()  # points project to positive ego-x (right half of the BEV cols)


def test_box_raster():
    boxes = np.array([[0, 0, 0, 2.0, 4.0, 1.5, 0.0]], np.float32)
    m = rasterize_polys(list(box_corners_bev(boxes)), [-6.4, -6.4, -1, 6.4, 6.4, 5.4], (32, 32))
    assert m.sum() > 0 and m[16, 16] == 1


def test_goose_real_frame():
    import glob as g
    import os

    import pytest
    if not (os.path.exists("data/goose/goose_label_mapping.csv") and g.glob("data/goose/lidar/*/*/*_vls128.bin")):
        pytest.skip("GOOSE data not present")
    from obf.config import load_cfg
    from obf.data.goose import GooseDataset
    cfg = load_cfg("configs/goose_transfer.yaml")
    ds = GooseDataset(cfg, "val")
    assert len(ds) > 0
    it = ds[0]
    Y, X = cfg.grid.bev_size
    H, W = cfg.data.img_size
    assert it["imgs"].shape == (1, 3, H, W) and it["seg"].shape == (Y, X)
    seg = it["seg"].numpy()
    valid = seg[seg != 255]
    assert valid.size > 0, "all-ignore BEV target"
    hist = np.bincount(valid, minlength=len(cfg.heads.seg.classes))
    assert (hist > 0).sum() >= 2, f"degenerate class histogram: {hist.tolist()}"
    assert it["lidar_feats"].shape[0] == cfg.model.lidar.max_pillars
