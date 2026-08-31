import torch

from obf.config import load_cfg
from obf.models import BEVFusion
from obf.utils.fake_batch import fake_batch


def test_forward_loss_decode():
    cfg = load_cfg("configs/tiny.yaml")
    m = BEVFusion(cfg)
    b = fake_batch(cfg, B=2)
    out = m(b)
    Y, X = cfg.grid.bev_size
    assert out["hm"].shape == (2, 10, Y, X) and out["reg"].shape == (2, 10, Y, X)
    assert out["seg"].shape == (2, 3, Y, X) and out["occ"].shape == (2, 18, cfg.grid.occ_z, Y, X)
    L = m.loss(out, b)
    assert torch.isfinite(L["total"])
    L["total"].backward()
    m.eval()
    dets = m.decode_det(m(b))
    assert len(dets) == 2 and dets[0]["boxes"].shape[1] == 9


def test_modality_subsets():
    for mods in (["cam"], ["cam", "lidar"], ["lidar", "radar"]):
        cfg = load_cfg("configs/tiny.yaml", [f"model.modalities={mods}"])
        m = BEVFusion(cfg).eval()
        out = m(fake_batch(cfg))
        assert out["seg"].shape[-2:] == tuple(cfg.grid.bev_size)


def test_uncertainty_weighting_learns():
    cfg = load_cfg("configs/tiny.yaml")
    m = BEVFusion(cfg)
    assert m.weighting.log_vars.requires_grad and len(m.weighting.tasks) == 3


def test_pfn_mask_mul_equals_masked_fill():
    """Stage-4b op swap: multiplying by the point mask before the max must reproduce the old masked_fill(-1e4)
    behaviour exactly — per pillar for occupied pillars, and on the scattered canvas for padded (empty) pillars."""
    from obf.models.encoders.pillars import build_pillar_encoder

    cfg = load_cfg("configs/tiny.yaml")
    b = fake_batch(cfg, B=2)
    feats, num, coors = b["lidar_feats"], b["lidar_num"].clone(), b["lidar_coors"].clone()
    num[:, -1] = 0; coors[:, -1] = -1  # force one padded/empty pillar per sample
    pfn, scatter = build_pillar_encoder(cfg.model.lidar, cfg.grid.pc_range)
    pfn.eval()

    def old_forward(feats, num, coors):  # pre-swap implementation, same weights
        B, P, N, F = feats.shape
        n = num.clamp(min=1).to(feats.dtype)[..., None, None]
        mean = feats[..., :3].sum(2, keepdim=True) / n
        f_cluster = feats[..., :3] - mean
        cx = coors[..., 1:2].to(feats.dtype) * pfn.pillar + pfn.pillar / 2 + pfn.x0
        cy = coors[..., 0:1].to(feats.dtype) * pfn.pillar + pfn.pillar / 2 + pfn.y0
        f_center = torch.stack([feats[..., 0] - cx, feats[..., 1] - cy], -1)
        x = torch.cat([feats, f_cluster, f_center], -1)
        mask = (torch.arange(N)[None, None, :] < num[..., None]).to(feats.dtype)
        x = x * mask[..., None]
        x = pfn.pfn[0](x)
        x = pfn.pfn[1](x.reshape(-1, x.shape[-1])).reshape(B, P, N, -1)
        x = pfn.pfn[2](x)
        x = x.masked_fill(mask[..., None] == 0, -1e4)
        return x.max(2).values

    with torch.no_grad():
        new_p, old_p = pfn(feats, num, coors), old_forward(feats, num, coors)
        occupied = num > 0
        assert torch.equal(new_p[occupied], old_p[occupied])  # bit-identical on every occupied pillar
        assert torch.equal(scatter(new_p, coors), scatter(old_p, coors))  # canvas identical incl. empty pillars
