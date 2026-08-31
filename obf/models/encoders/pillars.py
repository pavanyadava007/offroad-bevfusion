"""PointPillars encoders (shared by LiDAR and radar). Native implementation mirrors mmdet3d
`PillarFeatureNet` + `PointPillarsScatter`; set use_mmdet3d=true to import the mmdet3d modules instead."""
import torch
import torch.nn as nn


class PillarFeatureNet(nn.Module):
    def __init__(self, in_ch, feat_ch, pillar, pc_range):
        super().__init__()
        self.pillar = pillar
        self.x0, self.y0 = pc_range[0], pc_range[1]
        self.pfn = nn.Sequential(nn.Linear(in_ch + 5, feat_ch, bias=False), nn.BatchNorm1d(feat_ch, eps=1e-3, momentum=0.01), nn.ReLU(True))

    def forward(self, feats, num, coors):
        """feats [B,P,N,F] (x,y,z,...), num [B,P], coors [B,P,2]=(y,x) -> [B,P,C]"""
        B, P, N, F = feats.shape
        n = num.clamp(min=1).to(feats.dtype)[..., None, None]
        mean = feats[..., :3].sum(2, keepdim=True) / n
        f_cluster = feats[..., :3] - mean
        cx = coors[..., 1:2].to(feats.dtype) * self.pillar + self.pillar / 2 + self.x0
        cy = coors[..., 0:1].to(feats.dtype) * self.pillar + self.pillar / 2 + self.y0
        f_center = torch.stack([feats[..., 0] - cx, feats[..., 1] - cy], -1)
        x = torch.cat([feats, f_cluster, f_center], -1)
        mask = (torch.arange(N, device=feats.device)[None, None, :] < num[..., None]).to(feats.dtype)
        x = x * mask[..., None]
        x = self.pfn[0](x)
        x = self.pfn[1](x.reshape(-1, x.shape[-1])).reshape(B, P, N, -1)
        x = self.pfn[2](x)
        x = x * mask[..., None]  # padded slots -> 0; post-ReLU features are >= 0, so zeros never win the max
        return x.max(2).values


class PointPillarsScatter(nn.Module):
    def __init__(self, canvas_hw):
        super().__init__()
        self.H, self.W = canvas_hw

    def forward(self, feats, coors):
        """feats [B,P,C], coors [B,P,2]=(y,x); coors<0 -> padding -> [B,C,H,W]"""
        B, P, C = feats.shape
        valid = (coors[..., 0] >= 0).to(feats.dtype)
        idx = (coors[..., 0].clamp(min=0) * self.W + coors[..., 1].clamp(min=0)).long()
        canvas = feats.new_zeros(B, C, self.H * self.W)
        canvas = canvas.scatter_add(2, idx.unsqueeze(1).expand(-1, C, -1), (feats * valid[..., None]).transpose(1, 2))
        return canvas.view(B, C, self.H, self.W)


def build_pillar_encoder(mc, pc_range, use_mmdet3d=False):
    canvas = int(round((pc_range[3] - pc_range[0]) / mc.pillar)), int(round((pc_range[4] - pc_range[1]) / mc.pillar))
    if use_mmdet3d:
        from mmdet3d.models.middle_encoders import PointPillarsScatter as MMScatter  # noqa: F401
        from mmdet3d.models.voxel_encoders import PillarFeatureNet as MMPFN  # noqa: F401
        pfn = MMPFN(in_channels=mc.in_ch, feat_channels=(mc.feat_ch,), voxel_size=(mc.pillar, mc.pillar, pc_range[5] - pc_range[2]),
                    point_cloud_range=list(pc_range), with_distance=False, legacy=False)
        return pfn, MMScatter(in_channels=mc.feat_ch, output_shape=canvas)
    return PillarFeatureNet(mc.in_ch, mc.feat_ch, mc.pillar, pc_range), PointPillarsScatter(canvas)
