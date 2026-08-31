"""Camera + LiDAR + radar BEV multitask network."""
import torch
import torch.nn as nn

from .encoders.camera_lss import CameraEncoder, LSSViewTransform
from .encoders.pillars import build_pillar_encoder
from .fusion.bev_fuser import ConvFuser
from .fusion.deform_bev_encoder import DeformBEVEncoder
from .heads.center_head import CenterHead
from .heads.seg_occ_heads import BEVSegHead, OccHead
from .layers import BEVBackbone, conv_bn_relu
from .losses import FixedWeighting, UncertaintyWeighting


class BEVFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        g, m = cfg.grid, cfg.model
        self.mods = list(m.modalities)
        pc, bev = g.pc_range, tuple(g.bev_size)
        in_ch = 0
        if "cam" in self.mods:
            D = torch.arange(*m.cam.d_bound).numel()
            self.cam_enc = CameraEncoder(m.cam.backbone, m.cam.get("pretrained", True), D, m.cam.out_ch)
            self.lss = LSSViewTransform(bev)
            in_ch += m.cam.out_ch
        if "lidar" in self.mods:
            self.lidar_pfn, self.lidar_scatter = build_pillar_encoder(m.lidar, pc, m.lidar.get("use_mmdet3d", False))
            self.lidar_backbone = BEVBackbone(m.lidar.feat_ch, m.lidar.out_ch)
            in_ch += m.lidar.out_ch
        if "radar" in self.mods:
            self.radar_pfn, self.radar_scatter = build_pillar_encoder(m.radar, pc)
            up = int(round(m.radar.pillar / ((pc[3] - pc[0]) / bev[1])))
            self.radar_neck = nn.Sequential(conv_bn_relu(m.radar.feat_ch, m.radar.out_ch),
                                            nn.Upsample(scale_factor=up, mode="bilinear", align_corners=False),
                                            conv_bn_relu(m.radar.out_ch, m.radar.out_ch))
            self.radar_dropout = m.radar.get("dropout", 0.0)
            in_ch += m.radar.out_ch
        self.fuser = ConvFuser(in_ch, m.fuser.out_ch)
        be = m.bev_encoder
        self.bev_enc = DeformBEVEncoder(m.fuser.out_ch, bev, be.layers, be.heads, be.points, be.ffn)
        C = m.fuser.out_ch
        heads = {}
        if cfg.heads.get("det"):
            h = cfg.heads.det
            heads["det"] = CenterHead(C, h.classes, pc, bev, h.get("head_ch", 64), h.get("score_thr", 0.1), h.get("max_dets", 500))
        if cfg.heads.get("seg"):
            heads["seg"] = BEVSegHead(C, cfg.heads.seg.classes, cfg.heads.seg.get("multilabel", True))
        if cfg.heads.get("occ"):
            heads["occ"] = OccHead(C, cfg.heads.occ.classes, g.occ_z)
        self.heads = nn.ModuleDict(heads)
        self.weighting = UncertaintyWeighting(heads.keys()) if cfg.loss.get("uncertainty_weighting", True) \
            else FixedWeighting(cfg.loss.get("fixed_weights", {}))

    # ---------------------------------------------------------------------------------------------------------
    def bev_features(self, batch):
        feats = []
        if "cam" in self.mods:
            imgs = batch["imgs"]
            B, N = imgs.shape[:2]
            d, c = self.cam_enc(imgs.flatten(0, 1))
            d, c = d.view(B, N, *d.shape[1:]), c.view(B, N, *c.shape[1:])
            feats.append(self.lss(d, c, batch["cam_bev_idx"], batch["cam_valid"]))
        if "lidar" in self.mods:
            x = self.lidar_pfn(batch["lidar_feats"], batch["lidar_num"], batch["lidar_coors"])
            feats.append(self.lidar_backbone(self.lidar_scatter(x, batch["lidar_coors"])))
        if "radar" in self.mods:
            x = self.radar_pfn(batch["radar_feats"], batch["radar_num"], batch["radar_coors"])
            x = self.radar_scatter(x, batch["radar_coors"])
            if self.training and self.radar_dropout > 0:
                keep = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) > self.radar_dropout).to(x.dtype)
                x = x * keep
            feats.append(self.radar_neck(x))
        return self.bev_enc(self.fuser(torch.cat(feats, 1)))

    def forward(self, batch):
        bev = self.bev_features(batch)
        out = {}
        for head in self.heads.values():
            out.update(head(bev))
        return out

    def loss(self, out, batch):
        L = {}
        task = {}
        if "det" in self.heads:
            L.update(self.heads["det"].loss(out, batch["gt_boxes"], batch["gt_labels"]))
            task["det"] = L["det_hm"] + L["det_reg"]
        if "seg" in self.heads:
            L.update(self.heads["seg"].loss(out["seg"], batch["seg"]))
            task["seg"] = L["seg"]
        if "occ" in self.heads:
            L.update(self.heads["occ"].loss(out["occ"], batch["occ"]))
            task["occ"] = L["occ"]
        L["total"] = self.weighting(task)
        return L

    @torch.no_grad()
    def decode_det(self, out):
        return self.heads["det"].decode(out) if "det" in self.heads else None

    def export_io(self):
        """Ordered ONNX input names / output names (static shapes)."""
        ins = []
        if "cam" in self.mods:
            ins += ["imgs", "cam_bev_idx", "cam_valid"]
        if "lidar" in self.mods:
            ins += ["lidar_feats", "lidar_num", "lidar_coors"]
        if "radar" in self.mods:
            ins += ["radar_feats", "radar_num", "radar_coors"]
        outs = []
        if "det" in self.heads:
            outs += ["hm", "reg"]
        if "seg" in self.heads:
            outs += ["seg"]
        if "occ" in self.heads:
            outs += ["occ"]
        return ins, outs
