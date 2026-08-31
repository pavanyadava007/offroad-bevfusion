"""nuScenes-mini multimodal dataset. Consumes infos from scripts/create_infos.py.
Reference frame for everything (points, boxes, BEV grid, occupancy): ego frame at LIDAR_TOP keyframe timestamp."""
import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import DET_CLASSES, IMG_MEAN, IMG_STD, VEHICLE_SET, box_corners_bev, rasterize_polys
from .lss_geometry import cam_to_bev_index, make_frustum
from .voxelize import voxelize

LIST_KEYS = ("gt_boxes", "gt_labels", "token")


def collate(items):
    out = {}
    for k in items[0]:
        out[k] = [it[k] for it in items] if k in LIST_KEYS else torch.stack([it[k] for it in items])
    return out


class NuScenesDataset(Dataset):
    def __init__(self, cfg, split="train", tokens=None):
        self.cfg = cfg
        path = os.path.join(cfg.data.infos, f"nuscenes_{cfg.data.version}_infos.pkl")
        with open(path, "rb") as f:
            infos = pickle.load(f)
        self.samples = infos[split]
        if tokens is not None:
            tokens = set(tokens)
            self.samples = [s for s in self.samples if s["token"] in tokens]
        self.split = split
        self.train = split == "train"
        self.mods = cfg.model.modalities
        self.pc_range = cfg.grid.pc_range
        self.bev = tuple(cfg.grid.bev_size)
        self.img_size = tuple(cfg.data.img_size)
        fH, fW = self.img_size[0] // 16, self.img_size[1] // 16
        self.frustum = make_frustum(cfg.model.cam.d_bound, self.img_size, (fH, fW))
        self.cls_idx = {c: i for i, c in enumerate(cfg.heads.det.classes)} if cfg.heads.get("det") else {}
        self.ped_dilate = 1
        self.weights = None
        hw = cfg.data.get("hard_scene_weights")
        if hw and os.path.exists(hw) and self.train:
            import json
            w = json.load(open(hw)).get("scene_weights", {})
            self.weights = [float(w.get(s["scene_token"], 1.0)) for s in self.samples]

    def __len__(self):
        return len(self.samples)

    # ---- modalities -------------------------------------------------------------------------------------------
    def _cams(self, info):
        imgs, c2e, K, pr, pt = [], [], [], [], []
        H, W = self.img_size
        for cam in self.cfg.data.cams:
            c = info["cams"][cam]
            im = cv2.imread(c["path"])[:, :, ::-1]
            h0, w0 = im.shape[:2]
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            imgs.append(torch.from_numpy(((im - IMG_MEAN) / IMG_STD).transpose(2, 0, 1).copy()))
            c2e.append(c["cam2ego"]); K.append(c["intrinsic"])
            pr.append(np.diag([W / w0, H / h0, 1.0])); pt.append(np.zeros(3))
        f = lambda a: torch.as_tensor(np.stack(a), dtype=torch.float32)
        idx, valid = cam_to_bev_index(self.frustum, f(c2e), f(K), f(pr), f(pt), self.pc_range, self.bev)
        return torch.stack(imgs), idx, valid

    def _lidar(self, info):
        pts = []
        for s in info["lidar_sweeps"][: self.cfg.data.lidar_sweeps]:
            p = np.fromfile(s["path"], np.float32).reshape(-1, 5)[:, :4]
            M = s["sweep2ref"]
            xyz = p[:, :3] @ M[:3, :3].T + M[:3, 3]
            pts.append(np.concatenate([xyz, p[:, 3:4] / 255.0, np.full((len(p), 1), s["dt"], np.float32)], 1))
        return np.concatenate(pts, 0).astype(np.float32)

    def _pillars(self, pts, mc):
        f, n, c = voxelize(pts, mc.pillar, self.pc_range, mc.max_pillars, mc.max_points)
        return torch.from_numpy(f), torch.from_numpy(n).long(), torch.from_numpy(c).long()

    # ---- targets ----------------------------------------------------------------------------------------------
    def _boxes(self, info):
        boxes = np.asarray(info["gt_boxes"], np.float32).reshape(-1, 9)
        names = info["gt_names"]
        keep = [i for i, n in enumerate(names) if n in self.cls_idx]
        boxes = boxes[keep]
        labels = np.array([self.cls_idx[names[i]] for i in keep], np.int64)
        return boxes, labels, [names[i] for i in keep]

    def _seg(self, info, boxes, names):
        driv = np.load(info["drivable_path"]) if info.get("drivable_path") and os.path.exists(info["drivable_path"]) \
            else np.zeros(self.bev, np.uint8)
        veh = [i for i, n in enumerate(names) if n in VEHICLE_SET]
        ped = [i for i, n in enumerate(names) if n == "pedestrian"]
        veh_m = rasterize_polys(list(box_corners_bev(boxes[veh])) if veh else [], self.pc_range, self.bev)
        ped_m = rasterize_polys(list(box_corners_bev(boxes[ped])) if ped else [], self.pc_range, self.bev,
                                dilate=self.ped_dilate)
        return torch.from_numpy(np.stack([driv, veh_m, ped_m]).astype(np.float32))

    def _occ(self, info):
        Z = self.cfg.grid.occ_z
        p = info.get("occ_path")
        if not p or not os.path.exists(p):
            return torch.full((*self.bev, Z), 255, dtype=torch.long)
        d = np.load(p)
        sem = d["semantics"].astype(np.int64).transpose(1, 0, 2)  # Occ3D [X,Y,Z] -> ours [Y,X,Z]
        if self.cfg.heads.occ.get("use_camera_mask", True) and "mask_camera" in d:
            sem[d["mask_camera"].transpose(1, 0, 2) == 0] = 255
        return torch.from_numpy(sem)

    def __getitem__(self, i):
        info = self.samples[i]
        b = {"token": info["token"]}
        if "cam" in self.mods:
            b["imgs"], b["cam_bev_idx"], b["cam_valid"] = self._cams(info)
        if "lidar" in self.mods:
            b["lidar_feats"], b["lidar_num"], b["lidar_coors"] = self._pillars(self._lidar(info), self.cfg.model.lidar)
        if "radar" in self.mods:
            rp = np.load(info["radar_path"]) if info.get("radar_path") else np.zeros((0, 7), np.float32)
            b["radar_feats"], b["radar_num"], b["radar_coors"] = self._pillars(rp, self.cfg.model.radar)
        boxes, labels, names = self._boxes(info)
        b["gt_boxes"], b["gt_labels"] = torch.from_numpy(boxes), torch.from_numpy(labels)
        if self.cfg.heads.get("seg"):
            b["seg"] = self._seg(info, boxes, names)
        if self.cfg.heads.get("occ"):
            b["occ"] = self._occ(info)
        return b


__all__ = ["NuScenesDataset", "collate", "DET_CLASSES"]
