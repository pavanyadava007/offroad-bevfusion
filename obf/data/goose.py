"""GOOSE (Fraunhofer IOSB) off-road loader: single windshield camera + VLS-128 LiDAR with per-point labels.
BEV single-label seg target = majority class of labelled points in each 0.4 m cell (255 = no points).
Directory layout is configurable (GOOSE releases differ); class merge is by *name* from goose_label_mapping.csv."""
import csv
import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import IMG_MEAN, IMG_STD
from .lss_geometry import cam_to_bev_index, make_frustum
from .voxelize import voxelize

# GOOSE class-name substrings -> merged BEV classes (index into configs/goose_transfer.yaml heads.seg.classes)
MERGE = {
    "drivable": ["asphalt", "gravel", "soil", "cobble", "road", "sidewalk", "bikeway", "curb", "leaves"],
    "low_veg": ["low_grass", "high_grass", "moss", "scenery_vegetation", "crops", "bush"],
    "high_veg": ["forest", "tree_trunk", "tree_crown", "tree_root", "hedge"],
    "obstacle": ["obstacle", "wall", "fence", "guard_rail", "pole", "sign", "barrier", "building", "rock",
                 "debris", "container", "misc_sign", "boom_barrier", "kick_scooter", "bicycle", "military"],
    "person": ["person", "pedestrian", "rider", "animal"],
    "vehicle": ["car", "truck", "bus", "trailer", "on_rails", "motorcycle", "heavy_machinery", "wheeled"],
    "water": ["water", "puddle", "snow"],
    "terrain_other": ["rubble", "sand", "ego_vehicle", "outlier"],
}
SKY_LIKE = ["sky", "undefined"]


def load_label_map(csv_path, classes):
    """goose_label_mapping.csv (label_key, class_name, ...) -> np.array[max_id+1] of merged class ids (255=ignore)."""
    rows = list(csv.DictReader(open(csv_path)))
    key_col = "label_key" if "label_key" in rows[0] else list(rows[0].keys())[0]
    name_col = "class_name" if "class_name" in rows[0] else list(rows[0].keys())[1]
    n = max(int(r[key_col]) for r in rows) + 1
    lut = np.full(n, 255, np.uint8)
    for r in rows:
        name = r[name_col].lower()
        if any(s in name for s in SKY_LIKE):
            continue
        for merged, subs in MERGE.items():
            if any(s in name for s in subs) and merged in classes:
                lut[int(r[key_col])] = classes.index(merged)
                break
        else:
            if "obstacle" in classes:
                lut[int(r[key_col])] = classes.index("obstacle")
    return lut


class GooseDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        root = cfg.data.root
        self.classes = cfg.heads.seg.classes
        self.lut = load_label_map(os.path.join(root, "goose_label_mapping.csv"), self.classes)
        lidar = sorted(glob.glob(os.path.join(root, "lidar", split, "*", "*_vls128.bin")))
        self.items = []
        for lp in lidar:
            stem = os.path.basename(lp).replace("_vls128.bin", "")
            scene = os.path.basename(os.path.dirname(lp))
            lab = os.path.join(root, "labels", split, scene, stem + "_goose.label")  # goose_3d zips: labels/ tree
            if not os.path.exists(lab):
                lab = lp.replace(".bin", ".label")  # fallback: label next to the .bin
            img = os.path.join(root, "images", split, scene, stem + "_windshield_vis.png")
            calib = os.path.join(root, "calibration", scene + ".npz")  # cam2lidar 4x4, intrinsic 3x3 (see docs)
            if os.path.exists(lab) and os.path.exists(img):
                self.items.append((lp, lab, img, calib))
        self.pc_range = cfg.grid.pc_range
        self.bev = tuple(cfg.grid.bev_size)
        self.img_size = tuple(cfg.data.img_size)
        self.frustum = make_frustum(cfg.model.cam.d_bound, self.img_size, (self.img_size[0] // 16, self.img_size[1] // 16))
        self.train = split == "train"
        self.weights = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        lp, lab, img, calib = self.items[i]
        pts = np.fromfile(lp, np.float32).reshape(-1, 4)
        lbl = np.fromfile(lab, np.uint32).astype(np.int64) & 0xFFFF
        lbl = self.lut[np.clip(lbl, 0, len(self.lut) - 1)]
        cal = np.load(calib) if os.path.exists(calib) else {"cam2lidar": np.eye(4), "intrinsic": np.eye(3)}
        H, W = self.img_size
        im = cv2.imread(img)[:, :, ::-1]
        h0, w0 = im.shape[:2]
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        imgs = torch.from_numpy(((im - IMG_MEAN) / IMG_STD).transpose(2, 0, 1).copy())[None]
        f = lambda a: torch.as_tensor(np.asarray(a)[None], dtype=torch.float32)
        idx, valid = cam_to_bev_index(self.frustum, f(cal["cam2lidar"]), f(cal["intrinsic"]),
                                      f(np.diag([W / w0, H / h0, 1.0])), f(np.zeros(3)), self.pc_range, self.bev)
        feats = np.concatenate([pts[:, :3], pts[:, 3:4] / 255.0, np.zeros((len(pts), 1), np.float32)], 1)
        lf, ln, lc = voxelize(feats, self.cfg.model.lidar.pillar, self.pc_range,
                              self.cfg.model.lidar.max_pillars, self.cfg.model.lidar.max_points)
        # BEV majority label
        Y, X = self.bev
        x0, y0, _, x1, y1, _ = self.pc_range
        ix = ((pts[:, 0] - x0) / (x1 - x0) * X).astype(np.int64)
        iy = ((pts[:, 1] - y0) / (y1 - y0) * Y).astype(np.int64)
        m = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (lbl != 255)
        C = len(self.classes)
        cnt = np.bincount((iy[m] * X + ix[m]) * C + lbl[m], minlength=Y * X * C).reshape(Y, X, C)
        seg = np.where(cnt.sum(-1) > 0, cnt.argmax(-1), 255).astype(np.int64)
        return {"token": os.path.basename(lp), "imgs": imgs, "cam_bev_idx": idx, "cam_valid": valid,
                "lidar_feats": torch.from_numpy(lf), "lidar_num": torch.from_numpy(ln).long(),
                "lidar_coors": torch.from_numpy(lc).long(), "seg": torch.from_numpy(seg),
                "gt_boxes": torch.zeros(0, 9), "gt_labels": torch.zeros(0, dtype=torch.long)}
