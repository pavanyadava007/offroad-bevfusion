"""Synthetic dataset (dataset: fake) with the exact collate layout of NuScenesDataset — CPU smoke tests of the full
train -> eval -> export loop without nuScenes on disk. Not for reporting numbers."""
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.fake_batch import fake_batch


class FakeDataset(Dataset):
    def __init__(self, cfg, split="train", n=8):
        self.cfg, self.n = cfg, n
        classes = list(cfg.heads.det.classes) if cfg.heads.get("det") else ["car", "pedestrian"]
        rng = np.random.default_rng(0)
        self.samples = []
        for i in range(n):  # every other frame: pedestrian in the corridor < 5 m (VLA hazard case)
            boxes = [[3.0 + i % 3, 0.3, 0.0, 0.6, 0.6, 1.7, 0.0, 0.0, 0.0]] if i % 2 == 0 else []
            names = ["pedestrian"] if i % 2 == 0 else []
            boxes += rng.uniform([5, -8, 0, 1.5, 3.5, 1.4, -3, 0, 0], [30, 8, 0.5, 2.2, 5, 1.8, 3, 0, 0], (2, 9)).tolist()
            names += ["car", "truck" if "truck" in classes else classes[0]]
            self.samples.append({"token": f"fake_{split}_{i}", "ego2global": np.eye(4).tolist(), "scene_token": "fake_scene",
                                 "scene_description": "synthetic", "gt_boxes": boxes, "gt_names": names,
                                 "cams": {"CAM_FRONT": {"path": ""}}})
        self.weights = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        torch.manual_seed(i)
        b = fake_batch(self.cfg, B=1)
        item = {k: (v[0] if torch.is_tensor(v) else v[0]) for k, v in b.items()}
        item["token"] = self.samples[i]["token"]
        return item
