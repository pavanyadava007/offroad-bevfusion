"""True zero-shot GOOSE eval: the nuScenes-trained cam+LiDAR model runs with its OWN 3-channel multilabel seg head;
sigmoid(drivable channel) > 0.5 maps to {drivable, obstacle}. (Loading the checkpoint into a 2-class head, as the
original goose_zero_shot.yaml eval implied, is a strict state-dict shape mismatch — the head would be random.)"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from obf.config import load_cfg
from obf.data import build_dataset, collate
from obf.eval import ConfMat
from obf.models import BEVFusion
from obf.utils.misc import save_json, to_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_cfg", default="configs/ablation_cam_lidar.yaml")
    ap.add_argument("--data_cfg", default="configs/goose_zero_shot.yaml")
    ap.add_argument("--ckpt", default="checkpoints/cam_lidar/best.pt")
    ap.add_argument("--split", default="val"); ap.add_argument("--name", default="goose_zero_shot")
    a = ap.parse_args()
    cfgm, cfgd = load_cfg(a.model_cfg), load_cfg(a.data_cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = BEVFusion(cfgm).to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])
    ds = build_dataset(cfgd, a.split)
    dl = DataLoader(ds, batch_size=cfgd.data.batch_size, num_workers=cfgd.data.workers, collate_fn=collate)
    classes = list(cfgd.heads.seg.classes)
    cm = ConfMat(len(classes))
    with torch.no_grad():
        for batch in dl:
            batch = to_device(batch, dev)
            out = model(batch)
            drivable = out["seg"][:, 0].sigmoid() > 0.5  # nuScenes channel 0 = drivable_area
            cm.add((~drivable).long(), batch["seg"])     # 0 = drivable, 1 = obstacle
    iou = cm.iou()
    res = {"name": a.name, "n_samples": len(ds), "seg_IoU": {c: float(v) for c, v in zip(classes, iou)},
           "seg_mIoU": float(np.mean(iou)), "note": "zero-shot: nuScenes 3-ch head, drivable channel thresholded"}
    save_json(res, f"results/{a.name}.json")
    print(res)


if __name__ == "__main__":
    main()
