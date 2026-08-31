"""Dump preprocessed network inputs for K frames as .npy (Python) and raw .bin (C++/ROS) + manifest.
Used for INT8 calibration (100 frames), ROS 2 replay (all val), HF Space cache (5).
python -m obf.export.dump_samples --cfg configs/base.yaml --n 100 --out data/samples/calib [--raw] [--with_outputs --ckpt ...]"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import load_cfg
from ..data import build_dataset, collate
from ..models import BEVFusion
from .onnx_export import export_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--split", default="val"); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True); ap.add_argument("--raw", action="store_true"); ap.add_argument("--with_outputs", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--opts", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.opts)
    model = BEVFusion(cfg).eval()
    if a.with_outputs and a.ckpt:
        model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    ins, outs = model.export_io()
    ds = build_dataset(cfg, a.split)
    dl = DataLoader(ds, batch_size=1, collate_fn=collate, num_workers=cfg.data.workers)
    os.makedirs(a.out, exist_ok=True)
    manifest = {"inputs": ins, "outputs": outs, "frames": []}
    for i, batch in enumerate(dl):
        if i >= a.n:
            break
        tok = batch["token"][0]
        d = os.path.join(a.out, f"{i:04d}_{tok}")
        os.makedirs(d, exist_ok=True)
        tens = export_inputs(model, batch)
        shapes = {}
        for name, t in zip(ins, tens):
            arr = t.numpy()
            shapes[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            np.save(os.path.join(d, name + ".npy"), arr)
            if a.raw:
                arr.tofile(os.path.join(d, name + ".bin"))
        if a.with_outputs:
            with torch.no_grad():
                o = model(batch)
            for k in outs:
                np.save(os.path.join(d, k + ".npy"), o[k].numpy())
        info = ds.samples[i] if hasattr(ds, "samples") else {}
        manifest["frames"].append({"dir": d, "token": tok, "shapes": shapes, "ego2global": np.asarray(info.get("ego2global", np.eye(4))).tolist(),
                                   "lidar_path": info.get("lidar_sweeps", [{}])[0].get("path"),
                                   "cam_front": info.get("cams", {}).get("CAM_FRONT", {}).get("path")})
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("dumped", len(manifest["frames"]), "frames to", a.out)


if __name__ == "__main__":
    main()
