"""Auto-labelling: run the fused nuScenes model on unlabelled frames (e.g. GOOSE / RELLIS) and keep high-confidence
detections + seg as pseudo-labels (JSON per frame). Reused for semi-supervised fine-tuning."""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from ..config import load_cfg
from ..data import build_dataset, collate
from ..models import BEVFusion
from ..utils.misc import save_json, to_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val"); ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="data/pseudo_labels")
    ap.add_argument("--opts", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.opts)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = BEVFusion(cfg).to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])
    dl = DataLoader(build_dataset(cfg, a.split), batch_size=1, collate_fn=collate, num_workers=cfg.data.workers)
    os.makedirs(a.out, exist_ok=True)
    n = 0
    with torch.no_grad():
        for batch in dl:
            out = model(to_device(batch, dev))
            d = model.decode_det(out)[0]
            keep = d["scores"] > a.thr
            rec = {"boxes": d["boxes"][keep].cpu().tolist(), "labels": d["labels"][keep].cpu().tolist(),
                   "scores": d["scores"][keep].cpu().tolist(),
                   "drivable_ratio": float((out["seg"][0, 0].sigmoid() > 0.5).float().mean()) if "seg" in out else None}
            save_json(rec, os.path.join(a.out, f"{batch['token'][0]}.json")); n += 1
    print("pseudo-labelled frames:", n)


if __name__ == "__main__":
    main()
