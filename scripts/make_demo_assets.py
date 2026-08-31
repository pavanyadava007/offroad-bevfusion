"""assets/samples for the HF Space: 5 frames (2 real pedestrian-hazard train frames + 3 val frames), each with
input .npy (ONNX feed), model output .npy (CPU fallback), cam_front.jpg, and vla.json (LoRA VLM, GT perception)."""
import json
import os
import shutil

import numpy as np
import torch
from PIL import Image

from obf.config import load_cfg
from obf.data import build_dataset, collate
from obf.export.onnx_export import export_inputs
from obf.models import BEVFusion
from obf.vla.grounding import VLAGrounder
from obf.vla.perception_json import objects_from_boxes, perception_json

HAZARD = ["c923fe08b2ff4e27975d2bf30934383b", "e0845f5322254dafadbbed75aaa07969"]  # train scene-0061, ped 2.4/4.9 m
cfg = load_cfg("configs/base.yaml")
model = BEVFusion(cfg).cuda().eval()
model.load_state_dict(torch.load("checkpoints/cam_lidar_radar/best.pt", map_location="cuda")["model"])
ins, outs = model.export_io()
vla = VLAGrounder(lora="checkpoints/vla_lora", device="cuda")

sel = []
for split, toks in (("train", HAZARD), ("val", None)):
    ds = build_dataset(cfg, split)
    by_tok = {s["token"]: i for i, s in enumerate(ds.samples)}
    idxs = [by_tok[t] for t in toks] if toks else [0, 30, 60]
    sel += [(ds, i, split) for i in idxs]

os.makedirs("assets/samples", exist_ok=True)
for k, (ds, i, split) in enumerate(sel):
    info = ds.samples[i]
    d = f"assets/samples/{k:02d}_{info['token'][:8]}"
    os.makedirs(d, exist_ok=True)
    batch = collate([ds[i]])
    tens = export_inputs(model, batch)
    for name, t in zip(ins, tens):
        np.save(os.path.join(d, name + ".npy"), t.numpy())
    with torch.no_grad():
        out = model({kk: (v.cuda() if torch.is_tensor(v) else v) for kk, v in batch.items()})
    for name in outs:
        np.save(os.path.join(d, name + ".npy"), out[name].float().cpu().numpy())
    img = Image.open(info["cams"]["CAM_FRONT"]["path"]); img.thumbnail((704, 396)); img.save(os.path.join(d, "cam_front.jpg"))
    boxes = np.asarray(info["gt_boxes"], np.float32).reshape(-1, 9)
    keep = [(b, cfg.heads.det.classes.index(nm)) for b, nm in zip(boxes, info["gt_names"]) if nm in cfg.heads.det.classes]
    p = perception_json(objects_from_boxes([b for b, _ in keep], [l for _, l in keep], np.ones(len(keep)), cfg.heads.det.classes))
    act = vla(Image.open(info["cams"]["CAM_FRONT"]["path"]), p, "load gravel from the pile ahead")
    act.update({"perception_source": "gt", "lora": "checkpoints/vla_lora", "split": split,
                "nearest_pedestrian_in_path_m": p.get("nearest_pedestrian_in_path_m")})
    json.dump(act, open(os.path.join(d, "vla.json"), "w"), indent=1)
    print(d, "| ped_in_path:", p.get("nearest_pedestrian_in_path_m"), "| action:", act["action"])
shutil.copy("results/export/bevfusion.onnx", "assets/bevfusion.onnx")
print("assets ready")
