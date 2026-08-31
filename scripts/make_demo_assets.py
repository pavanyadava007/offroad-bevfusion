"""assets/samples for the demo: hazard + pedestrian-rich + clear frames. Per frame: ONNX input .npy, model output .npy,
cam_front.jpg with projected GT pedestrian boxes (green=GT annotation), vla.json (LoRA VLM, GT perception) carrying
gt_pedestrians [{x,y,dist,in_path}] so the app can highlight them on the BEV."""
import json
import os
import shutil

import numpy as np
import torch
from PIL import Image, ImageDraw

from obf.config import load_cfg
from obf.data import build_dataset, collate
from obf.export.onnx_export import export_inputs
from obf.models import BEVFusion
from obf.vla.grounding import VLAGrounder
from obf.vla.perception_json import objects_from_boxes, perception_json

FRAME_SET = [("train", "c923fe08b2ff4e27975d2bf30934383b"), ("train", "e0845f5322254dafadbbed75aaa07969"),  # <5 m hazards
             ("val", "a98fba72"), ("val", "b6b0d9f2"), ("val", "de7593d7"),                                # ped crowds (scene-0103)
             ("val", None), ("val", None)]                                                                  # clear scenes


def project_box(b, cam):  # ego box [9] -> image-plane corner points (or None)
    x, y, z, w, l, h, yaw = b[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    corners = []
    for dx, dy, dz in [(a, b2, c2) for a in (l / 2, -l / 2) for b2 in (w / 2, -w / 2) for c2 in (h / 2, -h / 2)]:
        corners.append([x + dx * c - dy * s, y + dx * s + dy * c, z + dz])
    P = np.asarray(corners).T  # [3,8] ego
    E = np.linalg.inv(np.asarray(cam["cam2ego"]))
    Pc = E[:3, :3] @ P + E[:3, 3:4]
    if (Pc[2] < 0.5).all():
        return None
    Pc = Pc[:, Pc[2] > 0.5]
    uv = np.asarray(cam["intrinsic"]) @ Pc
    uv = uv[:2] / uv[2:]
    return uv.min(1), uv.max(1)


def main():
    cfg = load_cfg("configs/base.yaml")
    model = BEVFusion(cfg).cuda().eval()
    model.load_state_dict(torch.load("checkpoints/cam_lidar_radar/best.pt", map_location="cuda")["model"])
    ins, outs = model.export_io()
    vla = VLAGrounder(lora="checkpoints/vla_lora", device="cuda")
    ds = {sp: build_dataset(cfg, sp) for sp in ("train", "val")}
    by_tok = {sp: {s["token"]: i for i, s in enumerate(d.samples)} for sp, d in ds.items()}
    clear_idx = iter([0, 60])  # scene-0916 parking lot frames
    os.makedirs("assets/samples", exist_ok=True)
    for k, (split, tok) in enumerate(FRAME_SET):
        d0 = ds[split]
        if tok is None:
            i = next(clear_idx)
        elif len(tok) == 8:
            i = next(v for t, v in by_tok[split].items() if t.startswith(tok))
        else:
            i = by_tok[split][tok]
        info = d0.samples[i]
        out_dir = f"assets/samples/{k:02d}_{info['token'][:8]}"
        os.makedirs(out_dir, exist_ok=True)
        batch = collate([d0[i]])
        for name, t in zip(ins, export_inputs(model, batch)):
            np.save(os.path.join(out_dir, name + ".npy"), t.numpy())
        with torch.no_grad():
            o = model({kk: (v.cuda() if torch.is_tensor(v) else v) for kk, v in batch.items()})
        for name in outs:
            np.save(os.path.join(out_dir, name + ".npy"), o[name].float().cpu().numpy())
        boxes = np.asarray(info["gt_boxes"], np.float32).reshape(-1, 9)
        cam = info["cams"]["CAM_FRONT"]
        img = Image.open(cam["path"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        peds = []
        for b, nm in zip(boxes, info["gt_names"]):
            if nm != "pedestrian":
                continue
            dist = float(np.hypot(b[0], b[1]))
            in_path = bool(0 < b[0] < 20 and abs(b[1]) < 2.5)
            peds.append({"x": round(float(b[0]), 1), "y": round(float(b[1]), 1), "dist": round(dist, 1), "in_path": in_path})
            pr = project_box(b, cam)
            if pr is not None:
                (u0, v0), (u1, v1) = pr
                col = (230, 103, 103) if (in_path and dist < 10) else (25, 158, 112)
                draw.rectangle([u0, v0, u1, v1], outline=col, width=4 if in_path else 2)
                draw.text((u0, max(0, v0 - 16)), f"{dist:.1f} m", fill=col)
        img.thumbnail((896, 504)); img.save(os.path.join(out_dir, "cam_front.jpg"), quality=88)
        keep = [(b, cfg.heads.det.classes.index(nm)) for b, nm in zip(boxes, info["gt_names"]) if nm in cfg.heads.det.classes]
        p = perception_json(objects_from_boxes([b for b, _ in keep], [l for _, l in keep], np.ones(len(keep)), cfg.heads.det.classes))
        act = vla(Image.open(cam["path"]), p, "load gravel from the pile ahead")
        act.update({"perception_source": "gt", "lora": "checkpoints/vla_lora", "split": split,
                    "nearest_pedestrian_in_path_m": p.get("nearest_pedestrian_in_path_m"),
                    "gt_pedestrians": sorted(peds, key=lambda q: q["dist"])[:40]})
        json.dump(act, open(os.path.join(out_dir, "vla.json"), "w"), indent=1)
        print(out_dir, "| peds:", len(peds), "| nearest in-path:", p.get("nearest_pedestrian_in_path_m"), "| action:", act["action"])
    shutil.copy("results/export/bevfusion.onnx", "assets/bevfusion.onnx")
    print("assets ready")


if __name__ == "__main__":
    main()
