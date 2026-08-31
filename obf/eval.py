"""Evaluation: nuScenes mAP/NDS (official devkit), BEV-seg IoU, Occ3D mIoU; subsets (rain/night/per-scene);
radar-dropout robustness. python -m obf.eval --cfg ... --ckpt ... [--subset rain] [--radar_drop 1.0] --name X"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_cfg
from .data import build_dataset, collate
from .data.common import DEFAULT_ATTR
from .utils.misc import load_json, save_json, to_device


# ---- nuScenes detection ----------------------------------------------------------------------------------------
def boxes_to_global(boxes, scores, labels, class_names, ego2global, token):
    R, t = ego2global[:3, :3], ego2global[:3, 3]
    yaw_e = float(np.arctan2(R[1, 0], R[0, 0]))
    from pyquaternion import Quaternion
    out = []
    for b, s, l in zip(boxes, scores, labels):
        c = R @ b[:3] + t
        name = class_names[int(l)]
        out.append({"sample_token": token, "translation": c.tolist(), "size": [float(b[3]), float(b[4]), float(b[5])],
                    "rotation": Quaternion(axis=[0, 0, 1], angle=float(b[6]) + yaw_e).elements.tolist(),
                    "velocity": (R[:2, :2] @ b[7:9]).tolist(), "detection_name": name, "detection_score": float(s),
                    "attribute_name": DEFAULT_ATTR[name]})
    return out


def nusc_detection_metrics(cfg, results_json, out_dir, tokens=None, eval_set=None):
    from nuscenes import NuScenes
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval
    nusc = NuScenes(cfg.data.version, cfg.data.root, verbose=False)
    eval_set = eval_set or ("mini_val" if "mini" in cfg.data.version else "val")
    ev = DetectionEval(nusc, config_factory("detection_cvpr_2019"), results_json, eval_set, out_dir, verbose=False)
    if tokens is not None:
        tokens = [t for t in tokens if t in ev.gt_boxes.sample_tokens]
        gt, pr = EvalBoxes(), EvalBoxes()
        for t in tokens:
            gt.add_boxes(t, ev.gt_boxes[t]); pr.add_boxes(t, ev.pred_boxes[t] if t in ev.pred_boxes.sample_tokens else [])
        ev.gt_boxes, ev.pred_boxes, ev.sample_tokens = gt, pr, tokens
    metrics, _ = ev.evaluate()
    s = metrics.serialize()
    return {"mAP": s["mean_ap"], "NDS": s["nd_score"], "per_class_AP": s["mean_dist_aps"], "tp_errors": s["tp_errors"]}


# ---- seg / occ accumulators --------------------------------------------------------------------------------------
class IoU:
    def __init__(self, C):
        self.i, self.u = np.zeros(C), np.zeros(C)

    def add(self, pred, gt):  # bool [B,C,Y,X]
        p, g = pred.bool(), gt.bool()
        self.i += (p & g).sum((0, 2, 3)).cpu().numpy(); self.u += (p | g).sum((0, 2, 3)).cpu().numpy()

    def result(self):
        return self.i / np.maximum(self.u, 1)


class ConfMat:
    def __init__(self, C):
        self.C, self.m = C, np.zeros((C, C), np.int64)

    def add(self, pred, gt, ignore=255):  # long tensors, same shape
        m = gt != ignore
        idx = gt[m].long() * self.C + pred[m].long()
        self.m += np.bincount(idx.cpu().numpy(), minlength=self.C * self.C).reshape(self.C, self.C)

    def iou(self):
        tp = np.diag(self.m); return tp / np.maximum(self.m.sum(0) + self.m.sum(1) - tp, 1)


@torch.no_grad()
def run_inference(cfg, model, dl, dev, radar_drop=0.0, return_out=False):
    model.eval()
    ds = dl.dataset
    info_by_tok = {s["token"]: s for s in getattr(ds, "samples", [])}
    det_json, outs = {}, {}
    seg_cls = cfg.heads.seg.classes if cfg.heads.get("seg") else []
    multilabel = cfg.heads.seg.get("multilabel", True) if seg_cls else True
    seg_iou = IoU(len(seg_cls)) if (seg_cls and multilabel) else (ConfMat(len(seg_cls)) if seg_cls else None)
    occ_cm = ConfMat(cfg.heads.occ.classes) if cfg.heads.get("occ") else None
    for batch in dl:
        batch = to_device(batch, dev)
        if radar_drop > 0 and "radar_feats" in batch:
            keep = (torch.rand(batch["radar_feats"].shape[0], device=dev) >= radar_drop)
            batch["radar_feats"] = batch["radar_feats"] * keep[:, None, None, None]
            batch["radar_num"] = batch["radar_num"] * keep[:, None]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda" and cfg.train.get("amp", True))):
            out = model(batch)
        if "det" in model.heads and cfg.get("dataset", "nuscenes") == "nuscenes":
            for b, d in enumerate(model.decode_det(out)):
                tok = batch["token"][b]
                info = info_by_tok[tok]
                det_json[tok] = boxes_to_global(d["boxes"].cpu().numpy(), d["scores"].cpu().numpy(), d["labels"].cpu().numpy(),
                                                cfg.heads.det.classes, np.asarray(info["ego2global"]), tok)
        if seg_iou is not None:
            if multilabel:
                seg_iou.add(out["seg"].sigmoid() > 0.5, batch["seg"])
            else:
                seg_iou.add(out["seg"].argmax(1), batch["seg"])
        if occ_cm is not None:
            occ_cm.add(out["occ"].argmax(1).permute(0, 2, 3, 1), batch["occ"])
        if return_out:
            for b, tok in enumerate(batch["token"]):
                outs[tok] = {k: v[b].float().cpu() for k, v in out.items()}
    metrics = {}
    if seg_iou is not None:
        iou = seg_iou.result() if multilabel else seg_iou.iou()
        metrics["seg_IoU"] = {c: float(v) for c, v in zip(seg_cls, iou)}
        metrics["seg_mIoU"] = float(np.mean(iou))
    if occ_cm is not None:
        iou = occ_cm.iou()
        metrics["occ_IoU"] = [float(v) for v in iou]
        metrics["occ_mIoU"] = float(np.mean(iou[: cfg.heads.occ.classes - 1]))  # exclude 'free'
    return det_json, metrics, outs


def evaluate_loader(cfg, model, dl, dev, name, subset_tokens=None, radar_drop=0.0, nusc_eval=True):
    det_json, metrics, _ = run_inference(cfg, model, dl, dev, radar_drop)
    out_dir = os.path.join("results", "eval", name)
    os.makedirs(out_dir, exist_ok=True)
    if det_json and nusc_eval:
        # devkit requires every eval-set token present
        tokens_all = [s["token"] for s in dl.dataset.samples]
        res = {"meta": {"use_camera": "cam" in cfg.model.modalities, "use_lidar": "lidar" in cfg.model.modalities,
                        "use_radar": "radar" in cfg.model.modalities, "use_map": False, "use_external": False},
               "results": {t: det_json.get(t, []) for t in tokens_all}}
        rp = os.path.join(out_dir, "detections.json")
        save_json(res, rp)
        split = getattr(dl.dataset, "split", "val")
        eval_set = ("mini_" + split) if "mini" in cfg.data.version else split
        try:
            metrics.update(nusc_detection_metrics(cfg, rp, out_dir, subset_tokens, eval_set=eval_set))
        except Exception as e:  # devkit missing / partial data
            metrics["nusc_eval_error"] = str(e)
    metrics.update({"name": name, "radar_drop": radar_drop, "modalities": list(cfg.model.modalities), "n_samples": len(dl.dataset)})
    save_json(metrics, os.path.join("results", f"{name}.json"))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val"); ap.add_argument("--subset", default="all")
    ap.add_argument("--radar_drop", type=float, default=0.0); ap.add_argument("--name", required=True)
    ap.add_argument("--opts", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.opts)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from .models import BEVFusion
    model = BEVFusion(cfg).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])
    tokens = None
    if a.subset != "all":
        tokens = load_json(os.path.join("data/splits", f"{a.split}_{a.subset}.json"))
    ds = build_dataset(cfg, a.split, tokens)
    dl = DataLoader(ds, batch_size=cfg.data.batch_size, num_workers=cfg.data.workers, collate_fn=collate)
    m = evaluate_loader(cfg, model, dl, dev, a.name, subset_tokens=tokens, radar_drop=a.radar_drop)
    print({k: v for k, v in m.items() if not isinstance(v, (dict, list))})


if __name__ == "__main__":
    main()
