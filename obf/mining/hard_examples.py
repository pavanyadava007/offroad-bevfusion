"""Per-scene mAP hard-example mining -> sampling weights for the next training round.
Mining is meant to run on TRAIN predictions (val stays untouched): produce them with
`python -m obf.eval --cfg ... --ckpt ... --name <x>_train --split train` (dvc stage `eval_train`).
The split is auto-detected from the tokens in the detections file; a val file still works but only weights val scenes.
python -m obf.mining.hard_examples --cfg configs/base.yaml --det results/eval/cam_lidar_radar_train/detections.json --k 3"""
import argparse
import os
import pickle

import numpy as np

from ..config import load_cfg
from ..eval import nusc_detection_metrics
from ..utils.misc import load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--det", required=True)
    ap.add_argument("--k", type=int, default=3, help="hardest scenes to up-weight"); ap.add_argument("--boost", type=float, default=3.0)
    ap.add_argument("--out", default="data/splits/hard_scenes.json")
    ap.add_argument("--opts", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.opts)
    infos = pickle.load(open(os.path.join(cfg.data.infos, f"nuscenes_{cfg.data.version}_infos.pkl"), "rb"))
    det_tokens = set(load_json(a.det)["results"])
    split = "train" if det_tokens & {s["token"] for s in infos["train"]} else "val"
    if split != "train":
        print("WARNING: detections are from the val split; weights will only cover val scenes (mine on --split train instead).")
    eval_set = ("mini_" + split) if "mini" in cfg.data.version else split
    by_scene = {}
    for s in infos[split]:
        by_scene.setdefault(s["scene_token"], {"name": s["scene_name"], "desc": s["scene_description"], "tokens": []})["tokens"].append(s["token"])
    per_scene = {}
    for st, sc in by_scene.items():
        try:
            m = nusc_detection_metrics(cfg, a.det, "results/mining_tmp", sc["tokens"], eval_set=eval_set)
        except Exception as e:
            print(f"scene {sc['name']} skipped: {e}")
            continue
        per_scene[st] = {"name": sc["name"], "desc": sc["desc"], "mAP": m["mAP"], "NDS": m["NDS"], "n": len(sc["tokens"])}
    ranked = sorted(per_scene.items(), key=lambda kv: kv[1]["mAP"])
    weights = {st: (a.boost if i < a.k else 1.0) for i, (st, _) in enumerate(ranked)}
    save_json({"split": split, "per_scene": per_scene, "hardest": [st for st, _ in ranked[: a.k]], "scene_weights": weights}, a.out)
    print("hardest scenes:", [(v["name"], round(v["mAP"], 3)) for _, v in ranked[: a.k]], "mean mAP", np.mean([v["mAP"] for v in per_scene.values()]))


if __name__ == "__main__":
    main()
