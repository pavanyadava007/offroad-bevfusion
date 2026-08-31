"""Optional LoRA fine-tune of Qwen2.5-VL-3B on (front image, perception JSON, task) -> teacher JSON action.
Teacher = hazard_rule + task heuristic (self-supervised from GT boxes on mini_train); hook for the 'Hypercritical'
critique/refine pipeline: pass --pairs <jsonl> with externally generated (prompt, target) pairs to override.
Fits a T4 (bf16 weights, r=16 LoRA on q/v, grad checkpointing, batch 1, images <= 512*28*28 px)."""
import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image

from ..config import load_cfg
from ..data import build_dataset
from .perception_json import hazard_rule, objects_from_boxes, perception_json
from .prompt import SYSTEM, user_prompt
from .safety_eval import TASKS

REASONS = {"stop": "pedestrian in path within 5 m", "wait_for_person": "pedestrian near path", "reverse": "path not drivable",
           "dump": "at truck, bucket loaded", "approach_pile": "path clear, heading to pile"}


def synth_pairs(images, n=300, seed=1):
    """Synthetic boundary augmentation: perception JSONs with a corridor pedestrian at U(0.5, 12) m (half in [3.5, 6.5]),
    random other objects, task sampled; label = hazard_rule. The JSON, not the image, carries the decision, so real
    frame images are reused."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        d = round(rng.uniform(3.5, 6.5) if rng.random() < 0.5 else rng.uniform(0.5, 12.0), 1)
        objs = [{"class": "pedestrian", "range_m": d, "bearing_deg": round(rng.uniform(-6, 6), 0),
                 "in_path": True, "vel_mps": round(rng.uniform(0, 1.5), 1), "score": round(rng.uniform(0.6, 1.0), 2)}]
        for _ in range(rng.randint(0, 6)):
            objs.append({"class": rng.choice(["car", "truck", "barrier", "traffic_cone", "pedestrian"]),
                         "range_m": round(rng.uniform(2.0, 35.0), 1), "bearing_deg": round(rng.uniform(-180, 180), 0),
                         "in_path": rng.random() < 0.2, "vel_mps": round(rng.uniform(0, 8), 1), "score": round(rng.uniform(0.3, 1.0), 2)})
        objs = sorted(objs, key=lambda o: o["range_m"])[:12]
        p = {"objects": objs,
             "nearest_pedestrian_in_path_m": min(o["range_m"] for o in objs if o["class"] == "pedestrian" and o["in_path"])}
        task = rng.choice(TASKS)
        act = hazard_rule(p) or ("dump" if "dump" in task else "approach_pile")
        tgt = {"action": act, "target": "truck" if act == "dump" else ("pile" if act == "approach_pile" else None), "reason": REASONS[act]}
        out.append({"image": rng.choice(images), "perception": p, "task": task, "target": json.dumps(tgt)})
    return out


def build_pairs(cfg, split="train", n=300, seed=0, exclude=(), synthetic=300):
    ds = build_dataset(cfg, split)
    rng = random.Random(seed)
    idx = [i for i in range(len(ds.samples)) if ds.samples[i]["token"] not in exclude]; rng.shuffle(idx)
    pairs = []
    for i in idx[:n]:
        s = ds.samples[i]
        boxes = np.asarray(s["gt_boxes"], np.float32).reshape(-1, 9)
        keep = [(b, cfg.heads.det.classes.index(nm)) for b, nm in zip(boxes, s["gt_names"]) if nm in cfg.heads.det.classes]
        objs = objects_from_boxes([b for b, _ in keep], [l for _, l in keep], np.ones(len(keep)), cfg.heads.det.classes)
        p = perception_json(objs)
        def pair(act, task, s=s, p=p):  # bind loop vars
            tgt = {"action": act, "target": "truck" if act == "dump" else ("pile" if act == "approach_pile" else None),
                   "reason": REASONS[act]}
            return {"image": s["cams"]["CAM_FRONT"]["path"], "perception": p, "task": task, "target": json.dumps(tgt)}

        rule = hazard_rule(p)
        if rule:  # rule-triggering frames are rare (mini: 1 stop / ~18 wait in 300) -> oversample: every task
            # (the rule overrides the task), and stop frames x5 on top, so the safety rule gets real gradient signal
            for _ in range(5 if rule == "stop" else 1):
                pairs += [pair(rule, t) for t in TASKS]
        else:
            task = rng.choice(TASKS)
            pairs.append(pair("dump" if "dump" in task else "approach_pile", task))
    if synthetic:
        images = [ds.samples[i]["cams"]["CAM_FRONT"]["path"] for i in idx]
        pairs += synth_pairs(images, n=synthetic, seed=seed + 1)
    rng.shuffle(pairs)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--pairs", default=None); ap.add_argument("--out", default="checkpoints/vla_lora")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct"); ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--exclude_tokens", nargs="*", default=[], help="held-out frame tokens (never trained on)")
    ap.add_argument("--synthetic", type=int, default=300, help="synthetic boundary-augmentation pairs (0 = off)")
    ap.add_argument("--opts", nargs="*", default=[])
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    cfg = load_cfg(a.cfg, a.opts)
    pairs = [json.loads(l) for l in open(a.pairs)] if a.pairs else build_pairs(cfg, exclude=set(a.exclude_tokens), synthetic=a.synthetic)
    proc = AutoProcessor.from_pretrained(a.model, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    for ep in range(a.epochs):
        random.shuffle(pairs)
        for k, ex in enumerate(pairs):
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [{"type": "image", "image": Image.open(ex["image"])},
                                                 {"type": "text", "text": user_prompt(json.dumps(ex["perception"], separators=(",", ":")), ex["task"])}]},
                    {"role": "assistant", "content": ex["target"]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            imgs, _ = process_vision_info(msgs)
            inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
            labels = inp.input_ids.clone()
            n_prompt = len(proc(text=[proc.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)], images=imgs, return_tensors="pt").input_ids[0])
            labels[:, :n_prompt] = -100  # loss only on the JSON answer
            loss = model(**inp, labels=labels).loss
            loss.backward(); opt.step(); opt.zero_grad()
            if k % 20 == 0:
                print(f"ep{ep} {k}/{len(pairs)} loss {loss.item():.3f}")
    os.makedirs(a.out, exist_ok=True); model.save_pretrained(a.out)
    print("saved LoRA to", a.out)


if __name__ == "__main__":
    main()
