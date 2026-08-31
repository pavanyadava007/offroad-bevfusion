"""python -m obf.train --cfg configs/base.yaml [--opts train.epochs=6 data.batch_size=1]"""
import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import load_cfg
from .data import build_dataset, collate
from .eval import evaluate_loader
from .models import BEVFusion
from .utils.misc import save_json, seed_all, to_device


def build_loader(cfg, split, shuffle):
    ds = build_dataset(cfg, split)
    sampler = None
    if shuffle and getattr(ds, "weights", None):
        sampler = WeightedRandomSampler(ds.weights, num_samples=len(ds), replacement=True)
    return DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=cfg.data.workers, collate_fn=collate, pin_memory=True, drop_last=shuffle)


def load_init(model, path):
    """Load backbone/fusion weights from a nuScenes checkpoint, skipping heads with mismatched shapes (GOOSE transfer)."""
    sd = torch.load(path, map_location="cpu")["model"]
    own = model.state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(ok, strict=False)
    return len(ok), len(own)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/base.yaml")
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.cfg, args.opts)
    seed_all(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = BEVFusion(cfg).to(dev)
    if cfg.train.get("init_from") and os.path.exists(cfg.train.init_from):
        print("init_from:", load_init(model, cfg.train.init_from))
    train_dl, val_dl = build_loader(cfg, "train", True), build_loader(cfg, "val", False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
    total = cfg.train.epochs * len(train_dl)
    warm = cfg.train.get("warmup_iters", 0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * it / max(1, total))))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and dev == "cuda")
    start, best, it = 0, -1.0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start, best, it = ck["epoch"] + 1, ck.get("best", -1), ck.get("it", 0)
    wb = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        wb = wandb.init(project=cfg.train.wandb_project, name=cfg.name, config=dict(cfg))
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    for ep in range(start, cfg.train.epochs):
        model.train()
        t0 = time.time()
        for i, batch in enumerate(train_dl):
            batch = to_device(batch, dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                out = model(batch)
            L = model.loss(out, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(L["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt); scaler.update(); sched.step(); it += 1
            if i % 20 == 0:
                msg = {k: round(float(v.detach()), 4) for k, v in L.items()}
                msg.update({f"w_{k}": round(v, 3) for k, v in model.weighting.weights().items()})
                print(f"ep{ep} it{i}/{len(train_dl)} lr={sched.get_last_lr()[0]:.2e}", msg)
                if wb:
                    wb.log({**msg, "lr": sched.get_last_lr()[0], "it": it})
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "cfg": dict(cfg), "best": best, "it": it}
        torch.save(ck, os.path.join(cfg.train.ckpt_dir, "last.pt"))
        if (ep + 1) % cfg.train.get("eval_every", 1) == 0 or ep + 1 == cfg.train.epochs:
            metrics = evaluate_loader(cfg, model, val_dl, dev, name=f"{cfg.name}_ep{ep}", nusc_eval=True)
            key = metrics.get("NDS", metrics.get("seg_mIoU", 0.0))
            print(f"epoch {ep} ({time.time() - t0:.0f}s):", {k: v for k, v in metrics.items() if not isinstance(v, dict)})
            if wb:
                wb.log({f"val/{k}": v for k, v in metrics.items() if not isinstance(v, dict)})
            if key > best:
                best = key; ck["best"] = best
                torch.save(ck, os.path.join(cfg.train.ckpt_dir, "best.pt"))
                save_json(metrics, os.path.join(cfg.train.ckpt_dir, "best_metrics.json"))


if __name__ == "__main__":
    main()
