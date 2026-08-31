"""Latency table on one GPU (local NVIDIA L4): PyTorch FP32 / PyTorch AMP-FP16 / ONNX Runtime CUDA / TRT FP16 / TRT INT8.
python -m obf.export.latency_bench --cfg configs/base.yaml --ckpt ... --trt configs/trt.yaml --sample data/samples/calib/0000_*"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import yaml

from ..config import load_cfg
from ..models import BEVFusion
from ..utils.misc import md_table
from .onnx_export import ExportWrapper
from .trt_runtime import TRTModule, feed_from_dir


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return ts.mean(), np.percentile(ts, 50), np.percentile(ts, 99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--ckpt", default=None); ap.add_argument("--trt", default="configs/trt.yaml")
    ap.add_argument("--sample", required=True, help="frame dir with <input>.npy")
    ap.add_argument("--opts", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.opts); tc = yaml.safe_load(open(a.trt))
    iters, warm = tc["bench"]["iters"], tc["bench"]["warmup"]
    sample = glob.glob(a.sample)[0] if "*" in a.sample else a.sample
    model = BEVFusion(cfg).cuda().eval()
    if a.ckpt:
        model.load_state_dict(torch.load(a.ckpt, map_location="cuda")["model"])
    w = ExportWrapper(model).cuda()
    feed = feed_from_dir(sample, w.ins)
    tens = [torch.from_numpy(feed[n]).cuda() for n in w.ins]
    rows = []
    with torch.no_grad():
        rows.append(["PyTorch FP32", *timeit(lambda: w(*tens), iters, warm)])
        with torch.autocast("cuda", dtype=torch.float16):
            rows.append(["PyTorch AMP FP16", *timeit(lambda: w(*tens), iters, warm)])
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(tc["onnx"], providers=["CUDAExecutionProvider"])
        rows.append(["ONNX Runtime (CUDA EP)", *timeit(lambda: sess.run(None, feed), iters, warm)])
    except Exception as e:
        print("ORT skipped:", e)
    for prec, path in tc["engines"].items():
        if os.path.exists(path):
            m = TRTModule(path)
            rows.append([f"TensorRT {prec.upper()}", *timeit(lambda m=m: m(feed), iters, warm)])
    rows = [[r[0], f"{r[1]:.1f}", f"{r[2]:.1f}", f"{r[3]:.1f}"] for r in rows]
    gpu = torch.cuda.get_device_name(0)
    md = f"GPU: {gpu}, batch 1, {iters} iters ({warm} warm-up). Inputs: {', '.join(w.ins)}.\n\n" + md_table(rows, ["Runtime", "mean ms", "p50 ms", "p99 ms"])
    md += "\n\n_Jetson Orin: **not measured** (no hardware available); engines built and timed on the GPU named above (local L4) only._\n"
    os.makedirs(os.path.dirname(tc["bench"]["out"]), exist_ok=True)
    open(tc["bench"]["out"], "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
