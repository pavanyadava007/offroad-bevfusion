"""PyTorch -> ONNX (opset 17, static shapes, batch 1).
python -m obf.export.onnx_export --cfg configs/base.yaml --ckpt checkpoints/cam_lidar_radar/best.pt --out results/export/bevfusion.onnx"""
import argparse
import inspect
import os

import numpy as np
import torch
import torch.nn as nn

from ..config import load_cfg
from ..models import BEVFusion
from ..utils.fake_batch import fake_batch


class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model
        self.ins, self.outs = model.export_io()
        self.eval()

    def forward(self, *args):
        batch = dict(zip(self.ins, args))
        if "cam_valid" in batch:
            batch["cam_valid"] = batch["cam_valid"] > 0.5  # float input -> bool
        out = self.m(batch)
        return tuple(out[k] for k in self.outs)


def export_inputs(model, batch):
    """Batch dict -> ordered tuple of export tensors (cam_valid as float32, indices int64)."""
    ins, _ = model.export_io()
    t = []
    for k in ins:
        v = batch[k]
        t.append(v.float() if k == "cam_valid" else v)
    return tuple(t)


def export(cfg, model, path, opset=17, sample=None, simplify=True, check=True):
    wrapper = ExportWrapper(model).cpu()
    batch = sample or fake_batch(cfg, B=1)
    inputs = export_inputs(model, batch)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    kw = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}  # torch<2.5 has no dynamo arg
    torch.onnx.export(wrapper, inputs, path, opset_version=opset, input_names=wrapper.ins, output_names=wrapper.outs,
                      dynamic_axes=None, do_constant_folding=True, **kw)
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as sim
            m, ok = sim(onnx.load(path))
            if ok:
                onnx.save(m, path)
        except Exception as e:
            print("onnxsim skipped:", e)
    wrapper.eval()  # torch.onnx.export restores the wrapper's original mode; force eval for the parity check
    if check:
        import onnx
        onnx.checker.check_model(path)
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            feed = {n: t.numpy() for n, t in zip(wrapper.ins, inputs)}
            ort_out = sess.run(None, feed)
            with torch.no_grad():
                ref = wrapper(*inputs)
            err = max(float(np.abs(o - r.numpy()).max()) for o, r in zip(ort_out, ref))
            print(f"ORT vs torch max abs err: {err:.2e}")
        except Exception as e:
            print("onnxruntime check skipped:", e)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="results/export/bevfusion.onnx"); ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--opts", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.opts)
    model = BEVFusion(cfg)
    if a.ckpt:
        model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    print("exported:", export(cfg, model, a.out, a.opset))


if __name__ == "__main__":
    main()
