"""INT8/FP16 accuracy sanity check: compare TRT engine outputs vs PyTorch on dumped frames (heatmap max-abs err,
seg IoU agreement, occ argmax agreement). python -m obf.export.accuracy_check --engine ... --frames data/samples/calib --cfg ... --ckpt ..."""
import argparse
import glob
import os

import numpy as np
import torch

from ..config import load_cfg
from ..models import BEVFusion
from ..utils.misc import save_json
from .onnx_export import ExportWrapper
from .trt_runtime import TRTModule, feed_from_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--engine", required=True)
    ap.add_argument("--frames", required=True); ap.add_argument("--n", type=int, default=20); ap.add_argument("--out", default=None)
    ap.add_argument("--opts", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.opts)
    model = BEVFusion(cfg).cuda().eval(); model.load_state_dict(torch.load(a.ckpt, map_location="cuda")["model"])
    w = ExportWrapper(model).cuda(); trt = TRTModule(a.engine)
    agg = {"hm_max_abs_err": [], "seg_agreement": [], "occ_agreement": []}
    for d in sorted(glob.glob(os.path.join(a.frames, "*_*")))[: a.n]:
        feed = feed_from_dir(d, w.ins)
        with torch.no_grad():
            ref = dict(zip(w.outs, w(*[torch.from_numpy(feed[n]).cuda() for n in w.ins])))
        out = trt(feed)
        if "hm" in ref:
            agg["hm_max_abs_err"].append(float((ref["hm"].sigmoid() - out["hm"].float().sigmoid()).abs().max()))
        if "seg" in ref:
            agg["seg_agreement"].append(float(((ref["seg"] > 0) == (out["seg"].float() > 0)).float().mean()))
        if "occ" in ref:
            agg["occ_agreement"].append(float((ref["occ"].argmax(1) == out["occ"].float().argmax(1)).float().mean()))
    res = {k: float(np.mean(v)) for k, v in agg.items() if v}
    print(res)
    if a.out:
        save_json(res, a.out)


if __name__ == "__main__":
    main()
