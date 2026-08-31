import os

import pytest
import torch

from obf.config import load_cfg
from obf.export.onnx_export import ExportWrapper, export
from obf.models import BEVFusion
from obf.utils.fake_batch import fake_batch


def test_export_wrapper_matches_model():
    cfg = load_cfg("configs/tiny.yaml")
    m = BEVFusion(cfg).eval(); w = ExportWrapper(m)
    b = fake_batch(cfg)
    ins = [b[k].float() if k == "cam_valid" else b[k] for k in w.ins]
    with torch.no_grad():
        a = w(*ins); ref = m(b)
    assert all(torch.allclose(x, ref[k]) for x, k in zip(a, w.outs))


def test_onnx_export_opset17(tmp_path):
    pytest.importorskip("onnx")
    cfg = load_cfg("configs/tiny.yaml")
    p = export(cfg, BEVFusion(cfg), str(tmp_path / "tiny.onnx"), opset=17, simplify=False, check=True)
    assert os.path.exists(p)
