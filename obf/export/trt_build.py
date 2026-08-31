"""ONNX -> TensorRT FP16 / INT8 (entropy PTQ, 100 calibration frames) with the TensorRT *Python builder API*
(works in Colab with `pip install tensorrt`). python -m obf.export.trt_build --trt configs/trt.yaml --precision fp16|int8"""
import argparse
import glob
import os

import numpy as np
import yaml


def build(onnx_path, engine_path, precision, calib_dir=None, n_calib=100, cache=None, workspace_gb=4, input_names=None,
          calib_algo="entropy2", heads_fp16=False):
    import tensorrt as trt
    import torch
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")  # standard TRT plugin registry (ScatterElements reduction=add -> bundled ScatterReduction)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if precision in ("fp16", "int8"):
        cfg.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        cfg.set_flag(trt.BuilderFlag.INT8)
        if heads_fp16:  # per-layer fallback: run the quantization-sensitive layers (task heads + deformable BEV
            # encoder: grid_sample/softmax attention every head depends on) in fp16; conv encoders stay int8
            cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            n_pinned = 0
            for i in range(net.num_layers):
                layer = net.get_layer(i)
                skip_types = tuple(getattr(trt.LayerType, t) for t in ('CONSTANT', 'SHAPE', 'PLUGIN', 'PLUGIN_V2', 'PLUGIN_V3') if hasattr(trt.LayerType, t))
                floaty = (layer.type not in skip_types
                          and all(layer.get_output(o).dtype == trt.DataType.FLOAT for o in range(layer.num_outputs)))
                if floaty and any(t in layer.name for t in ("/m/det/", "/m/seg/", "/m/occ/", "/m/bev_enc/",
                                             "/m/lidar_pfn/", "/m/radar_pfn/", "/m/lss/", "/m/fuser/")):  # PFNs use masked_fill(-1e4) -> int8 range collapse
                    layer.precision = trt.DataType.HALF; n_pinned += 1
            print(f"heads_fp16: pinned {n_pinned} layers to fp16")
        if cache and calib_algo != "entropy2":
            cache = cache.replace(".cache", f"_{calib_algo}.cache")  # per-algorithm cache
        base_calib = trt.IInt8MinMaxCalibrator if calib_algo == "minmax" else trt.IInt8EntropyCalibrator2

        class Calib(base_calib):
            def __init__(self):
                super().__init__()
                self.frames = sorted(glob.glob(os.path.join(calib_dir, "*_*")))[:n_calib]
                self.names = [net.get_input(i).name for i in range(net.num_inputs)]
                self.i, self.bufs = 0, {}

            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                if self.i >= len(self.frames):
                    return None
                d = self.frames[self.i]; self.i += 1
                ptrs = []
                for n in names:
                    arr = np.load(os.path.join(d, n + ".npy"))
                    t = torch.from_numpy(np.ascontiguousarray(arr)).cuda()
                    self.bufs[n] = t  # keep alive
                    ptrs.append(int(t.data_ptr()))
                return ptrs

            def read_calibration_cache(self):
                return open(cache, "rb").read() if cache and os.path.exists(cache) else None

            def write_calibration_cache(self, c):
                if cache:
                    os.makedirs(os.path.dirname(cache), exist_ok=True); open(cache, "wb").write(c)

        cfg.int8_calibrator = Calib()
    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        raise RuntimeError("engine build failed")
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(plan)
    return engine_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trt", default="configs/trt.yaml"); ap.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    ap.add_argument("--calib_algo", choices=["entropy2", "minmax"], default=None, help="override calib.algorithm from the yaml")
    ap.add_argument("--int8_heads_fp16", action="store_true", help="pin head layers to fp16 in the int8 engine")
    a = ap.parse_args()
    c = yaml.safe_load(open(a.trt))
    out = build(c["onnx"], c["engines"][a.precision], a.precision, c["calib"]["dir"], c["calib"]["n_frames"], c["calib"]["cache"], c["workspace_gb"],
                calib_algo=a.calib_algo or c["calib"].get("algorithm", "entropy2"), heads_fp16=a.int8_heads_fp16)
    print("engine:", out)


if __name__ == "__main__":
    main()
