"""Minimal TensorRT-10 Python runtime using torch tensors as device buffers (no pycuda)."""
import numpy as np
import torch

_DT = {"DataType.FLOAT": torch.float32, "DataType.HALF": torch.float16, "DataType.INT32": torch.int32,
       "DataType.INT64": torch.int64, "DataType.BOOL": torch.bool, "DataType.INT8": torch.int8}


class TRTModule:
    def __init__(self, engine_path):
        import tensorrt as trt
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")  # engines use TRT's bundled ScatterReduction plugin
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.bufs, self.inputs, self.outputs = {}, [], []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(n))
            dt = _DT[str(self.engine.get_tensor_dtype(n))]
            self.bufs[n] = torch.empty(shape, dtype=dt, device="cuda")
            self.ctx.set_tensor_address(n, int(self.bufs[n].data_ptr()))
            (self.inputs if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT else self.outputs).append(n)

    def __call__(self, feed):
        for n, v in feed.items():
            t = torch.as_tensor(v)
            self.bufs[n].copy_(t.to(self.bufs[n].dtype), non_blocking=True)
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return {n: self.bufs[n] for n in self.outputs}


def feed_from_dir(d, names):
    import os
    return {n: np.load(os.path.join(d, n + ".npy")) for n in names}
