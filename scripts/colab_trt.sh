#!/usr/bin/env bash
# Colab free T4: TensorRT engines via the pip wheel + Python builder API, latency table, accuracy check.
set -e
pip install -q tensorrt onnxruntime-gpu onnx onnxsim -e .
python -m obf.export.trt_build --trt configs/trt.yaml --precision fp16
python -m obf.export.trt_build --trt configs/trt.yaml --precision int8       # entropy calibration on 100 dumped frames
python -m obf.export.latency_bench --cfg configs/base.yaml --ckpt checkpoints/cam_lidar_radar/best.pt --sample "data/samples/calib/0000_*"
python -m obf.export.accuracy_check --cfg configs/base.yaml --ckpt checkpoints/cam_lidar_radar/best.pt --engine results/export/bevfusion_int8.engine --frames data/samples/calib --out results/int8_accuracy.json
# C++ runner (same engine):  cmake -S cpp -B build -DTENSORRT_ROOT=$(python -c "import tensorrt,os;print(os.path.dirname(tensorrt.__file__))") && cmake --build build -j
# ./build/obf_runner results/export/bevfusion_fp16.engine data/samples/replay/0000_* 200
