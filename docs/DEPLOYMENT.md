# Deployment

## Export chain
PyTorch (`BEVFusion`) → `ExportWrapper` (ordered tensor I/O) → ONNX opset 17, static batch 1 → onnxsim → TensorRT 10 builder
(Python API on Colab T4: FP16; INT8 entropy calibration on 100 dumped frames, cache file) → C++ `TrtRunner` / ROS 2 node.

Inputs (batch 1): `imgs[1,6,3,256,704] f32`, `cam_bev_idx[1,185856] i64`, `cam_valid[1,185856] f32`,
`lidar_feats[1,20000,20,5] f32`, `lidar_num[1,20000] i64`, `lidar_coors[1,20000,2] i64`, `radar_feats[1,1500,12,7]`, `radar_num`, `radar_coors`.
Outputs: `hm[1,10,200,200]`, `reg[1,10,200,200]`, `seg[1,3,200,200]`, `occ[1,18,16,200,200]`.

Pre-processing (image resize/normalise, LSS index, pillarisation) runs on CPU in `obf.data`; the ROS node replays dumped
tensors — a live pipeline would port `voxelize` and `cam_to_bev_index` to C++ (both are <60 lines of index arithmetic).

## Latency protocol
`obf.export.latency_bench`: same input frame, 30 warm-up + 200 timed iterations, CUDA-synchronised; rows PyTorch FP32,
PyTorch AMP, ORT CUDA EP, TRT FP16, TRT INT8. Output `results/latency_l4.md` (GPU name recorded in the file; currently a local NVIDIA L4). **Jetson Orin: not measured (no hardware).**

Measured C++ runner (obf_runner, FP16 engine, local NVIDIA L4, 200 iters after 20 warm-up):
`engine=results/export/bevfusion_fp16.engine iters=200 mean=13.9956ms p50=13.9663ms p99=14.6207ms`

## Environment pins (verified 2026-08-31, local NVIDIA L4)
* `tensorrt>=10,<11` — **TensorRT 11 removed the implicit-quantisation INT8 API** (`IInt8EntropyCalibrator2`,
  `IInt8MinMaxCalibrator`, the INT8/FP16 builder flags and `EXPLICIT_BATCH`); `trt_build.py` targets the TRT-10 API.
  Verified with 10.16. Engines additionally need the standard plugin registry (`init_libnvinfer_plugins`) on both build
  and load — `ScatterElements(reduction=add)` lowers to the bundled ScatterReduction plugin.
* `onnxruntime-gpu` — the CUDA EP **cannot run this graph end-to-end**: the pillar feature nets' BatchNormalization on
  [B,P,N,C]-shaped activations hits `CUDNN_STATUS_NOT_SUPPORTED` (ORT 1.23 / cuDNN 9). CPU EP works; the latency table's
  ORT row is therefore omitted rather than estimated.
* `transformers>=4.49,<6` — the Qwen2.5-VL grounding code is verified against 5.16.
* `numpy<2` — required by the nuScenes devkit stack; CUDA-torch installs tend to drag numpy 2.x in, re-pin after.

## INT8 quantisation findings (L4, TRT 10.16, 100-frame PTQ)
* FP16 is essentially lossless (hm max err 5.5e-3, seg/occ argmax agreement 0.9996/0.9970).
* Naive full-INT8 PTQ fails the accuracy gate with either calibrator (entropy2: hm err 0.209, occ agreement 0.901;
  minmax: 0.128/0.881). Replacing the PFN's `masked_fill(-1e4)` with mask multiplication (bit-identical op,
  `tests/test_model.py::test_pfn_mask_mul_equals_masked_fill`) did **not** move these numbers — the -1e4 sentinel was
  not the bottleneck; the error is distributed across the quantised conv trunk.
* Best configuration: minmax calibration + fp16-pinned quantisation-sensitive layers (task heads, deformable BEV
  encoder, pillar feature nets, LSS scatter path, fuser — `trt_build.py --int8_heads_fp16`; conv backbones stay INT8):
  hm err 0.070, seg agreement 0.978, occ agreement 0.894. Raw-agreement thresholds (>= 0.97 / <= 0.05) are still not met,
  but the **end-task cost is negligible**: mAP +0.0002, NDS -0.0010, seg mIoU -0.0031, occ mIoU -0.0039 vs PyTorch
  (results/cam_lidar_radar_trt_int8.json) — the argmax flips sit in low-confidence voxels.
* INT8 buys only ~0.5 ms over FP16 here (14.7 vs 15.2 ms) because most sensitive layers run fp16; on this model/GPU
  FP16 is the pragmatic deployment choice, INT8 is available with the documented caveat. Full history:
  results/int8_remediation.json.

## DVC remote restore (backup: private HF dataset)
The DVC cache (checkpoints, engines, dumped samples — 8.4 GB) is backed up to the private dataset repo
`pavanyadava07/offroad-bevfusion-dvc` (a byte-for-byte copy of the local `files/md5` remote). To restore on a new machine:
```bash
pip install "huggingface_hub>=0.26" && export HF_TOKEN=<read token>
python -c "from huggingface_hub import snapshot_download as s; s('pavanyadava07/offroad-bevfusion-dvc', repo_type='dataset', local_dir='/path/dvc-remote-obf')"
dvc remote add --local restored /path/dvc-remote-obf && dvc pull -r restored --allow-missing
```

## ROS 2 Humble
`obf_node` (C++): TRT inference at 10 Hz, CenterPoint decode on host, publishes `/obf/detections` (vision_msgs/Detection3DArray),
`/obf/markers`, `/obf/drivable` and `/obf/occ_topdown` (nav_msgs/OccupancyGrid). `replay_sensors.py` publishes PointCloud2 + CAM_FRONT.
Build: `colcon build --cmake-args -DTENSORRT_ROOT=<trt>`; launch: `ros2 launch obf_ros replay.launch.py`.
