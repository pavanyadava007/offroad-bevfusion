# offroad-bevfusion

[![ci](https://github.com/pavanyadava007/offroad-bevfusion/actions/workflows/ci.yml/badge.svg)](https://github.com/pavanyadava007/offroad-bevfusion/actions) [![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20demo-HF%20Space-blue)](https://huggingface.co/spaces/pavanyadava07/offroad-bevfusion)

Camera + LiDAR + radar BEV multitask perception (3D detection · BEV segmentation · 3D occupancy) with a deformable-attention
BEV encoder and uncertainty-weighted multitask loss, TensorRT FP16/INT8 + ROS 2 Humble deployment, off-road (GOOSE) transfer,
adverse-weather evaluation, and a VLM task-grounding interface (Qwen2.5-VL-3B → wheel-loader action primitives).
Built on a zero-cost stack: Kaggle GPU (training), Colab T4 (TensorRT), DVC + Google Drive, W&B free tier, GitHub Actions, HF Spaces.

```mermaid
flowchart LR
  subgraph enc[Encoders]
    C[6 cams 256x704<br/>ResNet-18 + LSS] --> CB[cam BEV 80ch]
    L[LiDAR PointPillars 0.2 m<br/>400² canvas → SECOND-lite] --> LB[LiDAR BEV 128ch]
    R[Radar pillars 0.8 m<br/>5 sweeps, x y z v_x v_y RCS dt] --> RB[Radar BEV 32ch]
  end
  CB & LB & RB --> F[concat → ConvFuser 128ch]
  F --> D[2× deformable-attention BEV encoder<br/>200×200 @0.4 m]
  D --> H1[CenterPoint head<br/>10 classes]
  D --> H2[BEV seg head<br/>drivable / vehicle / pedestrian]
  D --> H3[Occupancy head<br/>200×200×16, 18 classes]
  H1 & H2 & H3 --> W[Kendall uncertainty weighting]
  H1 & H2 & H3 --> V[perception JSON] --> Q[Qwen2.5-VL-3B] --> A["{approach_pile, dump, wait_for_person, stop, reverse}"]
```

## Evidence map (JD requirement → deliverable → artefact)

| JD requirement | Deliverable | Evidence (produced by the pipeline) |
|---|---|---|
| 3D det / seg / occupancy | Multitask heads on one shared 200×200 BEV (`obf/models/heads`) | `docs/REPORT.md` §1: mAP / NDS / seg mIoU / occ mIoU |
| LiDAR–camera–radar fusion | Radar pillar branch + `ConvFuser` (`obf/models/encoders/pillars.py`, `fusion/bev_fuser.py`) | ablation cam / cam+L / cam+L+R; radar-dropout robustness |
| Transformer / BEV / multitask | `DeformBEVEncoder` (grid_sample MSDeformAttn, TRT-native), `UncertaintyWeighting` | `configs/ablation_no_deform.yaml`, `ablation_fixed_weights.yaml` rows |
| VLA | `obf/vla`: perception JSON → Qwen2.5-VL-3B → JSON action, fail-safe parser, optional LoRA | `results/vla_safety.json` (50-frame stop-recall eval) |
| TensorRT / ONNX / Jetson | ONNX opset 17 static; FP16 + INT8 entropy PTQ (Python builder API); C++ runner (`cpp/`) | `results/latency_l4.md` — **Orin not measured (no hardware)** |
| ROS / C++ / mmdetection | ROS 2 Humble C++ node → `Detection3DArray`, `OccupancyGrid`; CMake TRT lib; mmdet3d-compatible pillar encoders (`use_mmdet3d`) | `docs/rviz_replay.gif` (`scripts/make_rviz_gif.sh`) |
| Off-road / adverse weather | GOOSE BEV-seg transfer (`obf/data/goose.py`) — LiDAR-only (no calib in 2D/3D zips, camera branch zeroed); nuScenes rain/night/clear subsets (`obf/data/splits.py`) | `docs/REPORT.md` §3–4 domain-shift tables |
| Data leadership | `dvc.yaml` (14 stages), per-scene mAP hard-example mining + weighted resampling, auto-labelling, dataset card | this repo, `docs/DATASET_CARD.md` |
| Publications | arXiv:2110.00791, thesis (SparseDrive, ISO 21448) | CV; SOTIF mapping in `docs/VLA_SAFETY.md` |

## Status

| Component | State |
|---|---|
| Code: data pipeline, model, training, eval, export, TRT, C++, ROS 2, VLA, DVC, CI | complete — CI runs unit tests, ONNX opset-17 export with ORT parity (4.8e-7), the full train→eval→dump→VLA-eval loop on synthetic data (`dataset=fake`), and a C++ syntax check |
| nuScenes-mini training / ablation numbers | done on a local NVIDIA L4 (12 ep each): cam 0.022/0.059 → cam+L 0.047/0.099 → cam+L+R 0.053/0.108 (mAP/NDS), seg mIoU 0.302, occ mIoU 0.086; radar-dropout and mined-finetune rows in `docs/REPORT.md` (rain/night subsets are empty on mini_val → “—”) |
| TensorRT engines + latency table | built + measured on a local NVIDIA L4 (`results/latency_l4.md`; `scripts/colab_trt.sh` remains for T4 — relabel the output if used) |
| GOOSE transfer | done (goose_2d/3d_val.zip, 8 scenes 6/2 split): zero-shot drivable IoU 0.021 → fine-tuned 0.348; LiDAR-only — the 2D/3D zips ship no calibration, `goose_calib.py` writes identity and the camera branch self-disables |
| Real-time latency / jitter under interference | done - C++ periodic benchmark (`cpp/bench`) at 30 Hz, 21 configs x 3 repeats x 2000 iterations on the L4 VM; report `results/REALTIME.md` |
| rviz2 GIF | done — `docs/rviz_replay.gif` recorded in the `ros:humble` container (Xvfb + ffmpeg; obf_node at 11.5 ms/frame on the L4 FP16 engine) |

## Real-time latency and jitter under interference

`cpp/bench/latency_bench` runs the FP16 TensorRT engine in a fixed 30 Hz loop (absolute-deadline `clock_nanosleep`,
deadline = 33.3 ms) end to end: host staging copy of the 24.2 MB input frame, H2D, `enqueueV3`, D2H of all 49.8 MB of
outputs. It logs per-iteration latency, wake-up jitter and deadline misses plus CPU time, context switches, RSS and
NVML GPU utilisation/memory, under CPU, memory-bandwidth and GPU co-tenant interference, with and without mitigations.
Each row: 3 repeats x 2000 iterations (warm-up excluded), percentiles pooled over the 6000 samples; e2e latency in ms.

| scenario | p50 | p99 | p99.9 | max | jitter sd (us) | misses / 6000 |
|---|---|---|---|---|---|---|
| idle | 19.54 | 19.92 | 20.13 | 20.71 | 9.1 | 0 |
| CPU hogs (32 threads), no mitigation | 19.42 | 46.65 | 52.86 | 78.25 | 2525.3 | 2229 |
| CPU hogs, taskset pinning only | 19.44 | 99.97 | 159.52 | 200.04 | 6943.3 | 1345 |
| CPU hogs, `chrt -f 80` (SCHED_FIFO) | 19.25 | 19.47 | 19.68 | 19.79 | 2.8 | 0 |
| CPU hogs, pinning + cgroup cpuset isolation | 19.65 | 19.79 | 19.84 | 20.03 | 5.0 | 0 |
| memory-bandwidth hogs, no mitigation | 28.34 | 44.64 | 56.25 | 61.47 | 818.9 | 2124 |
| memory-bandwidth hogs, pin + FIFO | 26.78 | 28.73 | 30.44 | 31.93 | 36.4 | 0 |
| GPU co-tenant (2nd TensorRT process), no mitigation | 33.08 | 43.01 | 60.00 | 70.47 | 1555.8 | 2795 |
| GPU co-tenant limited to 25 % SMs via MPS | 23.36 | 27.25 | 27.43 | 30.23 | 126.3 | 0 |
| in-process GPU hog, same stream priority | 82.27 | 84.52 | 85.22 | 88.18 | 86.3 | 6000 |
| in-process GPU hog, inference on high-priority stream | 29.57 | 30.55 | 30.83 | 31.14 | 5.8 | 0 |
| combined (CPU + membw + GPU co-tenant), no mitigation | 41.34 | 84.56 | 104.75 | 130.28 | 4190.0 | 5176 |
| combined, pin + FIFO + mlockall + MPS 25 % | 27.97 | 32.69 | 34.36 | 36.04 | 35.2 | 27 |

What helped: SCHED_FIFO (CPU contention tail gone), cgroup cpuset isolation, MPS SM limits or rate-limiting the GPU
co-tenant, and CUDA stream priority against a co-tenant **in the same process**. What did not: taskset pinning alone
made the tail worse (+114 % p99, background daemons were pushed onto the reserved core), stream priority against a
co-tenant in **another process** (no effect, -0.2 % p99), and nothing on the CPU side removes the memory-bandwidth
slowdown (p50 26.78 ms vs 19.54 ms idle). Under combined load the best config still misses 27 / 6000 deadlines.
Full tables (stage split, resources, co-tenant cost, all 21 configs, verdict rule): [results/REALTIME.md](results/REALTIME.md).
**Measured on an AWS EC2 NVIDIA L4 VM (AMD EPYC 7R13, kernel 6.1 PREEMPT_DYNAMIC): not an industrial edge device, not
PREEMPT_RT, hypervisor noise present.**

## Quick start
```bash
pip install -e .[dev]                          # CPU dev / tests
pytest -q && python -m obf.export.onnx_export --cfg configs/tiny.yaml --out /tmp/tiny.onnx
python -m obf.train --cfg configs/tiny.yaml --opts dataset=fake train.epochs=1   # CPU smoke of the whole loop, no data needed
bash scripts/download_data.sh                  # nuScenes-mini + map expansion, Occ3D-nuScenes, GOOSE
dvc repro infos splits                         # infos (radar 5-sweep accumulation, map raster, Occ3D links)
python -m obf.train --cfg configs/base.yaml    # Kaggle: bash scripts/kaggle_train.sh (fits 30 GPU-h/week)
python -m obf.eval  --cfg configs/base.yaml --ckpt checkpoints/cam_lidar_radar/best.pt --name cam_lidar_radar [--subset rain] [--radar_drop 1.0]
python -m obf.export.onnx_export --cfg configs/base.yaml --ckpt ... --out results/export/bevfusion.onnx
bash scripts/colab_trt.sh                      # FP16 / INT8 engines + latency + INT8 accuracy check (Colab T4)
python -m obf.vla.safety_eval --cfg configs/base.yaml --ckpt ... --perception model   # 50-frame safety eval
python -m obf.report                           # -> docs/REPORT.md
```
ROS 2: `cd ros2_ws && colcon build --cmake-args -DTENSORRT_ROOT=... && ros2 launch obf_ros replay.launch.py`.
C++ runner: `cmake -S cpp -B build -DTENSORRT_ROOT=... && cmake --build build && ./build/obf_runner <engine> data/samples/replay/0000_* 200`.

## Design notes
* One 0.4 m grid ([-40, 40] m², z ∈ [-1, 5.4]) for all heads — identical to the Occ3D-nuScenes voxel grid, so occupancy needs no resampling;
  detection range is therefore 40 m instead of the usual 51.2 m (stated when comparing to leaderboard numbers).
* All calibration math (LSS frustum → BEV index) and voxelisation run in the data pipeline; the network graph is pure tensor ops with
  static shapes → ONNX opset 17 → TensorRT with no custom plugins (grid_sample is native; scatter-add lowers to TRT's bundled
  ScatterReduction plugin, registered by the standard `init_libnvinfer_plugins` call — nothing to compile or ship).
* Radar: 5 accumulated sweeps per radar (25 total), ego-motion compensated, features (x, y, z, v_x, v_y comp., RCS, Δt); train-time
  radar dropout p = 0.2 for robustness; test-time dropout 50 % / 100 % reported.
* Multitask: Kendall homoscedastic uncertainty weighting (learned log σ² per task); fixed-weight ablation included.
* VLA safety: deterministic rule (pedestrian in path < 5 m → `stop`) is both the prompt's hard rule and the evaluation oracle;
  unparsable VLM output → `stop` (fail-safe). ISO 21448 mapping: `docs/VLA_SAFETY.md`.
* Licences: code MIT; nuScenes CC BY-NC-SA 4.0 (non-commercial — portfolio use only); GOOSE CC BY-SA 4.0; Occ3D MIT.
