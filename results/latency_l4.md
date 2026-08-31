GPU: NVIDIA L4, batch 1, 200 iters (30 warm-up). Inputs: imgs, cam_bev_idx, cam_valid, lidar_feats, lidar_num, lidar_coors, radar_feats, radar_num, radar_coors.

| Runtime | mean ms | p50 ms | p99 ms |
|---|---|---|---|
| PyTorch FP32 | 36.4 | 36.4 | 37.2 |
| PyTorch AMP FP16 | 25.0 | 25.0 | 25.9 |
| TensorRT FP16 | 15.2 | 15.1 | 15.8 |
| TensorRT INT8 | 14.7 | 14.7 | 15.0 |

_Jetson Orin: **not measured** (no hardware available); engines built and timed on the GPU named above (local L4) only._
