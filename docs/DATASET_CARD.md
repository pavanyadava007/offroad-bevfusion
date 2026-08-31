# Dataset card

## nuScenes-mini (v1.0-mini) + Map expansion v1.3
* 10 scenes (8 train / 2 val), 404 keyframes, 6 cameras, 1 LiDAR (32-beam), 5 radars, HD maps. Licence CC BY-NC-SA 4.0 (non-commercial).
* Reference frame: ego frame at LiDAR keyframe timestamp. Cameras: `cam2ego_ref = inv(T_ego_lidar→global) · T_ego_cam→global · T_cam→ego_cam`.
* Radar: 5 sweeps per sensor, ego-motion compensated; velocities `vx_comp/vy_comp` rotated into ego; Δt in s. Stored as `data/infos/radar/<token>.npy` [M,7].
* Drivable area: `drivable_area` polygons (exteriors − interiors) rasterised into the 0.4 m ego grid (`scripts/create_infos.py:drivable_mask`).
* Adverse-condition subsets (scene descriptions): `data/splits/val_{rain,night,clear_day}.json`. mini_val has 2 scenes only → subset sizes are small; report n.
* Detection GT filter: annotations with ≥1 LiDAR or radar point, 10 detection classes (official mapping in `obf/data/common.py`).

## Occ3D-nuScenes
* Voxel grid [-40, 40]×[-40, 40]×[-1, 5.4] m @ 0.4 m → 200×200×16, 18 classes (0–16 semantic, 17 free); `mask_camera` used in loss and metric.
* Only labels for the 10 mini scenes are needed (`gts/<scene_name>/<sample_token>/labels.npz`); missing labels → ignore (255).

## GOOSE (German Outdoor and Offroad Dataset, Fraunhofer IOSB) — CC BY-SA 4.0
* Single windshield camera + Velodyne VLS-128 with per-point semantic labels (64 classes). We use a ≤10 GB scene subset.
* 64 → 8 merged BEV classes by *name* (`obf/data/goose.py:MERGE`); sky/undefined ignored. BEV target = majority label per 0.4 m cell.
* Calibration: per-release format differs → `scripts/goose_calib.py` writes `calibration/<scene>.npz {cam2lidar, intrinsic}`; verify key names.
* Fallback: RELLIS-3D (same loader with path patterns changed); optional RADIATE (radar + LiDAR + cam in fog/rain/snow).

## Data pipeline (DVC)
`infos → splits → train_* → eval_* → mine → dump_samples → export_onnx → build_trt → vla_safety → goose → report`.
Remote: Google Drive (15 GB) for infos, checkpoints, ONNX, engines, dumped samples. Raw datasets are not pushed (licence + size).

## Hard-example mining
Per-scene mAP/NDS from the official devkit on training scenes → the k hardest scenes get ×3 sampling weight
(`WeightedRandomSampler`) for a short continuation run. Auto-labelling (`obf/mining/autolabel.py`) exports high-confidence
pseudo-labels for unlabelled off-road frames (semi-supervised fine-tuning).
