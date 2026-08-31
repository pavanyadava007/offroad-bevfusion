#!/usr/bin/env bash
# Full training ladder inside the 30 GPU-h/week Kaggle budget (P100 / 2xT4). ~1 h/epoch-set on mini for the fused model.
set -e
pip install -q -e . wandb
python scripts/create_infos.py --root data/nuscenes --version v1.0-mini --out data/infos --occ_root data/occ3d/gts
python -m obf.data.splits data/infos/nuscenes_v1.0-mini_infos.pkl data/splits
for c in ablation_cam ablation_cam_lidar ablation_cam_lidar_radar; do
  python -m obf.train --cfg configs/$c.yaml --opts train.epochs=12 data.batch_size=2
done
python -m obf.train --cfg configs/ablation_no_deform.yaml --opts train.epochs=12
python -m obf.train --cfg configs/ablation_fixed_weights.yaml --opts train.epochs=12
CK=checkpoints/cam_lidar_radar/best.pt; C=configs/base.yaml
python -m obf.eval --cfg $C --ckpt $CK --name cam_lidar_radar
python -m obf.eval --cfg $C --ckpt $CK --name cam_lidar_radar_rdrop50 --radar_drop 0.5
python -m obf.eval --cfg $C --ckpt $CK --name cam_lidar_radar_rdrop100 --radar_drop 1.0
for s in rain night clear_day; do python -m obf.eval --cfg $C --ckpt $CK --name cam_lidar_radar_$s --subset $s; done
python -m obf.eval --cfg configs/ablation_cam.yaml --ckpt checkpoints/cam/best.pt --name cam
python -m obf.eval --cfg configs/ablation_cam_lidar.yaml --ckpt checkpoints/cam_lidar/best.pt --name cam_lidar
python -m obf.eval --cfg $C --ckpt $CK --name cam_lidar_radar_train --split train
python -m obf.mining.hard_examples --cfg $C --det results/eval/cam_lidar_radar_train/detections.json --k 3
python -m obf.train --cfg $C --opts data.hard_scene_weights=data/splits/hard_scenes.json train.ckpt_dir=checkpoints/cam_lidar_radar_mined train.epochs=4 --resume $CK
python -m obf.export.dump_samples --cfg $C --n 100 --out data/samples/calib
python -m obf.export.dump_samples --cfg $C --n 81 --out data/samples/replay --raw
python -m obf.export.onnx_export --cfg $C --ckpt $CK --out results/export/bevfusion.onnx
python -m obf.report
