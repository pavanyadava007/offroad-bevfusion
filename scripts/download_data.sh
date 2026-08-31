#!/usr/bin/env bash
# Data acquisition (all free / non-commercial). nuScenes requires a (free) account; download manually, then run.
set -e
mkdir -p data/nuscenes data/occ3d data/goose
echo "1) nuScenes v1.0-mini (~4 GB) + Map expansion v1.3: https://www.nuscenes.org/download  -> extract to data/nuscenes/ (maps/ + v1.0-mini/ + samples/ + sweeps/)"
echo "2) Occ3D-nuScenes labels (gts.tar.gz): https://github.com/Tsinghua-MARS-Lab/Occ3D -> extract to data/occ3d/gts/ (only mini scenes are used)"
echo "3) GOOSE (Fraunhofer IOSB): https://goose-dataset.de -> data/goose/{images,lidar,goose_label_mapping.csv} (scene subset <= 10 GB); fallback RELLIS-3D"
echo "4) optional RADIATE (fog/rain/snow, radar+LiDAR+cam): https://pro.hw.ac.uk/radiate/"
echo "Kaggle: attach the datasets as Kaggle Datasets and symlink: ln -s /kaggle/input/nuscenes-mini data/nuscenes"
