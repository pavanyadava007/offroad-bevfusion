#!/usr/bin/env bash
# Off-road transfer: zero-shot nuScenes model (drivable vs rest) on GOOSE, then fine-tune all heads on GOOSE; report drop.
set -e
python scripts/goose_calib.py --root data/goose            # writes calibration/<scene>.npz (cam2lidar, intrinsic) from GOOSE calib files
python scripts/goose_zero_shot.py --ckpt checkpoints/cam_lidar/best.pt --split val   # true zero-shot: nuScenes head + drivable-channel mapping
python -m obf.train --cfg configs/goose_transfer.yaml
python -m obf.eval --cfg configs/goose_transfer.yaml --ckpt checkpoints/goose_transfer/best.pt --name goose_transfer --split val
python - <<'PY'
import json; z=json.load(open('results/goose_zero_shot.json')); f=json.load(open('results/goose_transfer.json'))
d={'drivable_zero_shot':z['seg_IoU'].get('drivable'),'drivable_finetuned':f['seg_IoU'].get('drivable')}
d['drivable_drop']=(d['drivable_finetuned'] or 0)-(d['drivable_zero_shot'] or 0); json.dump(d,open('results/goose_drop.json','w'),indent=2); print(d)
PY
