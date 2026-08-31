"""GOOSE calibration -> per-scene npz (cam2lidar 4x4, intrinsic 3x3).
The public goose_2d/3d_*.zip releases ship NO calibration files (calibration only exists in the rosbag `setups/`
downloads). If none are found, identity matrices are written: with K = I the LSS frustum unprojects far outside
pc_range, the valid mask discards every camera point, and the camera branch contributes zeros on GOOSE — the model
then works LiDAR-driven. This is a documented limitation, not a substitute for real calibration."""
import argparse
import glob
import json
import os

import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="data/goose"); a = ap.parse_args()
    out = os.path.join(a.root, "calibration"); os.makedirs(out, exist_ok=True)
    files = glob.glob(os.path.join(a.root, "calib*", "*.yaml")) + glob.glob(os.path.join(a.root, "calib*", "*.json")) \
        + glob.glob(os.path.join(a.root, "setups", "**", "*calib*.y*ml"), recursive=True)
    K, cam2lidar = np.eye(3), np.eye(4)
    if files:
        c = yaml.safe_load(open(files[0])) if files[0].endswith((".yaml", ".yml")) else json.load(open(files[0]))
        K = np.array(c.get("camera_matrix", c.get("K", np.eye(3).tolist()))).reshape(3, 3)
        cam2lidar = np.array(c.get("T_lidar_cam", c.get("cam2lidar", np.eye(4).tolist()))).reshape(4, 4)
        print(f"calibration parsed from {files[0]}")
    else:
        print("WARNING: no calibration files in this GOOSE release -> writing identity (camera branch will contribute "
              "zeros via the LSS valid mask; training/eval are LiDAR-driven)")
    scenes = {os.path.basename(s.rstrip("/")) for s in glob.glob(os.path.join(a.root, "lidar", "*", "*", ""))}
    for scene in sorted(scenes):
        np.savez(os.path.join(out, scene + ".npz"), cam2lidar=cam2lidar, intrinsic=K)
    print("wrote", len(scenes), "calibration files (identity=" + str(not files) + ")")


if __name__ == "__main__":
    main()
