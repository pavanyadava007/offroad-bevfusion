"""Build nuScenes-mini infos: per keyframe sample -> camera calib (cam->ref-ego), LiDAR sweeps (sweep->ref-ego),
radar points (5 sweeps x 5 radars, ego-motion compensated, features x y z vx_comp vy_comp rcs dt) saved as .npy,
GT boxes in ref-ego frame, rasterised drivable_area mask (map API polygons), Occ3D label path, scene description.
python scripts/create_infos.py --root data/nuscenes --version v1.0-mini --out data/infos --occ_root data/occ3d/gts"""
import argparse
import os
import pickle

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from obf.data.common import NUSC_CAT_MAP, rasterize_polys

CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
RADARS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]
PC_RANGE, BEV = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], (200, 200)


def T(rot, trans):
    m = np.eye(4); m[:3, :3] = Quaternion(rot).rotation_matrix; m[:3, 3] = trans
    return m


def sensor_to_ref(nusc, sd, ref_inv):
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"]); ep = nusc.get("ego_pose", sd["ego_pose_token"])
    return ref_inv @ T(ep["rotation"], ep["translation"]) @ T(cs["rotation"], cs["translation"])


def sweeps(nusc, sd_token, n, ref_inv, ref_ts):
    out, tok = [], sd_token
    while tok and len(out) < n:
        sd = nusc.get("sample_data", tok)
        out.append((sd, sensor_to_ref(nusc, sd, ref_inv), (ref_ts - sd["timestamp"]) * 1e-6))
        tok = sd["prev"]
    return out


def radar_points(nusc, sample, n, ref_inv, ref_ts):
    from nuscenes.utils.data_classes import RadarPointCloud
    pts = []
    for ch in RADARS:
        for sd, M, dt in sweeps(nusc, sample["data"][ch], n, ref_inv, ref_ts):
            pc = RadarPointCloud.from_file(os.path.join(nusc.dataroot, sd["filename"])).points  # [18,M]
            if pc.shape[1] == 0:
                continue
            xyz = M[:3, :3] @ pc[:3] + M[:3, 3:4]
            v = M[:2, :2] @ pc[8:10]  # vx_comp, vy_comp rotated into ref frame
            pts.append(np.concatenate([xyz, v, pc[5:6], np.full((1, pc.shape[1]), dt)], 0).T)
    return (np.concatenate(pts, 0) if pts else np.zeros((0, 7))).astype(np.float32)


def drivable_mask(nmap, ego2global):
    cx, cy = ego2global[:2, 3]; r = 60.0
    inv = np.linalg.inv(ego2global)
    polys, holes = [], []
    for tok in nmap.get_records_in_patch((cx - r, cy - r, cx + r, cy + r), ["drivable_area"], mode="intersect")["drivable_area"]:
        for ptok in nmap.get("drivable_area", tok)["polygon_tokens"]:
            poly = nmap.extract_polygon(ptok)
            if poly.is_empty:
                continue
            to_ego = lambda coords: (inv[:2, :2] @ np.array(coords)[:, :2].T + inv[:2, 3:4]).T
            polys.append(to_ego(poly.exterior.coords))
            holes += [to_ego(i.coords) for i in poly.interiors]
    return rasterize_polys(polys, PC_RANGE, BEV, holes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes"); ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default="data/infos"); ap.add_argument("--occ_root", default="data/occ3d/gts")
    ap.add_argument("--lidar_sweeps", type=int, default=1); ap.add_argument("--radar_sweeps", type=int, default=5)
    ap.add_argument("--no_map", action="store_true")
    a = ap.parse_args()
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    nusc = NuScenes(a.version, a.root, verbose=False)
    splits = create_splits_scenes()
    tr, va = (splits["mini_train"], splits["mini_val"]) if "mini" in a.version else (splits["train"], splits["val"])
    maps = {}
    if not a.no_map:
        from nuscenes.map_expansion.map_api import NuScenesMap
    os.makedirs(os.path.join(a.out, "radar"), exist_ok=True); os.makedirs(os.path.join(a.out, "drivable"), exist_ok=True)
    infos = {"train": [], "val": []}
    for sample in tqdm(nusc.sample):
        scene = nusc.get("scene", sample["scene_token"])
        split = "train" if scene["name"] in tr else ("val" if scene["name"] in va else None)
        if split is None:
            continue
        lid = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ep = nusc.get("ego_pose", lid["ego_pose_token"])
        ego2global = T(ep["rotation"], ep["translation"]); ref_inv = np.linalg.inv(ego2global)
        info = {"token": sample["token"], "scene_token": scene["token"], "scene_name": scene["name"],
                "scene_description": scene["description"], "timestamp": sample["timestamp"], "ego2global": ego2global,
                "location": nusc.get("log", scene["log_token"])["location"], "cams": {}, "lidar_sweeps": [], "gt_boxes": [], "gt_names": []}
        for cam in CAMS:
            sd = nusc.get("sample_data", sample["data"][cam]); cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            info["cams"][cam] = {"path": os.path.join(a.root, sd["filename"]), "cam2ego": sensor_to_ref(nusc, sd, ref_inv),
                                 "intrinsic": np.array(cs["camera_intrinsic"]), "timestamp": sd["timestamp"]}
        for sd, M, dt in sweeps(nusc, sample["data"]["LIDAR_TOP"], a.lidar_sweeps, ref_inv, lid["timestamp"]):
            info["lidar_sweeps"].append({"path": os.path.join(a.root, sd["filename"]), "sweep2ref": M, "dt": dt})
        rp = os.path.join(a.out, "radar", sample["token"] + ".npy")
        np.save(rp, radar_points(nusc, sample, a.radar_sweeps, ref_inv, lid["timestamp"])); info["radar_path"] = rp
        for box in nusc.get_boxes(lid["token"]):
            ann = nusc.get("sample_annotation", box.token)
            if ann["num_lidar_pts"] + ann["num_radar_pts"] == 0 or ann["category_name"] not in NUSC_CAT_MAP:
                continue
            vel = nusc.box_velocity(box.token); vel = np.zeros(3) if np.isnan(vel).any() else vel
            box.translate(-np.array(ep["translation"])); box.rotate(Quaternion(ep["rotation"]).inverse)
            v = Quaternion(ep["rotation"]).inverse.rotation_matrix @ vel
            info["gt_boxes"].append([*box.center, *box.wlh, box.orientation.yaw_pitch_roll[0], v[0], v[1]])
            info["gt_names"].append(NUSC_CAT_MAP[ann["category_name"]])
        info["gt_boxes"] = np.array(info["gt_boxes"], np.float32).reshape(-1, 9)
        if not a.no_map:
            loc = info["location"]
            maps.setdefault(loc, NuScenesMap(a.root, loc))
            dp = os.path.join(a.out, "drivable", sample["token"] + ".npy")
            np.save(dp, drivable_mask(maps[loc], ego2global)); info["drivable_path"] = dp
        occ = os.path.join(a.occ_root, scene["name"], sample["token"], "labels.npz")
        info["occ_path"] = occ if os.path.exists(occ) else None
        infos[split].append(info)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"nuscenes_{a.version}_infos.pkl")
    with open(out, "wb") as f:
        pickle.dump(infos, f)
    n_occ = sum(i["occ_path"] is not None for s in infos.values() for i in s)
    print(f"{out}: train={len(infos['train'])} val={len(infos['val'])} occ3d labels found={n_occ}")


if __name__ == "__main__":
    main()
