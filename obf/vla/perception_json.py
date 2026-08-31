"""BEV perception outputs (or GT) -> compact JSON the VLM grounds its decision on.
Ego frame: x forward, y left. 'in_path' = inside the loader's forward corridor (0 < x < path_len, |y| < half_width)."""
import math

import numpy as np

PATH_LEN, HALF_WIDTH, HAZARD_DIST = 20.0, 2.5, 5.0


def objects_from_boxes(boxes, labels, scores, class_names, max_objs=12):
    objs = []
    for b, l, s in zip(np.asarray(boxes), np.asarray(labels), np.asarray(scores)):
        x, y = float(b[0]), float(b[1])
        rng = math.hypot(x, y)
        objs.append({"class": class_names[int(l)], "range_m": round(rng, 1), "bearing_deg": round(math.degrees(math.atan2(y, x)), 0),
                     "in_path": bool(0 < x < PATH_LEN and abs(y) < HALF_WIDTH), "vel_mps": round(float(math.hypot(b[7], b[8])), 1),
                     "score": round(float(s), 2)})
    return sorted(objs, key=lambda o: o["range_m"])[:max_objs]


def perception_json(objs, seg=None, occ=None, pc_range=(-40, -40, -1, 40, 40, 5.4), free_cls=17):
    """objs: list from objects_from_boxes; seg: [3,Y,X] probabilities (optional); occ: [Y,X,Z] class ids (optional);
    free_cls: id of the 'free' class (Occ3D-nuScenes: 17 = n_classes - 1)."""
    p = {"objects": objs}
    peds = [o for o in objs if o["class"] == "pedestrian" and o["in_path"]]
    p["nearest_pedestrian_in_path_m"] = min([o["range_m"] for o in peds], default=None)
    if seg is not None:
        Y, X = seg.shape[1:]
        x0, y0, _, x1, y1, _ = pc_range
        ys = slice(int((0 - y0) / (y1 - y0) * Y) - 6, int((0 - y0) / (y1 - y0) * Y) + 6)  # |y| < 2.4 m
        xs = slice(int((0 - x0) / (x1 - x0) * X), int((PATH_LEN - x0) / (x1 - x0) * X))
        p["drivable_ahead_ratio"] = round(float((seg[0, ys, xs] > 0.5).mean()), 2)
    if occ is not None:
        Y, X, Z = occ.shape
        col = occ[Y // 2 - 6: Y // 2 + 6, X // 2:, :]
        free = (col == free_cls).all(-1)  # column contains only 'free' voxels
        blocked = np.where(~free.all(0))[0]
        p["nearest_occupied_ahead_m"] = round(float(blocked[0] * (pc_range[3] - pc_range[0]) / X), 1) if len(blocked) else None
    return p


def hazard_rule(p):
    """Deterministic teacher / safety oracle: pedestrian in path closer than HAZARD_DIST -> stop."""
    d = p.get("nearest_pedestrian_in_path_m")
    if d is not None and d < HAZARD_DIST:
        return "stop"
    if d is not None and d < 2 * HAZARD_DIST:
        return "wait_for_person"
    if p.get("drivable_ahead_ratio") is not None and p["drivable_ahead_ratio"] < 0.2:
        return "reverse"
    return None
