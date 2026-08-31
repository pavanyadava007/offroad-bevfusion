import numpy as np


def voxelize(points, pillar, pc_range, max_pillars, max_points, rng=None):
    """Pillarize points [M,F] (x,y,z,...) -> feats [P,N,F], num [P] int32, coors [P,2]=(y,x) int32.
    Static output shapes (zero padded; padded pillars have coors=-1) so the ONNX graph stays static."""
    rng = rng if rng is not None else np.random
    x0, y0, z0, x1, y1, z1 = pc_range
    W = int(round((x1 - x0) / pillar))
    H = int(round((y1 - y0) / pillar))
    F = points.shape[1]
    feats = np.zeros((max_pillars, max_points, F), np.float32)
    num = np.zeros(max_pillars, np.int32)
    coors = np.full((max_pillars, 2), -1, np.int32)
    m = ((points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)
         & (points[:, 2] >= z0) & (points[:, 2] < z1))
    pts = points[m].astype(np.float32)
    if len(pts) == 0:
        return feats, num, coors
    # clamp: a float32 coordinate just below the range edge (e.g. 39.999996 < 40) can still divide to exactly W
    ix = np.minimum(((pts[:, 0] - x0) / pillar).astype(np.int64), W - 1)
    iy = np.minimum(((pts[:, 1] - y0) / pillar).astype(np.int64), H - 1)
    key = iy * W + ix
    order = np.argsort(key, kind="stable")
    pts, key = pts[order], key[order]
    uniq, start, counts = np.unique(key, return_index=True, return_counts=True)
    P = len(uniq)
    new_id = np.arange(P) if P <= max_pillars else rng.permutation(P)
    inv = np.repeat(np.arange(P), counts)
    within = np.arange(len(pts)) - np.repeat(start, counts)
    pid = new_id[inv]
    ok = (pid < max_pillars) & (within < max_points)
    feats[pid[ok], within[ok]] = pts[ok]
    keep = new_id < max_pillars
    num[new_id[keep]] = np.minimum(counts, max_points)[keep]
    coors[new_id[keep], 0] = (uniq // W)[keep]
    coors[new_id[keep], 1] = (uniq % W)[keep]
    return feats, num, coors
