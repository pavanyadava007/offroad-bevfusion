import cv2
import numpy as np

DET_CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle",
               "pedestrian", "traffic_cone"]
NUSC_CAT_MAP = {
    "vehicle.car": "car", "vehicle.truck": "truck", "vehicle.construction": "construction_vehicle",
    "vehicle.bus.bendy": "bus", "vehicle.bus.rigid": "bus", "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier", "vehicle.motorcycle": "motorcycle", "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian", "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian", "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
}
DEFAULT_ATTR = {"car": "vehicle.parked", "pedestrian": "pedestrian.moving", "trailer": "vehicle.parked",
                "truck": "vehicle.parked", "bus": "vehicle.moving", "motorcycle": "cycle.without_rider",
                "construction_vehicle": "vehicle.parked", "bicycle": "cycle.without_rider", "barrier": "",
                "traffic_cone": ""}
VEHICLE_SET = {"car", "truck", "bus", "trailer", "construction_vehicle"}
IMG_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], np.float32)


def box_corners_bev(boxes):
    """boxes [K,>=7] (x,y,z,w,l,h,yaw) -> corners [K,4,2] (ego frame). w along box-y, l along heading."""
    x, y, w, l, yaw = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.stack([l / 2, l / 2, -l / 2, -l / 2], 1)
    dy = np.stack([w / 2, -w / 2, -w / 2, w / 2], 1)
    cx = x[:, None] + dx * c[:, None] - dy * s[:, None]
    cy = y[:, None] + dx * s[:, None] + dy * c[:, None]
    return np.stack([cx, cy], -1)


def rasterize_polys(polys, pc_range, bev_size, holes=(), dilate=0):
    """Polygons in ego xy (list of [n,2]) -> uint8 mask [Y,X]; col=x, row=y."""
    Y, X = bev_size
    x0, y0, _, x1, y1, _ = pc_range
    dx, dy = (x1 - x0) / X, (y1 - y0) / Y
    m = np.zeros((Y, X), np.uint8)
    for group, val in ((polys, 1), (holes, 0)):
        for p in group:
            pix = np.stack([(p[:, 0] - x0) / dx, (p[:, 1] - y0) / dy], 1)
            cv2.fillPoly(m, [np.round(pix).astype(np.int32)], val)
    if dilate:
        m = cv2.dilate(m, np.ones((2 * dilate + 1,) * 2, np.uint8))
    return m
