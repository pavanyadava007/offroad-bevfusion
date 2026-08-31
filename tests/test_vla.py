import numpy as np

from obf.vla.grounding import parse_action
from obf.vla.perception_json import hazard_rule, objects_from_boxes, perception_json


def test_parse_and_failsafe():
    assert parse_action('{"action": "dump", "target": "truck", "reason": "ok"}')["action"] == "dump"
    assert parse_action("garbage")["action"] == "stop" and not parse_action("garbage")["parsed"]
    assert parse_action('{"action": "fly"}')["action"] == "stop"


def test_hazard_rule():
    boxes = np.array([[3.0, 0.5, 0, 0.6, 0.6, 1.7, 0, 0, 0], [30.0, 0, 0, 2, 4, 1.5, 0, 0, 0]])
    objs = objects_from_boxes(boxes, [8, 0], [0.9, 0.9], ["car"] * 8 + ["pedestrian", "cone"])
    p = perception_json(objs)
    assert p["nearest_pedestrian_in_path_m"] == 3.0 and hazard_rule(p) == "stop"
    p2 = perception_json(objects_from_boxes(boxes[1:], [0], [0.9], ["car"]))
    assert hazard_rule(p2) is None


def test_occ_nearest_occupied():
    pr = (-6.4, -6.4, -1, 6.4, 6.4, 5.4)
    occ = np.full((32, 32, 4), 17, np.int64)  # all 'free'
    assert perception_json([], occ=occ, pc_range=pr)["nearest_occupied_ahead_m"] is None
    occ[16, 24, 0] = 3  # one occupied voxel 8 cells (3.2 m) ahead in the corridor
    assert perception_json([], occ=occ, pc_range=pr)["nearest_occupied_ahead_m"] == 3.2
