#!/usr/bin/env python3
"""Publishes the raw sensors of the same replay frames (PointCloud2 in ego frame + CAM_FRONT image) for rviz2 context.
Frame order == manifest.json order so it stays aligned with obf_node."""
import json
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
import cv2


class Replay(Node):
    def __init__(self):
        super().__init__("obf_replay_sensors")
        self.declare_parameter("replay_dir", "data/samples/replay"); self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("frame_id", "base_link")
        d = self.get_parameter("replay_dir").value
        self.frames = json.load(open(f"{d}/manifest.json"))["frames"]
        self.frame_id = self.get_parameter("frame_id").value
        self.pc_pub = self.create_publisher(PointCloud2, "/obf/lidar", 10)
        self.im_pub = self.create_publisher(Image, "/obf/cam_front", 10)
        self.i = 0
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self.step)

    def step(self):
        fr = self.frames[self.i % len(self.frames)]; self.i += 1
        h = Header(frame_id=self.frame_id, stamp=self.get_clock().now().to_msg())
        # lidar_feats.npy is already in ego frame: [1,P,N,5] -> valid points
        f = np.load(f"{fr['dir']}/lidar_feats.npy")[0]; n = np.load(f"{fr['dir']}/lidar_num.npy")[0]
        pts = np.concatenate([f[p, : n[p], :4] for p in range(len(n)) if n[p] > 0], 0).astype(np.float32)
        msg = PointCloud2(header=h, height=1, width=len(pts), is_dense=True, is_bigendian=False, point_step=16, row_step=16 * len(pts),
                          fields=[PointField(name=nm, offset=4 * k, datatype=PointField.FLOAT32, count=1) for k, nm in enumerate("xyzi")],
                          data=pts.tobytes())
        self.pc_pub.publish(msg)
        if fr.get("cam_front"):
            im = cv2.imread(fr["cam_front"])
            if im is not None:
                im = cv2.resize(im, (704, 396))
                self.im_pub.publish(Image(header=h, height=im.shape[0], width=im.shape[1], encoding="bgr8", step=im.shape[1] * 3, data=im.tobytes()))


def main():
    rclpy.init(); rclpy.spin(Replay()); rclpy.shutdown()


if __name__ == "__main__":
    main()
