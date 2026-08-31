#!/usr/bin/env bash
# Record the rviz2 replay (ROS 2 Humble) to docs/rviz_replay.gif
set -e
cd ros2_ws && colcon build --packages-select obf_ros --cmake-args -DTENSORRT_ROOT=${TENSORRT_ROOT:-/usr} && source install/setup.bash && cd ..
ros2 launch obf_ros replay.launch.py engine:=results/export/bevfusion_fp16.engine replay_dir:=data/samples/replay &
sleep 8
byzanz-record --duration=12 --x=0 --y=0 --width=1280 --height=800 docs/rviz_replay.gif   # apt install byzanz (or use peek)
kill %1
