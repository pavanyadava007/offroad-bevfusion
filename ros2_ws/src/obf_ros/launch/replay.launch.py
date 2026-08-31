from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    share = get_package_share_directory("obf_ros")
    engine = LaunchConfiguration("engine"); replay = LaunchConfiguration("replay_dir")
    return LaunchDescription([
        DeclareLaunchArgument("engine", default_value="results/export/bevfusion_fp16.engine"),
        DeclareLaunchArgument("replay_dir", default_value="data/samples/replay"),
        Node(package="obf_ros", executable="obf_node", parameters=[{"engine": engine, "replay_dir": replay, "rate_hz": 10.0}]),
        Node(package="obf_ros", executable="replay_sensors.py", parameters=[{"replay_dir": replay, "rate_hz": 10.0}]),
        Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(share, "rviz", "obf.rviz")]),
    ])
