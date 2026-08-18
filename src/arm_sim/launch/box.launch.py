"""
Spawn the graspable cube into a already-running Gazebo session.

Usage:
    ros2 launch arm_sim spawn_box.launch.py
    ros2 launch arm_sim spawn_box.launch.py x:=0.12 y:=0.03
    ros2 launch arm_sim spawn_box.launch.py name:=box2 x:=-0.08 y:=0.05

Defaults put the cube 10 cm out along +X, resting on the ground (z = half the
15 mm cube height), which is inside the gripper's teeth-down workspace.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sdf_path = os.path.join(
        get_package_share_directory('arm_sim'),
        'models',
        'box.sdf',
    )

    return LaunchDescription([
        DeclareLaunchArgument('name', default_value='box',
                              description='Entity name in Gazebo'),
        DeclareLaunchArgument('x', default_value='0.10'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.0075',
                              description='Half the cube height, so it rests on the ground'),

        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_box',
            output='screen',
            arguments=[
                '-entity', LaunchConfiguration('name'),
                '-file', sdf_path,
                '-x', LaunchConfiguration('x'),
                '-y', LaunchConfiguration('y'),
                '-z', LaunchConfiguration('z'),
            ],
        ),
    ])