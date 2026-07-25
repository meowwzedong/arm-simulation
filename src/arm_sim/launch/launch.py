import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    pkg_name = 'arm_sim'
    pkg_share = get_package_share_directory(pkg_name)

    # Tell Gazebo where to find package:// meshes
    model_path = os.path.join(pkg_share, '..')
    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] += os.pathsep + model_path
    else:
        os.environ['GAZEBO_MODEL_PATH'] = model_path

    # Process the xacro file into a URDF string
    xacro_file = os.path.join(pkg_share, 'urdf', 'arm.urdf.xacro')
    robot_description = xacro.process_file(xacro_file).toxml()

    # 1) Start Gazebo Classic
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch', 'gazebo.launch.py',
            )
        ),
        launch_arguments={'verbose': 'true'}.items(),
    )

    # 2) Publish the robot description + TF frames
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
    )

    # 3) Spawn the robot into Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'three_link_arm'],
        output='screen',
    )

    # 4) Controller spawners (started in sequence AFTER the robot spawns)
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )

    load_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller'],
        output='screen',
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        # After the robot is spawned -> start joint_state_broadcaster
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        # After joint_state_broadcaster -> start arm_controller
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_arm_controller],
            )
        ),
    ])