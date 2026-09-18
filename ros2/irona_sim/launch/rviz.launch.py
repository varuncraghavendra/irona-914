"""Viewer only: TF tree and RViz, in a terminal of its own.

Start the simulator separately with scripts/run_sim.sh --ros2. This launch file
deliberately does not start Isaac Sim, so the viewer can be restarted without
disturbing a running simulation.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def start(context):
    share = Path(get_package_share_directory('irona_sim'))
    domain = LaunchConfiguration('domain_id').perform(context)
    description = (share / 'urdf/irona.urdf').read_text()
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               name='robot_state_publisher', output='screen',
               parameters=[{'robot_description': description, 'use_sim_time': True,
                            'publish_frequency': 60.0}],
               additional_env={'ROS_DOMAIN_ID': domain})
    rviz = Node(package='rviz2', executable='rviz2', name='irona_rviz', output='screen',
                arguments=['-d', str(share / 'rviz/irona.rviz')],
                parameters=[{'use_sim_time': True}],
                condition=IfCondition(LaunchConfiguration('rviz')),
                additional_env={'ROS_DOMAIN_ID': domain})
    return [rsp, rviz]


def generate_launch_description():
    import os
    arguments = [('rviz', 'true'), ('domain_id', os.getenv('ROS_DOMAIN_ID', '0'))]
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k, v in arguments]
                             + [OpaqueFunction(function=start)])
