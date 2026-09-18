"""Launch Isaac Sim, the complete TF tree, and RViz for both sensors."""
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,OpaqueFunction,ExecuteProcess,RegisterEventHandler,EmitEvent
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def start(context):
    cfg=lambda n:LaunchConfiguration(n).perform(context)
    share=Path(get_package_share_directory('irona_sim'))
    project=Path(cfg('project_path')).expanduser().resolve()
    python=Path(cfg('isaac_python')).expanduser()
    if not python.is_file():raise RuntimeError('Set isaac_python:=/absolute/path/to/isaac-sim/python.sh')
    script=project/'source/run_isaac.py'
    if not script.is_file():raise RuntimeError(f'Missing project runtime: {script}')
    cmd=[str(python),str(script),'--ros2','--mode',cfg('mode'),'--seconds',cfg('seconds')]
    if cfg('headless').lower()=='true':cmd+=['--headless']
    if cfg('camera').lower()!='true':cmd+=['--no-camera']
    if cfg('lidar').lower()!='true':cmd+=['--no-lidar']
    if cfg('arena').lower()!='true':cmd+=['--no-arena']
    simulator=ExecuteProcess(cmd=cmd,output='screen',additional_env={'ROS_DOMAIN_ID':cfg('domain_id')})
    # The simulator exclusively publishes world->pelvis. RSP exclusively publishes
    # all robot joints and fixed camera/lidar optical frames from the same model.
    description=(share/'urdf/irona.urdf').read_text()
    rsp=Node(package='robot_state_publisher',executable='robot_state_publisher',name='robot_state_publisher',
             parameters=[{'robot_description':description,'use_sim_time':True,'publish_frequency':60.0}],
             additional_env={'ROS_DOMAIN_ID':cfg('domain_id')},output='screen')
    rviz=Node(package='rviz2',executable='rviz2',name='irona_rviz',arguments=['-d',str(share/'rviz/irona.rviz')],
              parameters=[{'use_sim_time':True}],condition=IfCondition(LaunchConfiguration('rviz')),
              additional_env={'ROS_DOMAIN_ID':cfg('domain_id')},output='screen')
    return [rsp,simulator,rviz,RegisterEventHandler(OnProcessExit(target_action=simulator,
             on_exit=[EmitEvent(event=Shutdown(reason='Isaac Sim exited'))]))]


def generate_launch_description():
    import os
    share=Path(get_package_share_directory('irona_sim'))
    args=[('isaac_python',str(Path.home()/'isaacsim/python.sh')),('project_path',str(share/'project')),
          ('mode','stand'),('seconds','0'),('headless','false'),('camera','true'),('lidar','true'),
          ('arena','true'),('rviz','true'),('domain_id',os.getenv('ROS_DOMAIN_ID','0'))]
    return LaunchDescription([DeclareLaunchArgument(k,default_value=v) for k,v in args]+[OpaqueFunction(function=start)])
