from pathlib import Path
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    main=Path(get_package_share_directory('irona_sim'))/'launch/irona_sim.launch.py'
    return LaunchDescription([IncludeLaunchDescription(PythonLaunchDescriptionSource(str(main)),
        launch_arguments={'camera':'false','lidar':'true'}.items())])
