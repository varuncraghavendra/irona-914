#!/usr/bin/env bash
# Start the viewer side of the demo in its own terminal: robot_state_publisher
# for the TF tree plus RViz with the camera, depth cloud and lidar displays.
# Uses the system ROS 2 Humble installation, never Isaac Sim's Python.
set -eo pipefail      # not -u: the ROS setup scripts read unset variables
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [ -f "$ROOT/ros2/install/setup.bash" ]; then
  source "$ROOT/ros2/install/setup.bash"
else
  echo "Workspace not built. Run: cd $ROOT/ros2 && colcon build --symlink-install" >&2
  exit 1
fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
exec ros2 launch irona_sim rviz.launch.py "$@"
