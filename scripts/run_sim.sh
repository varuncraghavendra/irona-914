#!/usr/bin/env bash
# Start the Isaac Sim side of the demo.
#
# Isaac Sim must NOT inherit a sourced ROS 2 environment: its Python is 3.11 and
# Humble's is 3.10, and its ROS 2 bridge ships its own Humble libraries. This
# script therefore clears the ROS variables and points the loader at the bridge's
# bundled libraries instead. It also puts the numpy 1.26 shim first on PYTHONPATH
# (see setup_isaac_numpy_shim.sh for why).
#
#   scripts/run_sim.sh --mode fetch --ros2            # the full demo
#   scripts/run_sim.sh --mode walk --ros2 --headless  # walking and sensors only
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC="${ISAAC_PATH:-$HOME/isaacsim}"
[ -x "$ISAAC/python.sh" ] || { echo "Isaac Sim not found at $ISAAC; set ISAAC_PATH" >&2; exit 1; }
[ -d "$ROOT/vendor/numpy126/numpy" ] || "$ROOT/scripts/setup_isaac_numpy_shim.sh"

unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
export PYTHONPATH="$ROOT/vendor/numpy126"
export LD_LIBRARY_PATH="$ISAAC/exts/isaacsim.ros2.bridge/humble/lib"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
exec "$ISAAC/python.sh" "$ROOT/source/run_isaac.py" "$@"
