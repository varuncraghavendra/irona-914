#!/usr/bin/env bash
# Isaac Sim 5.1.0-rc.19 ships numpy 2.2.6, but its omni.graph binding reads the
# numpy 1.x descriptor layout. Every Python write to an OmniGraph array attribute
# then fails with "Unable to write from unknown dtype, kind=..., size=0", which
# breaks the ROS 2 bridge graph setup and every replicator annotator attach.
#
# This installs numpy 1.26.4 into vendor/ so the launch scripts can put it ahead
# of Isaac's copy on PYTHONPATH, without touching the Isaac Sim installation.
set -euo pipefail
ISAAC=${ISAAC_PATH:-$HOME/isaacsim}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/vendor/numpy126"
if [ -d "$TARGET/numpy" ]; then
  echo "numpy shim already present: $TARGET"
  exit 0
fi
"$ISAAC/kit/python/bin/python3" -m pip install --no-deps --target "$TARGET" numpy==1.26.4
echo "numpy shim installed: $TARGET"
