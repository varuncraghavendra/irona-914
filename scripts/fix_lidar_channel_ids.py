"""One-off asset repair: make the RTX lidar emitter channel ids 1-based.

The packaged USD was generated with channelId 0..31. The lidar core rejects any
profile containing channel 0 ("Malformed model parameter update"), falls back to
a dummy profile and publishes no points. Run with Isaac Sim's python.sh.
"""
import sys
from pathlib import Path

# USD only resolves once Kit is up, so start a headless app before importing pxr.
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})
from pxr import Usd  # noqa: E402

ATTRIBUTE = "omni:sensor:Core:emitterState:s001:channelId"


def repair(path):
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        return f"{path}: cannot open"
    fixed = []
    for prim in stage.Traverse():
        attribute = prim.GetAttribute(ATTRIBUTE)
        if not attribute:
            continue
        channels = list(attribute.Get() or [])
        if channels and min(channels) == 0:
            attribute.Set([value + 1 for value in channels])
            fixed.append(str(prim.GetPath()))
    if fixed:
        stage.GetRootLayer().Save()
    return f"{path}: {'updated ' + ', '.join(fixed) if fixed else 'already 1-based'}"


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    targets = sys.argv[1:] or [root / "assets/irona.usd", root / "assets/irona.usda",
                               root / "ros2/irona_sim/project/assets/irona.usd",
                               root / "ros2/irona_sim/project/assets/irona.usda"]
    try:
        for target in targets:
            if Path(target).is_file():
                print(repair(target), flush=True)
    finally:
        _app.close()
