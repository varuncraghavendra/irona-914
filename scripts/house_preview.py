"""Render stills of the house scene: one third-person view and one head-camera view.

Useful for checking lighting and prop placement without opening the GUI:
    scripts/run_sim.sh is for running the demo; this only renders frames.
"""
import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720})

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))


def main(out_dir, eye, target):
    import numpy as np
    from pxr import Gf, UsdGeom
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.storage.native import get_assets_root_path
    import cv2

    from control import HOME, STAND_PELVIS
    from model import DT, NAMES, ROOT as PROJECT
    for ext in ["isaacsim.sensors.rtx", "isaacsim.core.nodes", "omni.replicator.core"]:
        enable_extension(ext)
    app.update()
    world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=1 / 60, backend="numpy", device="cpu")
    stage = world.stage
    world.scene.add_default_ground_plane()
    add_reference_to_stage(str(PROJECT / "assets/irona.usd"), "/World/Irona")
    from house import build_house
    report = build_house(stage, get_assets_root_path())
    ground = stage.GetPrimAtPath("/World/defaultGroundPlane")
    if ground:
        UsdGeom.Imageable(ground).MakeInvisible()
    robot = world.scene.add(SingleArticulation(prim_path="/World/Irona/Links/pelvis", name="irona"))

    view = UsdGeom.Camera.Define(stage, "/World/PreviewCamera")
    view.CreateFocalLengthAttr(18.0)
    view.CreateClippingRangeAttr(Gf.Vec2f(.05, 60.0))
    eye = np.asarray(eye, dtype=float); target = np.asarray(target, dtype=float)
    forward = target - eye; forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    basis = np.stack([right, up, -forward], axis=1)
    quat = Gf.Matrix3d(*basis.T.flatten().tolist()).ExtractRotation().GetQuat()
    view.AddTranslateOp().Set(Gf.Vec3d(*eye))
    view.AddOrientOp().Set(Gf.Quatf(quat))

    world.reset()
    indices = np.array([list(robot.dof_names).index(n) for n in NAMES], dtype=np.int32)
    robot.set_world_pose(position=STAND_PELVIS, orientation=np.array([1., 0., 0., 0.]))
    robot.set_joint_positions(HOME, joint_indices=indices)
    robot.apply_action(ArticulationAction(joint_positions=HOME, joint_indices=indices))

    from isaacsim.sensors.camera import Camera
    cameras = {"scene": Camera(prim_path="/World/PreviewCamera", resolution=(1280, 720), name="preview"),
               "head": Camera(prim_path="/World/Irona/Links/head/Sensors/depth", resolution=(640, 360), name="head")}
    for camera in cameras.values():
        camera.initialize()
    for _ in range(90):
        world.step(render=True)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, camera in cameras.items():
        rgba = camera.get_current_frame().get("rgb")
        if rgba is None:
            print(f"{name}: no frame", flush=True); continue
        image = np.asarray(rgba)[..., :3][..., ::-1]
        path = out_dir / f"house_{name}.png"
        cv2.imwrite(str(path), image)
        written.append(f"{name}: {path} mean {float(np.asarray(rgba)[...,:3].mean()):.1f}")
    print("scene:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report.items()
                     if k not in ("centre",)}, flush=True)
    print("\n".join(written), flush=True)


if __name__ == "__main__":
    try:
        out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "previews")
        main(out, eye=(-0.85, -1.45, 1.45), target=(1.35, 0.05, 0.45))
    finally:
        app.close()
