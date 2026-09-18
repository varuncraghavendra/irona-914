# Validation record — hands and perception revision

The revised model has 63 rigid bodies and 62 actuated joints. All original
standing, walking and flight reports were regenerated after the arm/hand/mass
changes. A hand-motion rollout was added. These runs use MuJoCo 3.13.0, not
Isaac Sim / PhysX.

## Current independent dynamics runs

| Mode | Simulation time (s) | Fell? | Forward displacement (m) | Maximum tilt (deg) | Peak joint torque (Nm) |
|---|---:|---|---:|---:|---:|
| stand | 6.0 | No | -0.0041 | 0.159 | 5.357 |
| walk | 24.0 | No | 0.3867 | 3.241 | 19.864 |
| hands | 8.0 | No | -0.0058 | 0.159 | 5.357 |
| flight | 16.0 | No | -0.0015 | 0.431 | 12.215 |

The revised 24-second walk traveled 0.3867 m forward with −0.0806 m lateral
sway/drift. The 16-second flight rose about 0.651 m and returned close to its
starting height. Peak summed jet force was 154.43 N. Walking uses joint drives
and ground contacts; flight uses external wrenches, not a prescribed root path.

Each mode's JSON report contains measured peak torques/speeds for all joints.
The corresponding trajectory includes root pose, joint positions, foot normal
forces, COM and thruster-force magnitudes. Physics dt is 0.001 s. Geometry-only
changes to joint covers/support rails do not change this proxy dynamics model.

## Asset and revision checks

Final geometry: 914.4000 mm tall. All 458 STL files passed mesh/bed-size checks:
442 exterior/detail parts and 16 static support/assembly aid files.

`asset_report.json` records the final measured height, print count, USD composition,
positive inertias, coincident joint anchors, ungrounded articulation, print-mesh
closure/size, leg IK and thrust-allocation rank. `revision_report.json` separately
checks all ten fingers, removal of facial features, camera forward/right/down
axes, 360-degree lidar emitter arrays, URDF agreement, graph connections, syntax
and RViz configuration. Read those generated reports for the authoritative
final file counts and pass/fail results.

The revised print parts are checked for watertight meshes, consistent winding,
positive volume and a maximum axis-aligned dimension of 200 mm. No physical
print or production tolerance test was performed.

## Runtime test supplied, but not run here

`ros2/irona_sim/scripts/sensor_smoke_test.py` is intended for the user's running
Isaac/ROS workstation. It checks actual images, CameraInfo, point clouds, clock,
all 62 joint positions and TF connectivity; with the default arena it also
checks lidar returns in all four quadrants and vertical 3D spread. No passing
ROS runtime report is fabricated or included.

The native graph layer and standalone source are checked against NVIDIA's
Isaac Sim 5.1 interfaces. Offline graph validation cannot establish that an
installed Isaac version will load every extension or publish data correctly.

## Scope limits

- Isaac/PhysX execution, RTX rendering, ROS transport and RViz behavior remain untested here.
- MuJoCo and PhysX use different contact solvers. The MuJoCo check does not reproduce the PhysX joint-friction coefficients.
- Full moving CAD clearance, hardware motor fit, real bearing supports, FEA, print tolerances, cable transmissions and power/thermal design are unverified.
- The hand demo demonstrates opening/closing in physics; no object grasp or manipulation success is claimed.
- The camera is ideal rendered RGB-D; the lidar is a generic configured model. Neither is a calibrated physical sensor emulator.
- The included video is an accelerated replay of recorded MuJoCo states using the CAD geometry. Thrust glyphs are illustrative, not exhaust-fluid simulation.
