# Running this project in Isaac Sim 5.1 — what had to change

The repository shipped source/API-checked against NVIDIA's 5.1 documentation but
had never been executed. Bringing it up on Isaac Sim 5.1.0-rc.19 with ROS 2
Humble on an RTX 4080 required the fixes below. Each one is described with the
symptom first, because the symptoms were mostly silent.

## 1. The runtime died right after `world.reset()` (silently)

`RigidPrim(...)` defaults to `prepare_contact_sensors=True`, which re-authors the
collision prims of the links it wraps. Doing that to links of an initialised
articulation deletes and recreates the shapes:

```
prim '/World/Irona/Links/pelvis/collision' was deleted while being used by a
shape in a tensor view class. The physics.tensors simulationView was invalidated.
```

Every later pose or velocity read then failed. `source/run_isaac.py` now uses one
regex view over all 63 links with `prepare_contact_sensors=False`, which is also
one batched tensor read per step instead of 63.

The failure was invisible because `SimulationApp.close()` terminates the process
before Python prints a traceback. `run_isaac.py` now catches, prints and exits
non-zero itself.

## 2. numpy 2 versus `omni.graph` — no ROS bridge, no camera, no lidar

This build ships numpy 2.2.6, but its `omni.graph` binding reads the numpy 1.x
array descriptor layout. Any Python write to an OmniGraph array attribute fails:

```
TypeError: Unable to write from unknown dtype, kind=i, size=0
```

That single bug took out the ROS 2 bridge graph (`inputs:translation` on the TF
node), every replicator annotator attach, and therefore the camera, the lidar and
the in-process perception. `scripts/setup_isaac_numpy_shim.sh` installs numpy
1.26.4 into `vendor/`; `scripts/run_sim.sh` puts it first on `PYTHONPATH`. The
Isaac Sim installation is not modified.

## 3. Node type renamed

`isaacsim.core.nodes.IsaacRunOneSimulationFrame` is registered in 5.1 as
`isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame`. With the old id the camera
and lidar graphs failed to build while the state graph still worked, so clock,
joint states and TF published but no images did.

## 4. The RTX lidar published nothing

Two separate causes:

- The emitter profile used `channelId` 0..31. The lidar core numbers channels
  from 1, rejected the profile (`Malformed model parameter update`) and fell back
  to a dummy one. `source/sensors.py` now emits 1..32, and
  `scripts/fix_lidar_channel_ids.py` repairs already-built USD assets.
- RTX sensors only tick when the *application* drives the update. The original
  loop stepped physics by hand (`world.step(render=False)` plus `world.render()`),
  and `SimulationContext.render()` explicitly disables sim playback around its
  app update, so the lidar never rotated. The loop now runs control from a
  physics callback and calls `world.step(render=True)`; a reference NVIDIA lidar
  config produced zero points under the old loop too, confirming the cause.

## 5. The renderer never saw the robot move

Mirroring PhysX transforms into USD every 1 ms costs about 6 ms/step — five times
the whole simulation — so it is off. But with it off, something still has to
publish poses to Fabric, and the `omni.physx.fabric` extension was not enabled.
The camera image, the RTX sensors and every ROS image were therefore frozen at
frame zero while physics ran normally. The symptom in perception was a detection
whose pixel count and range never changed while its world position swept around
as the head moved. `run_isaac.py` now enables `omni.physx.fabric` whenever the
USD write-back is off.

## 6. Simulation time must come from the physics step

With the application driving the substeps, their size is not guaranteed to equal
the requested `physics_dt`. Deriving time from a step counter made the gait run
at roughly half speed against real physics time and the robot fell. Time is now
accumulated from the `step_size` the physics callback is given.

## 7. Gait fixes found while walking

- The pelvis lateral target was `support_y * 0.91`, which only holds while the
  feet straddle y = 0. After a few steering steps it places the pelvis outboard of
  the support foot and the robot topples. It is now an offset from the stance
  foot towards the stance centre, which is identical at y = 0.
- `Walker` gained `steer()`, `march()`, `request_stop()` and `resume()` so a
  mission can aim the gait, step in place to line up, stop with the feet level and
  start again.

## 8. Referencing the residential assets into a metre stage

Three traps, all of which look like "the scene is empty" or "the props are specks":

- Isaac Sim's metrics assembler appends `xformOp:scale:unitsResolve` (0.01 for the
  centimetre-authored ArchVis props) **on a later update**, not at reference time.
  Applying a unit conversion as well makes everything 100x too small; measuring a
  prop before the assembler runs reports a bounding box 100x too large. `house.py`
  leaves the conversion to the assembler and pumps `app.update()` before it
  measures anything.
- The props keep their geometry on payloads, so a prim can exist with a correct
  extents hint - and therefore a correct-looking bounding box - while the renderer
  has nothing to draw. `stage.Load()` after referencing fixes that.
- A `RectLight` already emits along its own -Z. Adding a 180 degree flip aims the
  ceiling panels at the ceiling, and the camera sees a black room.

## 9. Holding the arm out saturates the shipped shoulder torque

Reaching to a table at grasp height needs more than the 4 Nm design cap on the
shoulder drives: the drive saturates, the shoulder sits about 14 degrees below its
command, the palm drops 60-90 mm and the fingers catch the table edge, which adds
more load. `--arm-effort-scale` (default 2) raises the arm caps at load time; with
it the arm tracks its command to about 1 degree. An XM540-W270 stalls near 10 Nm,
so 8 Nm is still a conservative cap.

## Measured behaviour on this workstation

| Quantity | Value |
|---|---|
| Physics | 1 ms steps, CPU TGS, 32/8 solver iterations |
| Real-time factor | ~0.2-0.4 headless, depending on active sensors |
| Standing | stable indefinitely; pelvis holds 0.4546 m, tilt < 0.2 deg |
| Walking | 1.67 m in 70 s continuous (0.024 m/s), tilt < 2.4 deg, no fall |
| Camera | 640x400 colour and 640x360 depth over ROS 2 |
| Lidar | ~57,000 points per full scan, frame `lidar_link` |
| Vase estimate | within ~20 mm of the true axis at 0.7 m (arena); ~15 mm at 1.1 m in the house |
| Arm tracking | 14 deg of shoulder sag at the shipped torque cap, ~1 deg at `--arm-effort-scale 2` |
| Grasp | palm within ~5 mm of the aim point after correction; vase lifted 0.090 m |
| Friction-only grasp | fails: the vase slides ~60 mm along the pedestal and rises 5 mm |

## 10. A loaded GR00T server can starve the RTX camera

With the GR00T N1.5 policy server resident (5.7 GB of 12 GB) a fetch run produced
no camera frames at all: `scanning: no frame` for the whole run, no CUDA error in
the log, and a dashboard with zero frames. The same scene runs normally with the
server stopped. Treat a silent camera as a VRAM symptom first.

## Known limits

- **Walking endurance.** The longest run measured here is 70 s of continuous
  walking, 1.67 m at a steady 0.024 m/s with tilt under 2.4 degrees and no fall.
  The gait is still open loop in the plan frame — it commands foot placements
  without measuring where the feet landed — so it has no recovery behaviour and
  nothing guarantees a longer run. The falls seen before fix 6 above were that
  timing bug, not the gait.
- **Walking speed.** 0.024 m/s. Approaching a vase a metre away takes about
  40 s of simulated time, roughly two minutes of wall clock.
- **Grasp assist.** The default lift replaces finger contact with a fixed joint
  (see the README). Measured with `--no-grasp-assist`, the same approach, closure
  and lift push the vase about 60 mm along the pedestal and raise it 5 mm: the
  fingers cannot hold it. Finger drives cap at 0.08 Nm, and the hand closes onto
  a 64 mm cylinder mostly with its fingertips.
- **Perception scope.** Hue segmentation on ideal rendered colour and depth, with
  the vase's own dimensions as a prior. It is not a clutter- or noise-robust
  detector, and it only looks for this object.
