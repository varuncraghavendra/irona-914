# Sensors, transforms and launch behavior

## D455-style RGB-D head

The face was removed completely. A neutral silver shell carries a single dark camera bar. The housing uses the nominal D455 124 × 26 × 29 mm envelope. The depth cameras have a 95 mm nominal stereo baseline. RGB is offset 32.5 mm from the depth origin along the robot's lateral axis. These are design placements, not a calibration file for a physical camera.

The optical origins lie just outside the front panel at robot-rest X = 83 mm, Z = 826 mm. This avoids placing the renderer inside an opaque printed shell. The head is a moving rigid body; camera poses therefore follow both neck joints, walking and flight.

| Camera prim under `/Irona/Links/head/Sensors` | Resolution | Nominal horizontal / vertical FOV | Clip range |
|---|---|---|---|
| `color` | 640 × 400 | 90° / 65° | 0.05–30 m |
| `depth` | 640 × 360 | 86° / 57° | 0.4–10 m |
| `infrared_right` | 640 × 360 | 86° / 57° | 0.4–10 m |

The third camera is an optical/stereo geometry reference, not a published IR stream. Depth comes from Isaac's rendered optical-axis distance, not triangulation between the cameras. RGB and depth are separate unaligned streams with distinct optical origins; no aligned-depth topic is claimed. CameraInfo is computed from the render product. No noise, distortion, IMU stream, infrared projector pattern or RealSense firmware is simulated.

The hardware reference is Intel's [D455 specification](https://www.intel.com/content/www/us/en/products/sku/205847/intel-realsense-depth-camera-d455/specifications.html) and the [D400-series datasheet](https://www.realsenseai.com/wp-content/uploads/2020/06/Intel-RealSense-D400-Series-Datasheet-June-2020.pdf). The project uses lower rendering resolutions for practical simulation. The range setting is a clipping choice, not a hardware accuracy guarantee.

## 3D lidar

The `lidar` prim is native `OmniLidar` with `OmniSensorGenericLidarCoreAPI` and one emitter-state API. All emitter arrays and scan settings are authored in the robot USD; no external lidar asset download is needed. The configuration is generic rotary scanning: 32 elevations from −15° to +15°, 360° azimuth, 10 revolutions/s and 18,000 azimuth firings/s. This is 1,800 horizontal samples × 32 channels per revolution before invalid returns are dropped. Range limits are 0.1–30 m. Range resolution and accuracy parameters are generic simulation settings.

Its rest optical center is X = −6 mm, Y = 0, Z = 898.4 mm. The head roof and lower ring sit below the scan band; the upper cap sets the overall robot height to 914.4 mm. Opaque geometry is omitted from the scan band so the sensor does not see its own cover. A static printed mock-up needs a transparent spacer to support the top cap. The printer cannot manufacture a working lidar optical window.

Full-scan point clouds are configured with sensor-frame coordinates and motion compensation. Self-occlusion can still occur when the robot moves its arms above its head or the neck tilts. Vertical field of view is 30°, not 360°. There is no claimed weather, multi-path, material or motion-distortion calibration.

Schema and integration follow NVIDIA's [RTX lidar documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_rtx_lidar.html), [OmniSensor schema](https://docs.omniverse.nvidia.com/kit/docs/omni.usd.schema.omni_sensors/107.3.1/omni_sensors_schema.html) and [ROS RTX lidar tutorial](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_rtx_lidar.html).

## TF ownership and axes

The robot uses +X forward, +Y left and +Z up. ROS optical frames use +Z forward, +X right and +Y down. Their fixed URDF rotation is roll −π/2, pitch 0, yaw −π/2. USD cameras look down −Z with +Y up; a separate camera rotation maps those axes into the same physical direction. Offline validation compares the resulting basis vectors.

Isaac publishes exactly one world-to-pelvis transform from the actual simulated floating base. `robot_state_publisher` publishes all articulated link transforms from `/joint_states`, plus fixed head-to-camera, optical and lidar transforms. No second node publishes the root transform. Every optical frame and `lidar_link` remains connected to `world`.

The graph uses simulation timestamps. RViz and robot_state_publisher use `use_sim_time=true`. RGB, depth and calibration share render products and frame-skip settings. The camera pipeline follows NVIDIA's [camera publishing tutorial](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_camera_publishing.html).

## Troubleshooting and acceptance

1. Start one `sensors.launch.py` process. Let RTX shader compilation finish and confirm that simulation time advances.
2. Use `ros2 topic list` and `ros2 topic hz /irona/lidar/points`. If discovery fails, compare `ROS_DOMAIN_ID`, DDS/RMW settings and NVIDIA's ROS environment requirements.
3. If images appear but point clouds do not, check that RTX rendering is enabled. Headless mode in this project still renders. An empty scene or surfaces outside range can produce no returns; keep the default arena for the acceptance test.
4. If RViz reports missing transforms, check `/joint_states`, `/clock`, `/tf` and `/tf_static`, then use `world` as the fixed frame. Keep the supplied Best Effort sensor subscriptions.
5. Run `ros2 run irona_sim sensor_smoke_test.py --timeout 30`. This is a live test supplied for your workstation, not a pre-recorded passing result.
6. If your installed minor version rejects a graph node or attribute, use `--export-stage` after resolving the version issue and inspect the registered graph. The shipped integration is checked against 5.1 APIs, not executed across every 5.x release.
