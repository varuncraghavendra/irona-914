"""Native USD RGB-D and OmniLidar sensor definitions, in metres.

D455 envelope + ideal pinhole RGB/depth simulation. Depth is rendered distance
along the optical axis, not a RealSense stereo/firmware or librealsense emulator.
"""
import json,math
import numpy as np
from pxr import UsdGeom,Sdf,Gf
from model import ROOT

# ROS optical coordinates: X right, Y down, Z forward; robot: X forward,Y left,Z up.
OPTICAL_R=np.array([[0,0,1],[-1,0,0],[0,-1,0]],dtype=float)
CAMERA_POSITION=[.083,0,.062]  # head-relative envelope centre on front face
LIDAR_POSITION=[-.006,0,.1344]
CAMERAS={
    "depth":dict(position=[.083,.0475,.062],resolution=[640,360],fov_deg=[86,57],clip=[.4,10.0],frame="camera_depth_optical_frame"),
    "color":dict(position=[.083,.015,.062],resolution=[640,400],fov_deg=[90,65],clip=[.05,30.0],frame="camera_color_optical_frame"),
    "infrared_right":dict(position=[.083,-.0475,.062],resolution=[640,360],fov_deg=[86,57],clip=[.4,10.0],frame="camera_right_ir_optical_frame"),
}
LIDAR=dict(channels=32,horizontal_fov_deg=360,vertical_fov_deg=30,rotation_hz=10,
           azimuth_samples_per_turn=1800,range_m=[.10,30.0],frame="lidar_link")
TOPICS={
    "rgb":"/irona/camera/color/image_raw", "rgb_info":"/irona/camera/color/camera_info",
    "depth":"/irona/camera/depth/image_rect_raw", "depth_info":"/irona/camera/depth/camera_info",
    "depth_points":"/irona/camera/depth/points", "lidar_points":"/irona/lidar/points",
    "joint_states":"/joint_states", "clock":"/clock", "tf":"/tf",
    "arm_command":"/irona/arm_hand_command",
}

def sensor_frames():
    frames=[dict(name="camera_link",parent="head",xyz=CAMERA_POSITION,rpy=[0,0,0]),
            dict(name="lidar_link",parent="head",xyz=LIDAR_POSITION,rpy=[0,0,0])]
    for cfg in CAMERAS.values():
        frames.append(dict(name=cfg['frame'],parent="camera_link",
                           xyz=(np.array(cfg['position'])-CAMERA_POSITION).tolist(),rpy=[-math.pi/2,0,-math.pi/2]))
    return frames

def author_sensors(stage,robot_path="/Irona"):
    root=robot_path+"/Links/head/Sensors"
    UsdGeom.Scope.Define(stage,root)
    for name,cfg in CAMERAS.items():
        cam=UsdGeom.Camera.Define(stage,root+"/"+name)
        cam.AddTranslateOp().Set(Gf.Vec3d(*cfg['position']))
        # USD camera: +X right, +Y up, -Z forward. This rotation points it along robot +X.
        usd_R=OPTICAL_R@np.diag([1,-1,-1])
        quat=Gf.Matrix3d(*usd_R.T.flatten().tolist()).ExtractRotation().GetQuat()
        cam.AddOrientOp().Set(Gf.Quatf(quat))
        focal=10.0
        cam.CreateFocalLengthAttr(focal)
        cam.CreateHorizontalApertureAttr(2*focal*math.tan(math.radians(cfg['fov_deg'][0]/2)))
        cam.CreateVerticalApertureAttr(2*focal*math.tan(math.radians(cfg['fov_deg'][1]/2)))
        cam.CreateClippingRangeAttr(Gf.Vec2f(*cfg['clip']))
        cam.GetPrim().CreateAttribute("irona:resolution",Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(*cfg['resolution']))
        cam.GetPrim().CreateAttribute("irona:rosFrame",Sdf.ValueTypeNames.String).Set(cfg['frame'])
    # Self-contained native 5.x OmniLidar: no remote sensor assets or legacy JSON path.
    p=stage.DefinePrim(root+"/lidar","OmniLidar")
    p.AddAppliedSchema("OmniSensorGenericLidarCoreAPI")
    p.CreateAttribute("omni:sensor:tickRate",Sdf.ValueTypeNames.Float,custom=False).Set(10.0)
    p.AddAppliedSchema("OmniSensorGenericLidarCoreEmitterStateAPI:s001")
    # Author xform properties directly so usd-core can author this extension schema offline.
    p.CreateAttribute("xformOp:translate",Sdf.ValueTypeNames.Double3,custom=False).Set(Gf.Vec3d(*LIDAR_POSITION))
    p.CreateAttribute("xformOpOrder",Sdf.ValueTypeNames.TokenArray,custom=False).Set(["xformOp:translate"])
    p.CreateAttribute("visibility",Sdf.ValueTypeNames.Token,custom=False).Set("invisible")
    for name,value in dict(scanType="ROTARY",rotationDirection="CCW",rayType="IDEALIZED",
                           auxOutputType="BASIC",elementsCoordsType="CARTESIAN",outputFrameOfReference="SENSOR",
                           outputMotionCompensationState="COMPENSATED").items():
        p.CreateAttribute("omni:sensor:Core:"+name,Sdf.ValueTypeNames.Token,custom=False).Set(value)
    for name,value in dict(nearRangeM=.10,farRangeM=30.0,validStartAzimuthDeg=0.0,validEndAzimuthDeg=360.0,
                          startAzimuthOffsetDeg=0.0,rangeResolutionM=.004,rangeAccuracyM=.02,
                          azimuthErrorStd=.005,elevationErrorStd=.005,minReflectionRangeM=30.0).items():
        p.CreateAttribute("omni:sensor:Core:"+name,Sdf.ValueTypeNames.Float,custom=False).Set(value)
    for name,value in dict(scanRateBaseHz=10,reportRateBaseHz=18000,numberOfEmitters=32,numberOfChannels=32,
                          maxReturns=1,stateResolutionStep=1).items():
        p.CreateAttribute("omni:sensor:Core:"+name,Sdf.ValueTypeNames.UInt,custom=False).Set(value)
    for name,values,typ in [
        ("azimuthDeg",[0.0]*32,Sdf.ValueTypeNames.FloatArray),
        ("elevationDeg",np.linspace(-15,15,32).tolist(),Sdf.ValueTypeNames.FloatArray),
        ("fireTimeNs",[0]*32,Sdf.ValueTypeNames.UIntArray),
        # The RTX lidar core numbers channels from 1; a zero makes it reject the whole
        # emitter profile, fall back to a dummy one and publish no points.
        ("channelId",list(range(1,33)),Sdf.ValueTypeNames.UIntArray)]:
        p.CreateAttribute("omni:sensor:Core:emitterState:s001:"+name,typ,custom=False).Set(values)
    p.CreateAttribute("irona:rosFrame",Sdf.ValueTypeNames.String).Set(LIDAR['frame'])
    (ROOT/"config/sensors.json").write_text(json.dumps(dict(cameras=CAMERAS,lidar=LIDAR,topics=TOPICS,frames=sensor_frames(),
        model_scope="Ideal rendered depth, approximate nominal D455 intrinsics/extrinsics; no measured calibration. Generic 32-channel lidar."),indent=2))
