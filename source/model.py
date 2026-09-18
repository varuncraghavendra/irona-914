"""Irona 914 — one SI-unit definition for USD, control and dynamics checks.

Frames: +X forward, +Y left, +Z up. All rest-frame rotations are identity.
Masses are a design budget, not measured hardware properties.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEIGHT = 0.9144
DT = 1 / 1000
LINKS = {}
JOINTS = []

def link(name, position, mass, size, com=(0, 0, 0), collision=True):
    size = np.asarray(size, dtype=float)
    inertia = mass / 12 * np.array([size[1]**2 + size[2]**2,
                                   size[0]**2 + size[2]**2,
                                   size[0]**2 + size[1]**2])
    LINKS[name] = dict(name=name, position=list(position), mass=mass,
                       size=size.tolist(), com=list(com), inertia=inertia.tolist(),
                       collision=collision)
    return name

def joint(name, parent, child, axis, limits, effort, kp, kd,
          kind="revolute", velocity=5.0, actuator="custom high-torque drive", friction=.03):
    JOINTS.append(dict(name=name, parent=parent, child=child, axis=axis,
                       limits=list(limits), effort=effort, kp=kp, kd=kd,
                       kind=kind, velocity=velocity, actuator=actuator, friction=friction,
                       anchor=LINKS[child]["position"]))

link("pelvis", (0, 0, .480), 2.0, (.125, .190, .095), (0, 0, .010))
link("waist_yaw_link", (0, 0, .535), .12, (.04, .04, .025), collision=False)
joint("waist_yaw", "pelvis", "waist_yaw_link", "Z", (-45, 45), 25, 260, 12)
link("torso", (0, 0, .545), 3.6, (.145, .210, .150), (-.012, 0, .076))
joint("waist_pitch", "waist_yaw_link", "torso", "Y", (-20, 25), 40, 450, 16)
link("neck_yaw_link", (0, 0, .744), .10, (.035, .035, .025), collision=False)
joint("neck_yaw", "torso", "neck_yaw_link", "Z", (-80, 80), 4, 35, 2)
link("head", (0, 0, .764), .91, (.132, .145, .130), (.010, 0, .069))
joint("neck_pitch", "neck_yaw_link", "head", "Y", (-30, 35), 5, 45, 2.5)

for side, sign in [("left", 1), ("right", -1)]:
    y = sign * .075
    link(side+"_hip_yaw_link", (0, y, .460), .10, (.035, .035, .035), collision=False)
    joint(side+"_hip_yaw", "pelvis", side+"_hip_yaw_link", "Z", (-40, 40), 30, 320, 10)
    link(side+"_hip_roll_link", (0, y, .460), .15, (.038, .038, .038), collision=False)
    joint(side+"_hip_roll", side+"_hip_yaw_link", side+"_hip_roll_link", "X", (-40, 40), 55, 550, 16)
    link(side+"_thigh", (0, y, .460), .95, (.072, .067, .160), (0, 0, -.095))
    joint(side+"_hip_pitch", side+"_hip_roll_link", side+"_thigh", "Y", (-105, 60), 65, 600, 18)
    link(side+"_shin", (0, y, .265), .70, (.064, .060, .160), (0, 0, -.094))
    joint(side+"_knee", side+"_thigh", side+"_shin", "Y", (0, 145), 65, 650, 18)
    link(side+"_ankle_pitch_link", (0, y, .070), .12, (.035, .035, .035), collision=False)
    joint(side+"_ankle_pitch", side+"_shin", side+"_ankle_pitch_link", "Y", (-60, 50), 40, 500, 12)
    link(side+"_foot", (0, y, .070), .60, (.170, .092, .055), (.0225, 0, -.0425))
    joint(side+"_ankle_roll", side+"_ankle_pitch_link", side+"_foot", "X", (-35, 35), 35, 500, 12)

    # Offset serial axes provide room for separate horns and bearing yokes.
    # DYNAMIXEL torque limits below are conservative DESIGN caps, not continuous ratings.
    sy = sign * .172
    arm_specs = [
        ("shoulder_pitch", "torso", "shoulder_pitch_link", (0,sy,.703), .20, (.042,.040,.050), (0,0,-.012), "Y", (-110,95), 4.0, 95, 3.5, 2.0, "XM540-W270"),
        ("shoulder_roll", "shoulder_pitch_link", "shoulder_roll_link", (0,sy+sign*.021,.680), .20, (.039,.038,.038), (0,0,-.006), "X", (-80,80), 4.0, 90, 3.2, 2.0, "XM540-W270"),
        ("shoulder_yaw", "shoulder_roll_link", "upper_arm", (0,sy+sign*.025,.657), .33, (.040,.044,.081), (0,0,-.049), "Z", (-100,100), 3.5, 65, 2.2, 2.0, "XM540-W270"),
        ("elbow", "upper_arm", "forearm", (0,sy+sign*.025,.553), .22, (.037,.040,.036), (0,0,-.017), "Y", (-135,0), 1.5, 55, 1.8, 3.0, "XM430-W350"),
        ("forearm_roll", "forearm", "forearm_roll_link", (0,sy+sign*.025,.516), .24, (.041,.049,.059), (0,0,-.035), "Z", (-110,110), 1.0, 40, 1.0, 3.0, "XM430-W350"),
        ("wrist_pitch", "forearm_roll_link", "wrist_pitch_link", (0,sy+sign*.025,.437), .10, (.032,.034,.025), (0,0,-.008), "Y", (-65,65), .85, 20, .6, 3.0, "XM430-W350"),
        ("wrist_roll", "wrist_pitch_link", "hand", (0,sy+sign*.025,.417), .16, (.023,.064,.038), (0,0,-.021), "X", (-35,35), .65, 15, .4, 3.0, "XM430-W350"),
    ]
    for name,parent,child,pos,mass,size,com,axis,limits,effort,kp,kd,velocity,motor in arm_specs:
        parent_name=parent if parent=="torso" else side+"_"+parent
        link(side+"_"+child,pos,mass,size,com,collision=child not in ["shoulder_pitch_link","shoulder_roll_link","wrist_pitch_link"])
        joint(side+"_"+name,parent_name,side+"_"+child,axis,limits,effort,kp,kd,velocity=velocity,actuator=motor,friction=.04)
    hy=sy+sign*.025
    # Four 3-link fingers, flexing toward +X (palm normal), with independent contacts.
    for finger,dy,lengths in [("index",-.025,[.027,.019,.014]),("middle",-.008,[.030,.021,.015]),
                              ("ring",.009,[.028,.020,.014]),("little",.026,[.022,.016,.012])]:
        z=.373;parent=side+"_hand"
        for k,(segment,length) in enumerate(zip(["mcp","pip","dip"],lengths)):
            body=f"{side}_{finger}_{segment}_link";name=f"{side}_{finger}_{segment}"
            link(body,(0,hy+sign*dy,z),[.009,.006,.004][k],(.011,.012,length-.003),(0,0,-length/2))
            joint(name,parent,body,"Y",[(-85,10),(-100,0),(-75,0)][k],[.08,.05,.025][k],
                  [1.3,.8,.5][k],[.035,.025,.015][k],velocity=2.5,
                  actuator="XL330-M288 remote tendon equivalent",friction=.02)
            parent=body;z-=length
    # Thumb: opposition plus basal spread, MCP flexion and IP flexion (four DOFs).
    ty=hy-sign*.045
    link(side+"_thumb_cmc_link",(0,ty,.396),.009,(.016,.015,.013),collision=False)
    joint(side+"_thumb_opposition",side+"_hand",side+"_thumb_cmc_link","Z",(-65,65),.10,1.5,.04,velocity=2.0,actuator="XL330-M288 remote tendon equivalent",friction=.02)
    parent=side+"_thumb_cmc_link";z=.393
    for k,(segment,length,axis,limits) in enumerate([("spread",.021,"X",(-40,40)),("mcp",.020,"Y",(-75,10)),("ip",.017,"Y",(-80,0))]):
        body=f"{side}_thumb_{segment}_link"
        link(body,(0,ty,z),[.010,.008,.005][k],(.013,.014,length-.003),(0,0,-length/2))
        joint(f"{side}_thumb_{segment}",parent,body,axis,limits,[.08,.07,.04][k],1.0,.03,velocity=2.0,
              actuator="XL330-M288 remote tendon equivalent",friction=.02)
        parent=body;z-=length

TOTAL_MASS = sum(v["mass"] for v in LINKS.values())
NAMES = [j["name"] for j in JOINTS]
INDEX = {name: i for i, name in enumerate(NAMES)}
THRUSTERS = [
    dict(name="left_boot_jet", body="left_foot", point=[.0225, 0, -.058], max_force=90.0),
    dict(name="right_boot_jet", body="right_foot", point=[.0225, 0, -.058], max_force=90.0),
    dict(name="left_back_jet", body="torso", point=[-.093, .090, .050], max_force=65.0),
    dict(name="right_back_jet", body="torso", point=[-.093, -.090, .050], max_force=65.0),
]
JET_TILT = np.deg2rad(25)
JET_TIME_CONSTANT = .08

def configuration():
    return dict(name="Irona 914 — hands and perception revision", height_m=HEIGHT, units="m, kg, s, N, Nm, rad",
                frame="X forward, Y left, Z up", nominal_mass_kg=TOTAL_MASS,
                mass_status="assumed engineering budget; replace with measured hardware",
                links=LINKS, joints=JOINTS, thrusters=THRUSTERS,
                jet_tilt_deg=25, jet_time_constant_s=JET_TIME_CONSTANT,
                physics_dt=DT, control_dt=DT)

def rotation(axis, angle):
    x = np.array({"X":[1,0,0],"Y":[0,1,0],"Z":[0,0,1]}[axis], dtype=float)
    K = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)

def forward_kinematics(q, root_position=None, root_rotation=None):
    """Dictionary of body origin positions and rotations in world frame."""
    root_position = np.array(LINKS["pelvis"]["position"] if root_position is None else root_position)
    root_rotation = np.eye(3) if root_rotation is None else root_rotation
    frames = {"pelvis": (root_position, root_rotation)}
    for i, j in enumerate(JOINTS):
        p, R = frames[j["parent"]]
        delta = np.array(LINKS[j["child"]]["position"])-LINKS[j["parent"]]["position"]
        p1 = p+R@delta
        if j["kind"] == "prismatic":
            p1 += R@np.array({"X":[q[i],0,0],"Y":[0,q[i],0],"Z":[0,0,q[i]]}[j["axis"]])
            R1 = R
        else:
            R1 = R@rotation(j["axis"], q[i])
        frames[j["child"]] = (p1, R1)
    return frames

def quaternion_matrices(q):
    """Batch scalar-first (w,x,y,z) quaternions to (N,3,3) rotation matrices."""
    q=np.atleast_2d(np.asarray(q,dtype=float))
    q=q/np.linalg.norm(q,axis=1,keepdims=True)
    w,x,y,z=q[:,0],q[:,1],q[:,2],q[:,3]
    return np.stack([np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],axis=-1),
                     np.stack([2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],axis=-1),
                     np.stack([2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1)],axis=-2)

def center_of_mass(frames):
    return sum(v["mass"]*(frames[n][0]+frames[n][1]@v["com"]) for n,v in LINKS.items())/TOTAL_MASS

if __name__ == "__main__":
    (ROOT/"config").mkdir(exist_ok=True)
    (ROOT/"config/robot.json").write_text(json.dumps(configuration(),indent=2))
    print(f"{len(LINKS)} bodies; {len(JOINTS)} joints; {TOTAL_MASS:.2f} kg")
