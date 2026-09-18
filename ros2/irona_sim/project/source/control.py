"""Engine-independent controllers. All angles in radians and torques in Nm.

Walking is a slow flat-floor research gait, not a trained locomotion policy.
Flight uses only bounded forces at four physical points, with finite response.
"""
import math
import numpy as np
from model import (JOINTS, LINKS, NAMES, INDEX, THRUSTERS, TOTAL_MASS,
                   JET_TILT, JET_TIME_CONSTANT, center_of_mass)

def smooth(u):
    u=np.clip(u,0,1)
    return u*u*u*(10+u*(-15+6*u))

def skew(r):
    x,y,z=r
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

def rpy(R):
    return np.array([math.atan2(R[2,1],R[2,2]),math.asin(np.clip(-R[2,0],-1,1)),math.atan2(R[1,0],R[0,0])])

def leg_ik(foot, pelvis, side):
    sign=1 if side=="left" else -1
    d=np.asarray(foot)-(np.asarray(pelvis)+[0,sign*.075,-.020])
    roll=math.atan2(d[1],-d[2])
    dz=-math.hypot(d[1],d[2])
    distance=np.clip(math.hypot(d[0],dz),.05,.3898)
    bend=math.acos(distance/.390)
    hip=math.atan2(-d[0],-dz)-bend
    return dict(hip_yaw=0,hip_roll=roll,hip_pitch=hip,knee=2*bend,
                ankle_pitch=-hip-2*bend,ankle_roll=-roll)

def pose_for_feet(feet,pelvis):
    q=np.zeros(len(JOINTS))
    for side in ["left","right"]:
        for key,value in leg_ik(feet[side],pelvis,side).items():
            q[INDEX[side+"_"+key]]=value
        q[INDEX[side+"_elbow"]]=-.12
    return q

STAND_PELVIS=np.array([.015,0,.455])
INITIAL_FEET={"left":np.array([0,.075,.070]),"right":np.array([0,-.075,.070])}
HOME=pose_for_feet(INITIAL_FEET,STAND_PELVIS)
LO=np.array([j["limits"][0] if j["kind"]=="prismatic" else np.deg2rad(j["limits"][0]) for j in JOINTS])
HI=np.array([j["limits"][1] if j["kind"]=="prismatic" else np.deg2rad(j["limits"][1]) for j in JOINTS])
KP=np.array([j["kp"] for j in JOINTS])
KD=np.array([j["kd"] for j in JOINTS])
EFFORT=np.array([j["effort"] for j in JOINTS])
MAX_VEL=np.array([j["velocity"] for j in JOINTS])

def joint_torques(q,qd,target):
    target=np.clip(target,LO,HI)
    tau=np.clip(KP*(target-q)-KD*qd,-EFFORT,EFFORT)
    # Soft speed guard. The simulator additionally enforces position limits.
    too_fast=(np.abs(qd)>MAX_VEL)&(tau*qd>0)
    tau[too_fast]=0
    return tau

STANCE_WIDTH=.150

class Walker:
    """Quasi-static step sequence with explicit support shifts and flat soles.

    The gait plan lives in the pelvis frame, so steering is a per-step lateral
    bias on foot placement rather than a world-frame path. steer() shifts both
    feet sideways, request_stop() finishes the current step, brings the feet
    level and holds double support, and resume() restarts the sequence.
    """
    def __init__(self,step_length=.06,step_time=2.4,lift=.030,max_lateral=.015):
        self.step_length=step_length;self.step_time=step_time;self.lift=lift
        self.max_lateral=max_lateral
        self.last_step=-1;self.feet={k:v.copy() for k,v in INITIAL_FEET.items()}
        self.pelvis=STAND_PELVIS.copy();self.pelvis_start=self.pelvis.copy()
        self.swing_start=None;self.swing_goal=None
        self.phase="stand"
        self.lateral=0.0;self.time_offset=0.0;self.marching=False
        self.stop_requested=False;self.stopping_step=None;self.stopped=False

    def steer(self,lateral):
        """Sideways bias per step, positive towards the robot's left."""
        self.lateral=float(np.clip(lateral,-self.max_lateral,self.max_lateral))

    def march(self,flag=True):
        """Keep stepping and steering sideways without advancing."""
        self.marching=bool(flag)

    def request_stop(self):
        if not self.stopped:self.stop_requested=True

    def resume(self,t):
        self.time_offset=float(t)-2.0
        self.last_step=-1;self.stop_requested=False;self.stopping_step=None;self.stopped=False
        self.phase="stand"

    def stance_pose(self,root_R=None):
        return self._finish(pose_for_feet(self.feet,self.pelvis),root_R)

    def _finish(self,q,root_R):
        # Measured orientation feedback adjusts sole angle without external supports.
        if root_R is not None:
            roll,pitch,yaw=rpy(root_R)
            for side in ["left","right"]:
                q[INDEX[side+"_ankle_pitch"]]+=.65*pitch
                q[INDEX[side+"_ankle_roll"]]+=.65*roll
        return np.clip(q,LO,HI)

    def target(self,t,root_R=None):
        clock=t-self.time_offset
        if clock<2 or self.stopped:
            return self._finish(HOME.copy() if self.last_step<0 else pose_for_feet(self.feet,self.pelvis),root_R)
        u=(clock-2)/self.step_time
        step=int(u);phase=u-step
        swing="left" if step%2==0 else "right"
        stance="right" if swing=="left" else "left"
        if step!=self.last_step:
            if self.last_step>=0:
                old_swing="left" if self.last_step%2==0 else "right"
                self.feet[old_swing]=self.swing_goal.copy()
            if self.stopping_step is not None and step>self.stopping_step:
                # The levelling step is complete: hold symmetric double support.
                self.pelvis=np.array([(self.feet["left"][0]+self.feet["right"][0])/2+.015,
                                      (self.feet["left"][1]+self.feet["right"][1])/2,STAND_PELVIS[2]])
                self.stopped=True;self.phase="stand"
                return self.stance_pose(root_R)
            self.pelvis_start=self.pelvis.copy()
            self.swing_start=self.feet[swing].copy()
            self.swing_goal=self.feet[swing].copy()
            advance=0.0 if (self.stop_requested or self.marching) else self.step_length
            bias=0.0 if self.stop_requested else self.lateral
            self.swing_goal[0]=self.feet[stance][0]+advance
            self.swing_goal[1]=self.feet[stance][1]+(STANCE_WIDTH if swing=="left" else -STANCE_WIDTH)+bias
            if self.stop_requested and self.stopping_step is None:self.stopping_step=step
            self.last_step=step
        support=self.feet[stance]
        # Shift the pelvis over the stance foot, 9% of the way back towards the
        # stance centre. Scaling support[1] directly only holds while the feet
        # straddle y=0; after a few steering steps it puts the pelvis outboard of
        # the support foot and the robot topples.
        centre=(self.feet["left"][1]+self.feet["right"][1])/2
        pelvis_goal=np.array([support[0]+.015,support[1]+(centre-support[1])*.09,STAND_PELVIS[2]])
        self.pelvis=self.pelvis_start+(pelvis_goal-self.pelvis_start)*smooth(phase/.53)
        feet={k:v.copy() for k,v in self.feet.items()}
        if phase>.53:
            v=(phase-.53)/.47
            feet[swing]=self.swing_start+(self.swing_goal-self.swing_start)*smooth(v)
            feet[swing][2]+=self.lift*math.sin(math.pi*v)**2
            self.phase="swing_"+swing
        else:
            self.phase="shift_to_"+stance
        return self._finish(pose_for_feet(feet,self.pelvis),root_R)

def project_jet(force,max_force):
    """Euclidean projection into a forward circular gimbal cone and force ball."""
    v=np.array(force,dtype=float)
    rho=np.linalg.norm(v[:2]);slope=math.tan(JET_TILT)
    if rho>slope*v[2] or v[2]<0:
        z=max(0,(rho*slope+v[2])/(1+slope*slope))
        v[:2] *= (z*slope/rho) if rho>1e-12 else 0
        v[2]=z
    length=np.linalg.norm(v)
    if length>max_force:
        v*=max_force/length
    return v

class FlightController:
    def __init__(self,dt):
        self.dt=dt;self.forces=np.zeros((4,3));self.origin=None
        self.allocation_residual=np.zeros(6)
        self.target_com=None

    def update(self,t,frames,com_velocity,angular_velocity,root_R,duration=16):
        com=center_of_mass(frames)
        if self.origin is None:
            self.origin=com.copy()
        if t<2:
            self.origin=com.copy();self.forces[:]=0
            return self.forces.copy()
        flight_time=t-2
        rise=smooth(flight_time/3)
        # Final four seconds descend to the starting COM elevation.
        fall=1-smooth((t-(duration-4))/3.5)
        self.target_com=self.origin+np.array([0,0,.65*rise*fall])
        acceleration=4.0*(self.target_com-com)-3.3*np.asarray(com_velocity)
        force=TOTAL_MASS*(acceleration+np.array([0,0,9.81]))
        attitude=.5*np.array([root_R[2,1]-root_R[1,2],root_R[0,2]-root_R[2,0],root_R[1,0]-root_R[0,1]])
        torque=-22*attitude-7*np.asarray(angular_velocity)
        wanted=np.r_[force,torque]
        A=np.zeros((6,12))
        for i,jet in enumerate(THRUSTERS):
            p,R=frames[jet["body"]]
            point=p+R@jet["point"]
            A[:3,3*i:3*i+3]=R
            A[3:,3*i:3*i+3]=skew(point-com)@R
        # Metres-to-newtons moment weighting; no unmodelled free torque is applied.
        W=np.diag([1,1,1,4,4,4]);Aw=W@A;bw=W@wanted
        f=np.linalg.lstsq(Aw,bw,rcond=None)[0].reshape(4,3)
        L=np.linalg.norm(Aw,2)**2+1e-9
        for _ in range(45):
            for i,jet in enumerate(THRUSTERS):
                f[i]=project_jet(f[i],jet["max_force"])
            f-=((Aw.T@(Aw@f.ravel()-bw))/L).reshape(4,3)
        for i,jet in enumerate(THRUSTERS):
            f[i]=project_jet(f[i],jet["max_force"])
        if t>duration-.35 and com[2]<self.origin[2]+.035:
            f[:]=0
        alpha=1-math.exp(-self.dt/JET_TIME_CONSTANT)
        self.forces+=alpha*(f-self.forces)
        self.allocation_residual=A@self.forces.ravel()-wanted
        return self.forces.copy()

ARM_HAND_INDICES=np.array([i for i,j in enumerate(JOINTS) if j['actuator']!='custom high-torque drive'],dtype=int)


def hand_pose(target,closure):
    """Software tendon-style pose synergy; every phalanx remains a contact body.

    This couples TARGETS, not physical tendons or rigid mimic constraints.
    """
    q=np.asarray(target).copy();c=float(np.clip(closure,0,1))
    for side,sign in [('left',1),('right',-1)]:
        for finger in ['index','middle','ring','little']:
            for segment,angle in [('mcp',-50),('pip',-65),('dip',-40)]:
                q[INDEX[f'{side}_{finger}_{segment}']]=np.deg2rad(angle)*c
        for segment,angle in [('opposition',sign*40),('spread',sign*25),('mcp',-40),('ip',-45)]:
            q[INDEX[f'{side}_thumb_{segment}']]=np.deg2rad(angle)*c
    return q


def hands_demo(t):
    # Six-second repeat with one-second open/closed dwells and smooth transitions.
    phase=t%8
    c=smooth((phase-1)/2) if phase<4 else 1-smooth((phase-5)/2)
    return hand_pose(HOME,c)


class TargetLimiter:
    """Joint-range and command-velocity limiting shared by both runtime backends."""
    def __init__(self,dt):self.q=HOME.copy();self.dt=dt
    def update(self,target):
        self.q+=np.clip(np.clip(target,LO,HI)-self.q,-MAX_VEL*self.dt,MAX_VEL*self.dt)
        return self.q.copy()
