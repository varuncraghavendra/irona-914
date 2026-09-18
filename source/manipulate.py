"""Arm kinematics and grasping for the Irona hands.

Damped least-squares IK over one arm's seven joints only: the legs hold the
stance and the waist stays out of the solution so reaching cannot lean the
robot off its support polygon. Engine independent; run_isaac.py owns the stage.
"""
import numpy as np
from model import JOINTS,LINKS,INDEX,rotation
from control import LO,HI,hand_pose
from vase import MODEL

ARM_JOINTS=['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','forearm_roll','wrist_pitch','wrist_roll']
_PARENT={j['child']:(i,j['parent']) for i,j in enumerate(JOINTS)}


def chain(link):
    """Joint indices from the pelvis out to `link`, root first."""
    sequence=[];node=link
    while node!='pelvis':
        index,parent=_PARENT[node];sequence.append(index);node=parent
    return sequence[::-1]


CHAIN={side:chain(side+'_hand') for side in ['left','right']}
ARM_DOF={side:np.array([INDEX[f'{side}_{name}'] for name in ARM_JOINTS],dtype=int) for side in ['left','right']}


def hand_frame(q,side,root_position,root_rotation):
    """World pose of one hand link, following only its own kinematic chain."""
    position=np.asarray(root_position,dtype=float);frame=np.asarray(root_rotation,dtype=float)
    for index in CHAIN[side]:
        joint=JOINTS[index]
        position=position+frame@(np.asarray(LINKS[joint['child']]['position'])
                                 -np.asarray(LINKS[joint['parent']]['position']))
        frame=frame@rotation(joint['axis'],q[index])
    return position,frame


def grasp_point(q,side,root_position,root_rotation):
    position,frame=hand_frame(q,side,root_position,root_rotation)
    return position+frame@MODEL.grasp_local,frame


def log_so3(R):
    """Rotation matrix to axis-angle vector."""
    angle=np.arccos(np.clip((np.trace(R)-1)/2,-1,1))
    if angle<1e-8:return np.zeros(3)
    axis=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/(2*np.sin(angle))
    return angle*axis


def palm_rotation(yaw):
    """Palm normal along the approach heading, fingers hanging down."""
    return rotation('Z',yaw)


def arm_ik(q,target,side,root_position,root_rotation,target_rotation=None,
           iterations=120,orientation_weight=.30,damping=.06,step=.7,tolerance=2e-4,
           position_tolerance=4e-3,relax=True):
    """Move one arm so its grasp point reaches `target`. Returns (q, position error).

    The wrist has only +-65 deg of pitch and +-35 deg of roll, so a palm
    orientation request can cost centimetres of reach. Solve with the requested
    weight first and relax it until the position error is acceptable: placing the
    palm on the vase matters more than the exact palm normal.
    """
    if relax and target_rotation is not None and orientation_weight>0:
        best=None
        for weight in [orientation_weight,orientation_weight/3,0.0]:
            solution,error=arm_ik(q,target,side,root_position,root_rotation,
                target_rotation=target_rotation if weight>0 else None,iterations=iterations,
                orientation_weight=weight,damping=damping,step=step,tolerance=tolerance,relax=False)
            if best is None or error<best[1]:best=(solution,error)
            if error<position_tolerance:break
        return best
    q=np.asarray(q,dtype=float).copy()
    dof=ARM_DOF[side]
    target=np.asarray(target,dtype=float)
    weights=np.array([1.,1.,1.,orientation_weight,orientation_weight,orientation_weight])
    def residual(joints):
        point,frame=grasp_point(joints,side,root_position,root_rotation)
        error=np.zeros(6);error[:3]=target-point
        if target_rotation is not None:error[3:]=log_so3(np.asarray(target_rotation)@frame.T)
        return error*weights,point
    for _ in range(iterations):
        error,point=residual(q)
        if np.linalg.norm(error[:3])<tolerance:break
        jacobian=np.zeros((6,dof.size))
        for column,index in enumerate(dof):
            probe=q.copy();probe[index]+=1e-5
            jacobian[:,column]=(residual(probe)[0]-error)/1e-5
        # jacobian is d(residual)/dq, so the descent step carries a minus sign.
        delta=-jacobian.T@np.linalg.solve(jacobian@jacobian.T+damping**2*np.eye(6),error)
        q[dof]=np.clip(q[dof]+step*delta,LO[dof],HI[dof])
    return q,float(np.linalg.norm(residual(q)[0][:3]))


def close_hand(q,side,closure):
    """Apply the finger/thumb synergy to one hand only."""
    both=hand_pose(q,closure)
    keep=np.ones(len(JOINTS),dtype=bool)
    other='left' if side=='right' else 'right'
    for i,joint in enumerate(JOINTS):
        if joint['name'].startswith(other+'_'):keep[i]=False
    result=q.copy();result[keep]=both[keep]
    return result


def attach(stage,hand_path,object_path,hand_pose,object_pose,joint_path='/World/GraspJoint'):
    """Weld the held object to the hand with a physics fixed joint.

    This is the gripper model for this demo: a five-finger friction grasp of a
    free rigid body is not reliable at this scale, so once the fingers have
    enclosed the vase the contact is replaced by a constraint, the way the
    surface-gripper samples do it. --no-grasp-assist leaves the lift to friction.

    The poses are the measured physics poses, not USD transforms: this runtime
    keeps the PhysX to USD transform write-back off.
    """
    from pxr import Gf,UsdPhysics
    if stage.GetPrimAtPath(joint_path):stage.RemovePrim(joint_path)
    joint=UsdPhysics.FixedJoint.Define(stage,joint_path)
    joint.CreateBody0Rel().SetTargets([hand_path])
    joint.CreateBody1Rel().SetTargets([object_path])
    hand_position,hand_rotation=hand_pose
    object_position,object_rotation=object_pose
    relative_position=hand_rotation.T@(np.asarray(object_position)-np.asarray(hand_position))
    relative_rotation=hand_rotation.T@object_rotation
    trace=np.trace(relative_rotation)
    w=np.sqrt(max(1+trace,0))/2
    if w>1e-6:
        vector=np.array([relative_rotation[2,1]-relative_rotation[1,2],
                         relative_rotation[0,2]-relative_rotation[2,0],
                         relative_rotation[1,0]-relative_rotation[0,1]])/(4*w)
    else:
        w=0.0;vector=np.sqrt(np.clip(np.diag(relative_rotation)+1,0,None)/2)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*relative_position.astype(float)))
    joint.CreateLocalRot0Attr(Gf.Quatf(float(w),Gf.Vec3f(*vector.astype(float))))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1,0,0,0))
    joint.CreateBreakForceAttr(1e6);joint.CreateBreakTorqueAttr(1e6)
    return joint_path


def detach(stage,joint_path='/World/GraspJoint'):
    if stage.GetPrimAtPath(joint_path):stage.RemovePrim(joint_path)
