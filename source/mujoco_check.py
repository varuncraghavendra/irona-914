"""Independent dynamics check of the SAME joints, inertias and collision boxes.

This is not an Isaac Sim / PhysX test. Generates its own report and optional video.
pip install mujoco numpy
"""
import os
os.environ.setdefault("MUJOCO_GL","egl")
from pathlib import Path
import argparse,json,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from model import *
from control import *

def vec(v): return " ".join(f"{x:.9g}" for x in v)

def build_mjcf():
    root=ET.Element("mujoco",model="Irona_914_dynamics_check")
    ET.SubElement(root,"compiler",angle="radian",inertiafromgeom="false",balanceinertia="false")
    ET.SubElement(root,"option",timestep=str(DT),gravity="0 0 -9.81",integrator="implicitfast",iterations="100",tolerance="1e-10")
    world=ET.SubElement(root,"worldbody")
    ET.SubElement(world,"geom",name="floor",type="plane",size="10 10 .1",friction=".8 .01 .001",rgba=".2 .23 .27 1")
    ET.SubElement(world,"light",pos="2 -2 3",dir="-1 1 -2",directional="true")
    ET.SubElement(world,"camera",name="overview",pos="1.6 -2 1.1",xyaxes=".78 .625 0 -.13 .16 .976")
    bodies={}
    j_by_child={j["child"]:j for j in JOINTS}
    for n,v in LINKS.items():
        if n=="pelvis":
            parent=world;pos=v["position"]
        else:
            j=j_by_child[n];parent=bodies[j["parent"]]
            pos=np.array(v["position"])-LINKS[j["parent"]]["position"]
        b=ET.SubElement(parent,"body",name=n,pos=vec(pos));bodies[n]=b
        if n=="pelvis":ET.SubElement(b,"freejoint",name="floating_base")
        else:
            j=j_by_child[n]
            lim=j["limits"] if j["kind"]=="prismatic" else np.deg2rad(j["limits"])
            ET.SubElement(b,"joint",name=j["name"],type="slide" if j["kind"]=="prismatic" else "hinge",
                          axis={"X":"1 0 0","Y":"0 1 0","Z":"0 0 1"}[j["axis"]],range=vec(lim),damping="0",armature="0")
        ET.SubElement(b,"inertial",pos=vec(v["com"]),mass=str(v["mass"]),diaginertia=vec(v["inertia"]))
        if v["collision"]:
            ET.SubElement(b,"geom",name=n+"_collision",type="box",pos=vec(v["com"]),size=vec(np.array(v["size"])/2),
                          friction=".8 .01 .001",rgba=".62 .69 .73 1",condim="4",solref=".006 1",solimp=".95 .99 .001")
    actuators=ET.SubElement(root,"actuator")
    for j in JOINTS:
        limits=j["limits"] if j["kind"]=="prismatic" else np.deg2rad(j["limits"])
        ET.SubElement(actuators,"position",name=j["name"],joint=j["name"],gear="1",kp=str(j["kp"]),kv=str(j["kd"]),
                      ctrllimited="true",ctrlrange=vec(limits),forcelimited="true",forcerange=vec([-j["effort"],j["effort"]]))
    contacts=ET.SubElement(root,"contact")
    graph={n:set() for n in LINKS}
    for j in JOINTS:
        graph[j["parent"]].add(j["child"]);graph[j["child"]].add(j["parent"])
    for n in LINKS:
        seen={n};f={n}
        for _ in range(3): f=set().union(*(graph[x] for x in f))-seen;seen|=f
        for other in seen-{n}:
            if n<other: ET.SubElement(contacts,"exclude",body1=n,body2=other)
    xml=ET.tostring(root,encoding="unicode")
    (ROOT/"validation/irona_check.xml").write_text(xml)
    return xml

def run(mode="stand",duration=16,save=True):
    m=mujoco.MjModel.from_xml_string(build_mjcf());d=mujoco.MjData(m)
    qadr=np.array([m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in NAMES])
    vadr=np.array([m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in NAMES])
    bids={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in LINKS}
    d.qpos[qadr]=HOME;d.qpos[:3]=STAND_PELVIS;d.qpos[3:7]=[1,0,0,0]
    d.ctrl[:]=HOME
    mujoco.mj_forward(m,d)
    walker=Walker();flight=FlightController(DT);limiter=TargetLimiter(DT);history=[];peak_torque=np.zeros(len(JOINTS))
    peak_speed=np.zeros(len(JOINTS));max_tilt=0;fallen=False;jet_total_peak=0
    initial=d.qpos[:3].copy()
    for step in range(int(duration/DT)):
        t=step*DT
        frames={n:(d.xpos[b].copy(),d.xmat[b].reshape(3,3).copy()) for n,b in bids.items()}
        R=frames["pelvis"][1]
        q=d.qpos[qadr];qd=d.qvel[vadr]
        target=walker.target(t,R) if mode=="walk" else (hands_demo(t) if mode=="hands" else HOME)
        d.ctrl[:]=limiter.update(target)
        tau=d.actuator_force.copy()
        peak_torque=np.maximum(peak_torque,np.abs(tau));peak_speed=np.maximum(peak_speed,np.abs(qd))
        d.xfrc_applied[:]=0;jetforces=np.zeros((4,3))
        if mode=="flight":
            # subtree COM linear velocity is computed in world frame by mj_subtreeVel.
            mujoco.mj_subtreeVel(m,d)
            cv=d.subtree_linvel[bids["pelvis"]]
            vel6=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,bids["pelvis"],vel6,0)
            jetforces=flight.update(t,frames,cv,vel6[:3],R,duration)
            for jet,force_local in zip(THRUSTERS,jetforces):
                b=bids[jet["body"]];p,Rj=frames[jet["body"]]
                force=Rj@force_local;point=p+Rj@jet["point"]
                d.xfrc_applied[b,:3]+=force
                d.xfrc_applied[b,3:]+=np.cross(point-d.xipos[b],force)
            jet_total_peak=max(jet_total_peak,float(np.linalg.norm(jetforces,axis=1).sum()))
        tilt=np.arccos(np.clip(R[2,2],-1,1));max_tilt=max(max_tilt,tilt)
        if d.qpos[2]<.24 or tilt>np.deg2rad(55):
            fallen=True
            break
        if step%8==0:
            # Store actual simulated poses, never a kinematic animation.
            fz={"left":0.,"right":0.}
            for ci in range(d.ncon):
                con=d.contact[ci]
                gn=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in [con.geom1,con.geom2]]
                if "floor" in gn:
                    fc=np.zeros(6);mujoco.mj_contactForce(m,d,ci,fc)
                    for side in fz:
                        if side+"_foot_collision" in gn:fz[side]+=max(0,fc[0])
            history.append(dict(t=t,root=d.qpos[:3].tolist(),root_quat_wxyz=d.qpos[3:7].tolist(),q=q.tolist(),tilt_deg=float(np.rad2deg(tilt)),
                                foot_fz=fz,jet_norms=np.linalg.norm(jetforces,axis=1).tolist(),
                                com=center_of_mass(frames).tolist()))
        mujoco.mj_step(m,d)
    report=dict(engine="MuJoCo "+mujoco.__version__,isaac_sim_tested=False,mode=mode,
                requested_duration_s=duration,simulated_duration_s=float(d.time),fallen=fallen,
                root_displacement_m=(d.qpos[:3]-initial).tolist(),max_tilt_deg=float(np.rad2deg(max_tilt)),
                min_root_height_m=min(x["root"][2] for x in history),max_root_height_m=max(x["root"][2] for x in history),
                peak_torque_Nm_or_N=dict(zip(NAMES,peak_torque.tolist())),peak_speed_rad_s_or_m_s=dict(zip(NAMES,peak_speed.tolist())),
                peak_total_jet_force_N=jet_total_peak,
                limitations="Independent proxy dynamics only; not PhysX, actuator hardware, shell clearance or real jet validation.")
    if save:
        (ROOT/f"validation/{mode}_report.json").write_text(json.dumps(report,indent=2))
        (ROOT/f"validation/{mode}_trajectory.json").write_text(json.dumps(history,separators=(",",":")))
    print(json.dumps({k:v for k,v in report.items() if not k.startswith("peak_torque") and not k.startswith("peak_speed")},indent=2))
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["stand","walk","flight","hands"],default="stand");p.add_argument("--seconds",type=float,default=16)
    a=p.parse_args();run(a.mode,a.seconds)
