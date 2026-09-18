"""Validate real USD composition, inertias, joint frames and exported print meshes."""
import json,csv,hashlib,ast
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd,UsdGeom,UsdPhysics
from model import *
from control import *

def main():
    report={"checks":{},"limitations":["Isaac Sim / PhysX not installed in the authoring environment.",
            "No physical print, actuator test, structural FEA or measured inertial identification.",
            "Collision boxes are simplified; full moving shell clearance is not certified."]}
    stage=Usd.Stage.Open(str(ROOT/"assets/irona.usd"));assert stage
    scene=Usd.Stage.Open(str(ROOT/"assets/scene.usda"));assert scene.GetPrimAtPath("/World/Irona/Links/pelvis")
    assert not stage.GetCompositionErrors();assert not scene.GetCompositionErrors()
    assert UsdGeom.GetStageMetersPerUnit(stage)==1
    assert UsdGeom.GetStageUpAxis(stage)=="Z"
    rigid=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    roots=[p for p in rigid if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(rigid)==len(LINKS) and len(joints)==len(JOINTS) and len(roots)==1
    assert str(roots[0].GetPath())=="/Irona/Links/pelvis"
    assert not any(p.IsA(UsdPhysics.FixedJoint) for p in stage.Traverse())
    xf=UsdGeom.XformCache();mass_sum=0
    for n,v in LINKS.items():
        p=stage.GetPrimAtPath("/Irona/Links/"+n);api=UsdPhysics.MassAPI(p)
        mass_sum+=api.GetMassAttr().Get()
        inertia=np.asarray(api.GetDiagonalInertiaAttr().Get())
        assert (inertia>0).all() and 2*max(inertia)<=sum(inertia)+1e-10
    assert abs(mass_sum-TOTAL_MASS)<1e-5
    for j in JOINTS:
        prim=stage.GetPrimAtPath("/Irona/Joints/"+j["name"]);obj=UsdPhysics.Joint(prim)
        b0=obj.GetBody0Rel().GetTargets()[0];b1=obj.GetBody1Rel().GetTargets()[0]
        a=xf.GetLocalToWorldTransform(stage.GetPrimAtPath(b0)).Transform(obj.GetLocalPos0Attr().Get())
        b=xf.GetLocalToWorldTransform(stage.GetPrimAtPath(b1)).Transform(obj.GetLocalPos1Attr().Get())
        assert np.linalg.norm(np.asarray(a)-b)<1e-6,j["name"]
        drive=UsdPhysics.DriveAPI(prim,"linear" if j["kind"]=="prismatic" else "angular")
        assert abs(drive.GetMaxForceAttr().Get()-j["effort"])<1e-6
        multiplier=1 if j["kind"]=="prismatic" else np.pi/180
        assert abs(drive.GetStiffnessAttr().Get()-j["kp"]*multiplier)<1e-5
    for jet in THRUSTERS:
        p=stage.GetPrimAtPath("/Irona/Links/"+jet["body"]+"/"+jet["name"])
        assert p.IsA(UsdGeom.Xform),jet["name"]+" must remain a force frame, not a visual mesh"
    verts=[]
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh):
            M=xf.GetLocalToWorldTransform(p)
            points=UsdGeom.Mesh(p).GetPointsAttr().Get()
            verts.append(np.array([M.Transform(Gf.Vec3d(*v)) for v in points]))
    points=np.concatenate(verts)
    bounds=np.stack([points.min(0),points.max(0)])
    assert abs(bounds[1,2]-bounds[0,2]-HEIGHT)<1e-6
    report["checks"].update(usd_composition="pass",units="metres, Z-up",rigid_bodies=len(rigid),
            actuated_joints=len(joints),free_floating_root="pass",joint_anchor_coincidence="pass",
            positive_physical_inertias="pass",mass_kg=mass_sum,geometry_bounds_m=bounds.tolist(),
            height_mm=float(1000*(bounds[1,2]-bounds[0,2])),jet_frames="pass")
    failures=[];printed=[]
    for f in sorted((ROOT/"print").rglob("*.stl")):
        m=trimesh.load_mesh(f,process=True)
        good=m.is_watertight and m.is_winding_consistent and m.volume>0
        dims=m.extents
        if not good or dims.max()>200.01:
            failures.append(dict(file=str(f.relative_to(ROOT)),watertight=m.is_watertight,
                                 winding=m.is_winding_consistent,volume=float(m.volume),size=dims.tolist()))
        printed.append(dict(file=str(f.relative_to(ROOT)),watertight=bool(m.is_watertight),
                            volume_cm3=float(m.volume/1000),size_mm=dims.tolist()))
    report["checks"]["print_mesh_count"]=len(printed)
    report["checks"]["all_print_meshes_watertight_and_bed_sized"]=not failures
    report["print_failures"]=failures
    report["print_meshes"]=printed
    assert not failures,json.dumps(failures,indent=2)
    frames=forward_kinematics(HOME,STAND_PELVIS)
    for side in ["left","right"]:
        assert np.linalg.norm(frames[side+"_foot"][0]-INITIAL_FEET[side])<1e-9
        assert np.linalg.norm(frames[side+"_foot"][1]-np.eye(3))<1e-9
    report["checks"]["leg_ik_home_pose"]= "pass"
    com=center_of_mass(frames);A=np.zeros((6,12))
    for i,j in enumerate(THRUSTERS):
        p,R=frames[j["body"]];A[:3,3*i:3*i+3]=R;A[3:,3*i:3*i+3]=skew(p+R@j["point"]-com)@R
    assert np.linalg.matrix_rank(A)==6
    report["checks"]["vectored_thrust_allocation_rank"]=6
    for file in (ROOT/"source").glob("*.py"):ast.parse(file.read_text())
    report["checks"]["python_syntax"]="pass"
    (ROOT/"validation/asset_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"checks":report["checks"],"failures":failures},indent=2))

if __name__=="__main__":
    from pxr import Gf
    main()
