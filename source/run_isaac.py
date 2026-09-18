"""Launch with Isaac Sim 5.x's python.sh, not system Python.

Example: ./python.sh /path/irona_914/source/run_isaac.py --mode walk
This integration is source/API checked but has NOT been executed in Isaac Sim.
"""
import argparse
import sys
import traceback
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode",choices=["stand","walk","flight","hands","fetch","airfetch"],default="stand")
parser.add_argument("--seconds",type=float,default=60)
parser.add_argument("--headless",action="store_true")
parser.add_argument("--log",type=Path,default=None)
parser.add_argument("--ros2",action="store_true",help="Native ROS 2 sensors, clock, joint states and root TF")
parser.add_argument("--no-camera",action="store_true")
parser.add_argument("--no-lidar",action="store_true")
parser.add_argument("--no-arena",action="store_true")
parser.add_argument("--export-stage",type=Path,default=None,help="Export runtime-composed USD with registered graphs")
parser.add_argument("--usd-transforms",action="store_true",help="Mirror physics transforms into USD (about 5x slower)")
parser.add_argument("--scene",choices=["house","arena"],default="house",
                    help="Furnished room with a scanned vase, or the coloured-block perception arena")
parser.add_argument("--vase",type=float,nargs=2,metavar=("X","Y"),default=[1.0,-.10],help="Arena vase axis position, world metres")
parser.add_argument("--table",type=float,nargs=2,metavar=("X","Y"),default=[1.45,-.10],help="House coffee table centre, world metres")
parser.add_argument("--vase-scale",type=float,default=.72,help="Scale applied to the scanned vase asset")
parser.add_argument("--vase-inset",type=float,default=.13,
                    help="House scene: how far the vase stands in from the table's near edge. "
                         "The body has to sit 0.28 m from the vase to reach it, so a larger inset "
                         "moves the robot closer to the table, and past about 0.28 m it cannot reach at all")
parser.add_argument("--pedestal",type=float,default=.32,help="Pedestal height in metres")
parser.add_argument("--no-vase",action="store_true")
parser.add_argument("--vase-boxes",action="store_true",help="Box collision proxies instead of cylinders")
parser.add_argument("--grasp-side",choices=["left","right"],default="right")
parser.add_argument("--arm-effort-scale",type=float,default=2.0,
                    help="Multiply the arm joint torque caps. The shipped 4 Nm shoulder cap is a conservative "
                         "design value and saturates while holding the arm out at a table, which drops the palm")
parser.add_argument("--no-grasp-assist",action="store_true",help="Lift on finger friction alone, without the fixed joint")
parser.add_argument("--carry",action="store_true",help="Resume walking after the lift")
parser.add_argument("--grasp-policy",choices=["scripted","groot","vlad"],default="scripted",
                    help="Who performs the grasp: the scripted IK sequence with the hue detector, "
                         "GR00T N1.5 over the policy server, or vlad = the zero-shot VLAD-Grasp-derived "
                         "grasp synthesis (no colour prior, no object model, no training)")
parser.add_argument("--place-offset",type=float,nargs=2,metavar=("FORWARD","OUTBOARD"),default=None,
                    help="Turn the fetch into pick and place: set the object down this far "
                         "(metres, pelvis frame, outboard on the grasping side) from where it was picked up")
parser.add_argument("--live-dashboard",type=Path,default=None,
                    help="Publish annotated frames and the intended next actions here for "
                         "scripts/live_dashboard.py")
parser.add_argument("--live-period",type=float,default=.25,
                    help="Seconds of simulated time between live dashboard updates")
parser.add_argument("--groot-host",default="127.0.0.1")
parser.add_argument("--groot-port",type=int,default=5599)
parser.add_argument("--groot-instruction",default="pick up the vase")
parser.add_argument("--groot-seconds",type=float,default=10.0,
                    help="How long GR00T drives the arm before the scripted grasp takes over")
parser.add_argument("--hands",choices=["lead","both"],default="lead",
                    help="'both' mirrors the grasping arm onto the other one as a counterweight, "
                         "cancelling the roll and yaw a single-arm reach throws into a hovering "
                         "base. The arms cannot cross to the centre line, so one object cannot be "
                         "pinched between them.")
parser.add_argument("--perception-log",action="store_true",help="Print every vase detection")
parser.add_argument("--dashboard",type=Path,default=None,help="Record annotated frames into this directory")
parser.add_argument("--dashboard-period",type=float,default=.5,help="Seconds of simulated time between recorded frames")
args=parser.parse_args()
if args.mode=="flight" and args.seconds<=0:parser.error("Flight needs a finite --seconds duration")

# SimulationApp must be imported/started before Kit-dependent modules.
from isaacsim import SimulationApp
app=SimulationApp({"headless":args.headless,"width":1600,"height":1000,"renderer":"RayTracedLighting"})

try:
    import json
    import numpy as np
    from pxr import Gf,UsdGeom,UsdLux,PhysxSchema
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation,RigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.viewports import set_camera_view
    from model import ROOT,LINKS,JOINTS,NAMES,DT,THRUSTERS,TOTAL_MASS,center_of_mass,quaternion_matrices
    from control import HOME,STAND_PELVIS,Walker,FlightController,LO,HI,TargetLimiter,hands_demo
    from manipulate import ARM_JOINTS

    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.sensors.rtx")
    enable_extension("isaacsim.core.nodes")
    # With the USD transform write-back off, PhysX must publish poses to Fabric or
    # the renderer, the RTX sensors and every ROS image stay frozen at frame zero.
    if not args.usd_transforms:enable_extension("omni.physx.fabric")
    if args.ros2:enable_extension("isaacsim.ros2.bridge")
    app.update()
    world=World(stage_units_in_meters=1.0,physics_dt=DT,rendering_dt=1/60,backend="numpy",device="cpu")
    stage=world.stage
    world.scene.add_default_ground_plane(static_friction=.9,dynamic_friction=.8,restitution=0)
    add_reference_to_stage(str(ROOT/"assets/irona.usd"),"/World/Irona")
    scene_report=None
    if args.scene=="house":
        from house import VASE_PATH,build_house
        from isaacsim.storage.native import get_assets_root_path
        assets_root=get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Isaac Sim asset root is unreachable; use --scene arena")
        from vase import MODEL
        MODEL.grasp_fraction=.42     # grip the body below its widest point
        scene_report=build_house(stage,assets_root,table_position=args.table,vase_scale=args.vase_scale,
                                 vase_inset=args.vase_inset)
        vase_rest=scene_report["centre"]
        # The room shell draws the floor; the default ground plane stays for its
        # known friction, just out of sight.
        ground=stage.GetPrimAtPath("/World/defaultGroundPlane")
        if ground:UsdGeom.Imageable(ground).MakeInvisible()
        print(f"House scene: vase radius {MODEL.radius*1000:.0f} mm, height {MODEL.height*1000:.0f} mm, "
              f"grasp height {scene_report['grasp_z']:.3f} m, table edge x={scene_report['table_edge']:.3f}",flush=True)
    else:
        if not args.no_arena:
            from arena import add_arena
            add_arena(stage)
        from vase import VASE_PATH,add_vase
        vase_rest=None if args.no_vase else add_vase(stage,position=args.vase,pedestal_height=args.pedestal,
                                                     cylinders=not args.vase_boxes)
    root_path="/World/Irona/Links/pelvis"
    robot=world.scene.add(SingleArticulation(prim_path=root_path,name="irona"))
    # Force the expected CPU/TGS settings explicitly rather than inherit UI defaults.
    phys=world.get_physics_context()
    phys.enable_gpu_dynamics(False)
    phys.set_solver_type("TGS")
    # Writing 63 link transforms into USD every 1 ms costs ~6 ms/step (5x the whole
    # simulation). The renderer, RTX sensors and ROS graphs read Fabric, and this
    # runtime reads poses from the physics tensor views, so keep the USD write off
    # unless a tool that walks the USD stage needs it.
    phys.set_physx_update_transformations_settings(update_to_usd=args.usd_transforms,
        update_velocities_to_usd=args.usd_transforms,output_velocities_local_space=False)
    PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(root_path)).CreateSolverPositionIterationCountAttr(32)
    PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(root_path)).CreateSolverVelocityIterationCountAttr(8)

    if args.arm_effort_scale!=1.0:
        # Raise the torque caps on the arm drives only. Holding the arm extended at
        # grasp height needs more than the shipped 4 Nm: the drive saturates, the
        # hand sags about 14 degrees at the shoulder and the fingers catch the table.
        scaled=0
        for side in ("left","right"):
            for name in ARM_JOINTS:
                prim=stage.GetPrimAtPath(f"/World/Irona/Joints/{side}_{name}")
                if not prim:continue
                for attribute in prim.GetAttributes():
                    if attribute.GetName().endswith("physics:maxForce"):
                        value=attribute.Get()
                        if value:attribute.Set(float(value)*args.arm_effort_scale);scaled+=1
        print(f"Arm torque caps scaled by {args.arm_effort_scale:g} on {scaled} drives",flush=True)

    light=UsdLux.DomeLight.Define(stage,"/World/IronaDome");light.CreateIntensityAttr(800)
    key=UsdLux.DistantLight.Define(stage,"/World/IronaKey");key.CreateIntensityAttr(1800)
    key.AddRotateXYZOp().Set(Gf.Vec3f(-35,-25,20))
    set_camera_view(eye=[1.5,-1.9,1.15],target=[.05,0,.47])

    world.reset()
    # Resolve names after PhysX initializes; never assume the USD traversal order.
    runtime_names=list(robot.dof_names)
    indices=np.array([runtime_names.index(n) for n in NAMES],dtype=np.int32)
    for i,j in enumerate(JOINTS):
        expected="PhysicsPrismaticJoint" if j["kind"]=="prismatic" else "PhysicsRevoluteJoint"
        assert stage.GetPrimAtPath("/World/Irona/Joints/"+j["name"]).GetTypeName()==expected
    robot.set_world_pose(position=STAND_PELVIS,orientation=np.array([1.,0.,0.,0.]))
    robot.set_joint_positions(HOME,joint_indices=indices)
    robot.set_joint_velocities(np.zeros(len(NAMES)),joint_indices=indices)
    robot.apply_action(ArticulationAction(joint_positions=HOME,joint_indices=indices))
    robot.set_linear_velocity(np.zeros(3));robot.set_angular_velocity(np.zeros(3))

    # One physics-tensor view over every link. Resolve the row order from the view's
    # own prim paths instead of assuming it. prepare_contact_sensors defaults to True,
    # which re-authors the collision prims and invalidates the PhysX simulation view
    # (Isaac Sim 5.1), so every later link pose/velocity read fails. Keep it off.
    bodies=RigidPrim(prim_paths_expr="/World/Irona/Links/.*",name="irona_links",
                     reset_xform_properties=False,prepare_contact_sensors=False)
    bodies.initialize()
    rows={path.rsplit('/',1)[-1]:i for i,path in enumerate(bodies.prim_paths)}
    assert set(rows)==set(LINKS),"link view does not match the model"
    order=list(LINKS)
    body_rows=np.array([rows[n] for n in order],dtype=np.int32)
    body_mass=np.array([LINKS[n]["mass"] for n in order])
    walker=Walker();flight=FlightController(DT);limiter=TargetLimiter(DT)
    vase_view=None
    if vase_rest is not None:
        vase_view=RigidPrim(prim_paths_expr=VASE_PATH,name="vase_body",reset_xform_properties=False,
                            prepare_contact_sensors=False)
        vase_view.initialize()
    mission=None;recorder=None;live=None;plan=None;finder=None
    if args.mode in ("fetch","airfetch"):
        from manipulate import attach,detach
        from mission import FetchMission
        from vase import MODEL
        if vase_rest is None:raise RuntimeError(f"--mode {args.mode} needs the vase; drop --no-vase")
        if args.grasp_policy=="vlad":
            # Geometric, colour-blind detector: it finds whatever stands on the
            # support surface and fits a grasp to it, with no knowledge of the vase.
            from zeroshot_grasp import ZeroShotGraspFinder
            # Four times the pixels of the hue detector's product. The grasp is
            # fitted to a cross-section a few pixels wide, so angular resolution is
            # what sets the range at which a grasp can be fitted at all.
            finder=ZeroShotGraspFinder("/World/Irona/Links/head/Sensors/depth",resolution=(640,360))
            print("Grasp source: VLAD-Grasp-derived zero-shot synthesis "
                  "(arXiv:2511.05791 stages 2-3, local analytic stage 1)",flush=True)
        else:
            from perception import VaseFinder
            finder=VaseFinder("/World/Irona/Links/head/Sensors/depth",resolution=(320,180))
        finder.initialize()
        # Let the synthetic-data pipeline produce its first frames before the mission
        # starts asking the detector for data.
        for _ in range(6):world.render()
        policy=None
        if args.grasp_policy=="groot":
            from groot_client import GrootClient
            policy=GrootClient(host=args.groot_host,port=args.groot_port,
                               instruction=args.groot_instruction,side=args.grasp_side)
            if policy.available():
                print(f"GR00T N1.5 policy server reachable at {args.groot_host}:{args.groot_port}",flush=True)
            else:
                print(f"GR00T policy server unreachable ({policy.last_error}); "
                      f"start scripts/groot_server.py or use --grasp-policy scripted",flush=True)
                raise RuntimeError("GR00T policy server is not running")
        hands=([args.grasp_side,"left" if args.grasp_side=="right" else "right"]
               if args.hands=="both" else [args.grasp_side])
        if len(hands)>1:
            print(f"Grasping with the {args.grasp_side} arm, mirroring it on the "
                  f"{hands[1]} arm as a counterweight",flush=True)
        mission_walker=walker
        if args.mode=="airfetch":
            from airfetch import AirFetchPlan,StanceWalker,com_setpoint
            mission_walker=StanceWalker()
            plan=AirFetchPlan(side=args.grasp_side,walker=mission_walker,controller=flight)
            if args.place_offset is not None:
                print("--place-offset is ignored in airfetch: the object is flown home, not set down",
                      flush=True)
        mission=FetchMission(mission_walker,finder,side=args.grasp_side,
                             grasp_assist=not args.no_grasp_assist,
                             verbose=args.perception_log,policy=policy,policy_seconds=args.groot_seconds,
                             place_offset=None if args.mode=="airfetch" else args.place_offset,
                             hands=hands)
        if args.dashboard is not None:
            from recorder import FrameRecorder
            recorder=FrameRecorder(args.dashboard,period=args.dashboard_period)
        if args.live_dashboard is not None:
            from live_view import LivePublisher
            live=LivePublisher(args.live_dashboard,period=args.live_period)
            print(f"Live dashboard feed: {args.live_dashboard} "
                  f"(serve it with scripts/live_dashboard.py)",flush=True)
    bridge=None
    if args.ros2:
        from ros_bridge import RosBridge
        bridge=RosBridge(stage,camera=not args.no_camera,lidar=not args.no_lidar)
    if args.export_stage:
        args.export_stage.parent.mkdir(parents=True,exist_ok=True)
        stage.Flatten().Export(str(args.export_stage))
    # The RTX sensors only tick when the application drives the update, so physics
    # runs from a callback and the main loop calls world.step(render=True). That
    # keeps control at the 1 ms physics rate while the renderer, the RTX lidar and
    # the ROS graphs advance once per frame.
    log=[];runtime=dict(step=0,time=0.0,fallen=False,stop=False,error=None,base_tf=None,step_size=DT,
                        vase=None,vase_rotation=None,flight_target=None,frames=None,vase_settled=None,
                        stamp=-1.0,stamp_at=0.0,stale=False)

    def control(step_size):
        if runtime["stop"]:return
        try:
            # Take simulation time from the physics step itself. The application
            # drives the substeps, and their size is not guaranteed to equal the
            # requested physics_dt, so a step counter would drift from sim time.
            runtime["time"]+=float(step_size);runtime["step_size"]=float(step_size)
            step=runtime["step"]
            # One batched read per step. Isaac Sim quaternions are scalar-first.
            positions,quats=bodies.get_world_poses()
            velocities=np.asarray(bodies.get_velocities(),dtype=float)[body_rows]
            points=np.asarray(positions,dtype=float)[body_rows]
            rotations=quaternion_matrices(np.asarray(quats,dtype=float)[body_rows])
            frames={n:(points[k],rotations[k]) for k,n in enumerate(order)}
            # RigidPrim reports COM linear velocity followed by angular velocity.
            com_velocity=(body_mass[:,None]*velocities[:,:3]).sum(0)/TOTAL_MASS
            root_angular=velocities[order.index("pelvis")][3:]
            pelvis_velocity=velocities[order.index("pelvis")][:3]
            t=runtime["time"];R=frames["pelvis"][1]
            vase_pose=None
            if vase_view is not None:
                vase_position,vase_quat=vase_view.get_world_poses()
                vase_pose=(np.asarray(vase_position[0],dtype=float),
                           quaternion_matrices(np.asarray(vase_quat,dtype=float))[0])
            if mission is not None:
                measured=robot.get_joint_positions()[indices]
                target=mission.update(t,frames,None if vase_pose is None else vase_pose[0],measured)
                # A render product can stop updating while get_current_frame() keeps
                # handing back the last image: the detector then re-segments a still
                # every tick and looks perfectly healthy while the mission is blind.
                # Watch the render stamp, not finder.frames -- that only counts calls.
                if finder.stamp>runtime["stamp"]:
                    runtime["stamp"]=finder.stamp;runtime["stamp_at"]=t;runtime["stale"]=False
                elif t-runtime["stamp_at"]>2.0 and not runtime["stale"]:
                    runtime["stale"]=True
                    print(f"[{t:7.2f}s] camera   no new render for {t-runtime['stamp_at']:.1f}s "
                          f"(last stamp {finder.stamp:.3f}s, render frame {finder.render_frame}, "
                          f"detector says '{finder.status}'); the mission is running on a still",
                          flush=True)
                if mission.attach_request:
                    attach(stage,"/World/Irona/Links/"+args.grasp_side+"_hand",VASE_PATH,
                           frames[args.grasp_side+"_hand"],vase_pose)
                    mission.confirm_attach(t)
                    mission.note(t,"vase welded to the palm (grasp assist)")
                if mission.detach_request:
                    detach(stage)
                    mission.confirm_detach(t)
                    mission.note(t,"grasp constraint released; the object stands on its own")
                if plan is not None:
                    desired_pelvis=plan.update(t,frames,mission,pelvis_velocity,runtime["step_size"])
                    runtime["flight_target"]=desired_pelvis
                if live is not None and live.due(t):
                    annotation=dict(grasp_policy=mission.policy_note)
                    if plan is not None:annotation["flight"]=plan.summary()
                    live.publish(t,finder,mission,frames,vase_pose,extra=annotation)
                if recorder is not None and recorder.due(t):
                    annotation=dict(policy=mission.policy_note)
                    if mission.policy is not None and mission.policy_chunk is not None:
                        chunk=mission.policy_chunk
                        annotation["groot"]=dict(latency_ms=mission.policy.last_latency,
                            calls=mission.policy_calls,
                            right_arm=[round(float(v),3) for v in np.asarray(chunk[args.grasp_side+"_arm"])[0][:3]],
                            hand=[round(float(v),3) for v in np.asarray(chunk[args.grasp_side+"_hand"])[0][:3]])
                    recorder.capture(t,finder,mission,frames,vase_pose,extra=annotation)
                if args.carry and mission.state=="done" and walker.stopped:
                    walker.resume(t);mission.note(t,"resuming the walk while holding the vase")
            else:
                target=walker.target(t,R) if args.mode=="walk" else (hands_demo(t) if args.mode=="hands" else HOME.copy())
            if bridge:target=bridge.command_target(target)
            target=limiter.update(target)
            robot.apply_action(ArticulationAction(joint_positions=np.clip(target,LO,HI),joint_indices=indices))
            # Native implicit, force-limited USD drives apply joint torque. No root pose
            # or velocity setters are called inside this simulation loop.
            jet_forces=np.zeros((4,3))
            if args.mode in ("flight","airfetch"):
                if plan is not None:
                    jet_forces=flight.update(t,frames,com_velocity,root_angular,R,args.seconds,
                        target=com_setpoint(runtime["flight_target"],frames),cut=plan.cut)
                else:
                    jet_forces=flight.update(t,frames,com_velocity,root_angular,R,args.seconds)
                # Combine multiple jets on the same body into one wrench at its COM.
                # This prevents repeated API calls from replacing a prior force.
                wrenches={n:[np.zeros(3),np.zeros(3)] for n in set(j["body"] for j in THRUSTERS)}
                for jet,f_local in zip(THRUSTERS,jet_forces):
                    p,Rj=frames[jet["body"]]
                    f_world=Rj@f_local;offset=Rj@(np.asarray(jet["point"])-LINKS[jet["body"]]["com"])
                    wrenches[jet["body"]][0]+=f_world
                    wrenches[jet["body"]][1]+=np.cross(offset,f_world)
                for n,(f,moment) in wrenches.items():
                    p,Rj=frames[n];com=p+Rj@LINKS[n]["com"]
                    bodies.apply_forces_and_torques_at_pos(forces=f[None,:],torques=moment[None,:],positions=com[None,:],
                                                           indices=np.array([rows[n]],dtype=np.int32),is_global=True)
            tilt=float(np.degrees(np.arccos(np.clip(R[2,2],-1,1))))
            if frames["pelvis"][0][2]<.24 or tilt>55:
                runtime["fallen"]=True;runtime["stop"]=True
                print(f"Stopped: fall detected at {t:.3f}s; tilt={tilt:.1f} degrees.",flush=True)
                return
            if step%50==0:
                # Lowest point of the robot and of the grasping hand, so a run that
                # goes unstable while reaching can be told apart from one that simply
                # dragged its fingers across the table it was reaching over.
                lowest=min((p[2],n) for n,(p,R) in frames.items())
                hand_links=[n for n in frames if n.startswith(args.grasp_side)
                            and ("hand" in n or "finger" in n or "thumb" in n)]
                hand_low=min((frames[n][0][2],n) for n in hand_links) if hand_links else (None,None)
                sample=dict(t=t,root=frames["pelvis"][0].tolist(),q=robot.get_joint_positions()[indices].tolist(),
                            com=center_of_mass(frames).tolist(),tilt_deg=tilt,physics_dt=runtime["step_size"],
                            jet_norms=np.linalg.norm(jet_forces,axis=1).tolist(),
                            root_R=[round(float(v),5) for v in R.ravel()],
                            hand=frames[args.grasp_side+"_hand"][0].tolist(),
                            lowest=[round(lowest[0],4),lowest[1]],
                            hand_lowest=[None if hand_low[0] is None else round(hand_low[0],4),hand_low[1]],
                            jet_residual=[round(float(v),3) for v in flight.allocation_residual],
                            jet_forces=[[round(float(v),2) for v in f] for f in jet_forces])
                if vase_pose is not None:sample["vase"]=vase_pose[0].tolist()
                if mission is not None:sample["state"]=mission.state
                log.append(sample)
            if vase_pose is not None:
                runtime["vase"]=vase_pose[0];runtime["vase_rotation"]=vase_pose[1]
                # The build-time centre is computed from USD bounds before physics
                # runs; the body then drops a few centimetres onto the table. Take
                # the settled pose as the reference the lift is measured against,
                # or every lift reads about 50 mm short.
                if runtime["vase_settled"] is None and t>.5:
                    runtime["vase_settled"]=np.asarray(vase_pose[0],dtype=float).copy()
            runtime["frames"]=frames
            pelvis_row=rows["pelvis"]
            runtime["base_tf"]=(np.asarray(positions,dtype=float)[pelvis_row],np.asarray(quats,dtype=float)[pelvis_row])
            runtime["step"]=step+1
        except Exception:
            runtime["error"]=traceback.format_exc();runtime["stop"]=True

    world.add_physics_callback("irona_control",control)
    stalled=0
    while app.is_running() and not runtime["stop"]:
        if args.seconds>0 and runtime["time"]>=args.seconds:break
        before=runtime["step"]
        world.step(render=True)
        # A stage whose timeline stops (some referenced assets carry short time
        # ranges) leaves physics frozen while rendering continues. Say so instead
        # of spinning silently.
        stalled=stalled+1 if runtime["step"]==before else 0
        if stalled==120:
            print(f"Physics has not advanced for {stalled} frames; is the timeline playing? "
                  f"sim time {runtime['time']:.3f}s",flush=True)
        if bridge and runtime["base_tf"] is not None:
            bridge.update_base_tf(*runtime["base_tf"])
            if mission is not None and mission.estimate is not None:
                bridge.update_vase_tf(mission.estimate)
    world.remove_physics_callback("irona_control")
    if not app.is_running():print("Application exited; stopping the run.",flush=True)
    elif not world.is_playing():print("Timeline stopped playing; stopping the run.",flush=True)
    fallen=runtime["fallen"]
    print(f"Simulated {runtime['time']:.2f}s in {runtime['step']} control steps "
          f"(mean physics step {runtime['time']/max(runtime['step'],1)*1000:.2f} ms)",flush=True)
    if runtime["error"]:
        sys.stderr.write(runtime["error"]);sys.stderr.flush()
        raise RuntimeError("simulation control callback failed")
    log_path=args.log or ROOT/f"validation/isaac_{args.mode}_trajectory.json"
    log_path.parent.mkdir(parents=True,exist_ok=True)
    log_path.write_text(json.dumps(dict(mode=args.mode,fallen=fallen,physics_dt=DT,samples=log),indent=2))
    print(f"Saved {len(log)} samples to {log_path}",flush=True)
    if mission is not None:
        summary=mission.summary()
        summary["vase_rest"]=vase_rest.tolist()
        final=runtime["vase"]
        settled=runtime["vase_settled"]
        summary["vase_final"]=None if final is None else np.asarray(final).tolist()
        summary["vase_settled"]=None if settled is None else settled.tolist()
        # Measured against where the body actually came to rest, not the USD bounds
        # centre computed before physics ever ran.
        reference=vase_rest if settled is None else settled
        summary["vase_lift_m"]=None if final is None else round(float(final[2]-reference[2]),4)
        summary["vase_lift_vs_build_centre_m"]=(None if final is None
                                                else round(float(final[2]-vase_rest[2]),4))
        summary["vase_drift_before_contact_mm"]=(None if settled is None else
            round(float(np.linalg.norm(settled[:2]-np.asarray(vase_rest)[:2]))*1000,1))
        summary["grasp_assist"]=not args.no_grasp_assist
        summary["scene"]=args.scene
        summary["object"]=MODEL.describe()
        if mission.place_target is not None and final is not None:
            # Did the object end up where the robot meant to put it, standing up?
            target=np.asarray(mission.place_target,dtype=float)
            baseline=vase_rest if runtime["vase_settled"] is None else runtime["vase_settled"]
            rotation_final=runtime["vase_rotation"]
            tilt=None if rotation_final is None else round(float(np.degrees(
                np.arccos(np.clip(np.asarray(rotation_final)[2,2],-1,1)))),1)
            placed=dict(target=target.round(4).tolist(),
                        error_xy_mm=round(float(np.linalg.norm(np.asarray(final)[:2]-target[:2]))*1000,1),
                        height_vs_pickup_mm=round(float(np.asarray(final)[2]-baseline[2])*1000,1),
                        tilt_deg=tilt,
                        upright=None if tilt is None else bool(tilt<20.0),
                        moved_from_pickup_mm=round(float(np.linalg.norm(
                            np.asarray(final)[:2]-np.asarray(baseline)[:2]))*1000,1))
            placed["succeeded"]=bool(mission.released_at is not None and placed["error_xy_mm"]<60.0
                                     and placed["upright"] and abs(placed["height_vs_pickup_mm"])<40.0)
            summary["placed"]=placed
        if scene_report is not None:
            summary["scene_report"]={k:(v.tolist() if hasattr(v,"tolist") else v) for k,v in scene_report.items()}
        summary["fallen"]=fallen
        if plan is not None:
            summary["flight"]=plan.summary(runtime["frames"])
        report=ROOT/("validation/isaac_fetch_mission.json" if args.mode=="fetch"
                     else f"validation/isaac_{args.mode}_mission.json")
        report.write_text(json.dumps(summary,indent=2))
        if recorder is not None:
            index=recorder.close(summary)
            print(f"Dashboard: {len(recorder.index)} annotated frames -> {index}",flush=True)
        if live is not None:live.finish(summary)
        print(f"Mission state: {summary['state']}; camera detections {summary['detections']}/{summary['frames']} frames; "
              f"vase raised {summary['vase_lift_m']} m -> {report}",flush=True)
        if summary.get("flight") is not None:
            fl=summary["flight"]
            print(f"Flight: finished in phase {fl['phase']}; landed {fl['return_error_mm']} mm from "
                  f"the lift-off point; grasp station {fl['grasp_station']}",flush=True)
        if summary.get("placed") is not None:
            placed=summary["placed"]
            print(f"Placement: {'SUCCEEDED' if placed['succeeded'] else 'did not meet the check'}; "
                  f"{placed['error_xy_mm']} mm from the target, tilt {placed['tilt_deg']} deg, "
                  f"settled {placed['height_vs_pickup_mm']} mm vs the pickup height, "
                  f"carried {placed['moved_from_pickup_mm']} mm",flush=True)
except Exception:
    # SimulationApp.close() terminates the process, so a traceback that is left to
    # propagate past it is never printed. Report it here instead.
    traceback.print_exc(file=sys.stderr);sys.stderr.flush()
    failed=True
else:
    failed=False
finally:
    app.close()
    sys.exit(1 if failed else 0)
