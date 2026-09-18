"""Build editable CAD, printable parts and native USD from a common design.

Run with ordinary Python: pip install cadquery usd-core numpy trimesh
CAD/STL units are millimetres; USD and physics units are metres.
"""
from pathlib import Path
import json, csv, math
import numpy as np
import cadquery as cq
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdLux, Sdf, Gf, Vt
from model import ROOT, LINKS, JOINTS, THRUSTERS, HEIGHT, DT, configuration

OUT = ROOT
for d in ["assets", "cad", "print/stl", "print/display_armature", "config", "previews", "validation"]:
    (OUT/d).mkdir(parents=True, exist_ok=True)

COLORS = {
    "silver":(.58,.66,.71), "bright_silver":(.76,.81,.83),
    "graphite":(.060,.082,.105), "ivory":(.93,.92,.86),
    "copper":(.67,.17,.095), "black":(.013,.020,.025),
    "cyan":(.10,.72,.86), "rubber":(.07,.075,.085), "red":(.50,.07,.065),
}
PARTS = []

def box(size, center, radius=0):
    s = cq.Workplane("XY").box(*size)
    if radius:
        s = s.edges().fillet(radius)
    return s.val().translate(tuple(center))

def cylinder(radius, length, center, axis="Z"):
    direction = {"X":(1,0,0), "Y":(0,1,0), "Z":(0,0,1)}[axis]
    base = np.asarray(center)-np.asarray(direction)*length/2
    return cq.Solid.makeCylinder(radius, length, tuple(base), direction)

def sphere(radius, center):
    return cq.Workplane("XY").sphere(radius).val().translate(center)

def shell_box(size, center, thickness=2.4, radius=10):
    return box(size,center,radius).cut(box(tuple(s-2*thickness for s in size),center,max(1,radius-thickness)))

def tube(radius,length,center,thickness=2.4,axis="Z"):
    return cylinder(radius,length,center,axis).cut(cylinder(radius-thickness,length+2,center,axis))

def loft_ellipse(sections):
    w = cq.Workplane("XY").workplane(offset=sections[0][0]).ellipse(sections[0][1],sections[0][2])
    for a,b in zip(sections,sections[1:]):
        w = w.workplane(offset=b[0]-a[0]).ellipse(b[1],b[2])
    return w.loft(combine=True).val()

def add(name, body, shape, material="silver", split=None, role="cosmetic exterior"):
    name=name.replace("-","m")
    assert shape.isValid(), name
    PARTS.append(dict(name=name,body=body,shape=shape,material=material,split=split,role=role))

def limb_shell(name,body,z0,z1,y,r0,r1):
    # Axial sleeve with open ends; no actuator attachment is implied.
    outer = cq.Solid.makeCone(r0,r1,z1-z0,(0,y,z0))
    inner = cq.Solid.makeCone(r0-2.4,r1-2.4,z1-z0+2,(0,y,z0-1))
    add(name,body,outer.cut(inner),split="X")

def make_geometry():
    add("pelvis_shell","pelvis",shell_box((124,198,76),(0,0,503),radius=20),"graphite",split="X")
    outer = loft_ellipse([(466,81,119),(494,73,107),(538,59,87)])
    inner = loft_ellipse([(465,78.6,116.6),(494,70.6,104.6),(539,56.6,84.6)])
    # The skirt is split into front/back and left/right panels for printability.
    add("skirt","pelvis",outer.cut(inner),"graphite",split="XY")
    add("skirt_apron","pelvis",box((3,145,64),(78,0,508),7 if False else 1.4),"ivory")
    add("waist_band","pelvis",tube(64,12,(0,0,538)),"bright_silver",split="X")

    outer = loft_ellipse([(549,59,85),(608,70,100),(685,76,113),(714,61,96)])
    inner = loft_ellipse([(548,56.6,82.6),(608,67.6,97.6),(685,73.6,110.6),(715,58.6,93.6)])
    add("torso_shell","torso",outer.cut(inner),"graphite",split="XY")
    apron_outer=loft_ellipse([(549,62,88),(608,73,103),(685,79,116),(714,64,99)])
    apron=apron_outer.cut(outer).intersect(box((120,130,111),(90,0,631)))
    add("chest_apron","torso",apron,"ivory")
    for sign in [-1,1]:
        collar = box((10,57,28),(62,sign*35,699),4).rotate((62,sign*35,699),(63,sign*35,699),-sign*19)
        add(f"collar_{sign+1}","torso",collar,"ivory")
    add("chest_badge","torso",cylinder(13,3,(80,0,663),"X"),"copper")
    add("badge_center","torso",cylinder(7,3,(82,0,663),"X"),"bright_silver")
    for z in [638,624,610]:
        add(f"apron_button_{z}","torso",sphere(3.8,(79,0,z)),"graphite")
    add("neck_sleeve","neck_yaw_link",tube(19,26,(0,0,747)),"silver",split="X")
    add("neck_base","torso",tube(20,30,(0,0,728),4),"graphite",split="X")

    # Neutral sensor head; no eyes, mouth, teeth, nose, hair or facial expression.
    add("head_shell","head",shell_box((116,143,108),(0,0,820),radius=20),"bright_silver",split="X")
    add("camera_mount_bezel","head",shell_box((13,138,40),(61,0,826),thickness=2.4,radius=5),"graphite",split="Y")
    # D455 nominal envelope: 124 wide x 26 high x 29 deep millimetres.
    add("d455_housing","head",shell_box((29,124,26),(66,0,826),thickness=2.0,radius=5),"graphite",split="Y",role="D455 envelope mock-up; remove when fitting a real camera")
    add("d455_front_panel","head",box((2,120,22),(81,0,826),.8),"black")
    for i,y in enumerate([-47.5,-16,15,47.5]):
        add(f"camera_optical_window_{i}","head",cylinder(4.0,1,(82.5,y,826),"X"),"graphite",role="optical window mock-up; not a printable lens")
    for sign in [-1,1]:
        add(f"head_service_plate_{sign}","head",box((62,2.6,54),(-7,sign*71.8,817),1),"graphite")
        for zz in [801,834]:
            add(f"head_fastener_{sign}_{zz}","head",cylinder(2.8,2,(-25,sign*74,zz),"Y"),"silver")
    # Optical band at 898.4 mm has no opaque geometry in its +/-15 degree scan cone.
    add("lidar_pedestal","head",cylinder(18,12,(-6,0,879)),"graphite")
    add("lidar_base","head",cylinder(32,5,(-6,0,886)),"graphite")
    add("lidar_upper_cap","head",cylinder(31,6,(-6,0,911.4)),"graphite",role="lidar housing mock-up; cap needs a transparent support for display")
    add("lidar_cap_trim","head",tube(31.2,2,(-6,0,910),1.2),"copper")

    for side,sign in [("left",1),("right",-1)]:
        y=sign*75
        limb_shell(side+"_thigh_shell",side+"_thigh",283,438,y,34,39)
        limb_shell(side+"_shin_shell",side+"_shin",88,246,y,30,34)
        for label,z,body in [("hip",460,side+"_thigh"),("knee",265,side+"_shin"),("ankle",70,side+"_foot")]:
            add(side+"_"+label+"_hub",body,tube(21,58,(0,y,z),5,"Y"),"graphite",split="Y",role="visual joint cover; not a bearing")
            add(side+"_"+label+"_disc",body,cylinder(14,4,(0,y+sign*32,z),"Y"),"bright_silver")
            add(side+"_"+label+"_bolt",body,cylinder(5.5,5,(0,y+sign*35,z),"Y"),"graphite")
        outlet=cylinder(17,30,(22.5,y,5))
        boot=shell_box((170,92,67),(22.5,y,33.5),radius=14).cut(outlet)
        # Sole and boot are a hollow exterior; the simulator uses the flat box proxy.
        add(side+"_boot_shell",side+"_foot",boot,"silver",split="X")
        add(side+"_sole",side+"_foot",box((163,87,6),(22.5,y,3),2.5).cut(outlet),"rubber")
        for dz in [123,169,215]:
            add(side+f"_shin_band_{dz}",side+"_shin",tube(34,7,(0,y,dz)),"bright_silver",split="X")
        # Decorative jet outlets. Actual force acts at the named USD jet frame.
        add(side+"_boot_nozzle",side+"_foot",tube(17,13,(22.5,y,13),3.5),"graphite")
        add(side+"_boot_nozzle_ring",side+"_foot",tube(19,4,(22.5,y,9),3),"copper")

        sy=sign*197
        add(side+"_shoulder_mount", "torso", box((24,68,25),(-12,sign*143,703),4),"graphite",role="shoulder support concept; not a qualified motor bracket")
        # Continuous rear yoke rails connect neighboring pivot supports. Each
        # belongs to the upstream rigid link and moves with that link.
        for label in ["shoulder_roll","shoulder_yaw","elbow","forearm_roll","wrist_pitch","wrist_roll"]:
            jj=next(j for j in JOINTS if j["name"]==side+"_"+label)
            a=np.array(LINKS[jj["parent"]]["position"])*1000+[-13.4,0,0]
            b=np.array(jj["anchor"])*1000+[-13.4,0,0]
            delta=b-a;length=float(np.linalg.norm(delta));direction=tuple(delta/length)
            rail=cq.Solid.makeCylinder(5.1,length,tuple(a),direction)
            rail=rail.fuse(sphere(5.1,tuple(a)),sphere(5.1,tuple(b)))
            add(jj["name"]+"_yoke_rail",jj["parent"],rail,"graphite",role="joint-support rail concept")
        add(side+"_thumb_support",side+"_hand",box((15,20,13),(0,sy-sign*37,396),3),"graphite")
        # Exposed paired bearing races, shaft bores and fasteners at each actual arm axis.
        arm_joint_names=["shoulder_pitch","shoulder_roll","shoulder_yaw","elbow","forearm_roll","wrist_pitch","wrist_roll"]
        for label in arm_joint_names:
            j=next(j for j in JOINTS if j["name"]==side+"_"+label)
            center=np.array(j["anchor"])*1000;axis=j["axis"]
            r=14 if label.startswith("shoulder") else 10
            width=28 if label.startswith("shoulder") else 22
            direction=np.eye(3)["XYZ".index(axis)]
            # A visible through shaft belongs to the rotating child; support races to parent.
            add(j["name"]+"_shaft",j["child"],cylinder(3.0,width+8,center,axis),"silver",role="joint axle mock-up; use metal shaft on hardware")
            for sign2 in [-1,1]:
                c=center+direction*sign2*(width/2)
                add(j["name"]+f"_bearing_{sign2}",j["parent"],tube(r,4,c,4,axis),"graphite",role="bearing race visual; substitute a purchased bearing")
                add(j["name"]+f"_horn_{sign2}",j["child"],tube(r-3,2.5,c+direction*sign2*3.5,3,axis),"bright_silver")
                u=np.eye(3)[("XYZ".index(axis)+1)%3];v=np.cross(direction,u)
                for k in range(4):
                    pos=c+direction*sign2*5.5+(r-4.5)*(u*np.cos(k*np.pi/2)+v*np.sin(k*np.pi/2))
                    add(j["name"]+f"_screw_{sign2}_{k}",j["child"],cylinder(1.5,1.5,pos,axis),"black",role="fastener mock-up")
        # Open channels allow visual inspection of joints and motor envelopes.
        add(side+"_upper_arm_backbone",side+"_upper_arm",box((8,39,77),(-14,sy,607),2),"graphite")
        add(side+"_upper_arm_cover",side+"_upper_arm",shell_box((42,47,68),(0,sy,611),radius=6),"silver",split="X")
        add(side+"_forearm_cover",side+"_forearm_roll_link",shell_box((47,58,61),(0,sy,479),radius=7),"silver",split="X")
        add(side+"_forearm_tendon_panel",side+"_forearm_roll_link",box((2,35,42),(24,sy,479),.8),"graphite")
        for yy in [-12,0,12]:
            add(side+f"_tendon_port_{yy+12}",side+"_forearm_roll_link",tube(2.5,5,(0,sy+yy,447),1),"copper")
        add(side+"_palm",side+"_hand",shell_box((25,66,38),(0,sy,396),thickness=2.0,radius=5),"silver",split="X")
        add(side+"_palm_pad",side+"_hand",box((3,53,28),(13,sy,396),1),"rubber")
        for j in JOINTS:
            if not j["name"].startswith(side+"_") or not any(k in j["name"] for k in ["_index_","_middle_","_ring_","_little_","_thumb_"]):continue
            body=LINKS[j["child"]];p=np.array(body["position"])*1000;c=p+np.array(body["com"])*1000
            if "opposition" in j["name"]:
                add(j["name"]+"_mount",j["child"],box((16,15,10),c,2),"graphite")
                continue
            size=np.array(body["size"])*1000
            sh=box(tuple(size),c,2).cut(cylinder(1.2,30,p+np.array([0,0,-4]),"Y"))
            add(j["name"]+"_phalanx",j["child"],sh,"bright_silver",role="articulated finger segment mock-up with 2.4 mm pilot bore")
            add(j["name"]+"_pad",j["child"],box((2.2,size[1]-3,max(4,size[2]-6)),c+np.array([size[0]/2+.6,0,0]),.8),"rubber")
            add(j["name"]+"_knuckle",j["child"],tube(4.2,size[1]+1,p,2.9,j["axis"]),"graphite")

        jetcenter=(-93,sign*90,635)
        add(side+"_back_jet", "torso",tube(28,86,jetcenter,3),"silver",split="X")
        add(side+"_back_nozzle", "torso",tube(24,20,(-93,sign*90,592),3),"graphite")
        add(side+"_back_nozzle_ring", "torso",tube(26,5,(-93,sign*90,584),3),"copper")
        add(side+"_jet_mount", "torso",box((38,14,16),(-70,sign*90,654),3),"graphite")

def split_shape(shape, axes):
    shapes=[shape]
    # Cosmetic seam at the world symmetry plane, or part bounding-box midpoint.
    for axis in axes or "":
        ai="XYZ".index(axis)
        more=[]
        for sh in shapes:
            b=sh.BoundingBox()
            mins=np.array([b.xmin,b.ymin,b.zmin]); maxs=np.array([b.xmax,b.ymax,b.zmax])
            mid=(mins[ai]+maxs[ai])/2
            for sign in [-1,1]:
                center=(mins+maxs)/2
                center[ai]=mid+sign*1000
                cut=sh.intersect(box((2000,2000,2000),tuple(center)))
                more.extend(s for s in cut.Solids() if s.Volume()>.05)
        shapes=more
    # Guaranteed maximum 200 mm axis-aligned part envelope.
    pending=shapes[:]; result=[]
    while pending:
        sh=pending.pop(0); b=sh.BoundingBox()
        dims=np.array([b.xlen,b.ylen,b.zlen])
        if dims.max()<=200.00001:
            result.append(sh); continue
        axis="XYZ"[int(dims.argmax())]
        pending.extend(split_shape(sh,axis))
    return result

def mesh_arrays(shape):
    verts,faces=shape.tessellate(.35,.15)
    return np.array([v.toTuple() for v in verts]),np.array(faces,dtype=int)

def make_material(stage,name,rgb):
    mat=UsdShade.Material.Define(stage,"/Irona/Looks/"+name)
    shader=UsdShade.Shader.Define(stage,str(mat.GetPath())+"/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.36 if "silver" in name else .55)
    shader.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(.70 if "silver" in name else .12)
    shader.CreateOutput("surface",Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),"surface")
    return mat

def mesh_prim(stage,path,verts,faces,material):
    m=UsdGeom.Mesh.Define(stage,path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    m.CreateFaceVertexCountsAttr([3]*len(faces))
    m.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateExtentAttr([Gf.Vec3f(*verts.min(axis=0)),Gf.Vec3f(*verts.max(axis=0))])
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m

def attr(prim,name,typ,value):
    prim.CreateAttribute(name,typ,custom=False).Set(value)

def build_usd_and_prints():
    for old in (OUT/"print/stl").glob("*.stl"):
        old.unlink()
    stage=Usd.Stage.CreateNew(str(OUT/"assets/irona.usda"))
    root=UsdGeom.Xform.Define(stage,"/Irona")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1)
    UsdGeom.SetStageUpAxis(stage,"Z")
    stage.SetMetadata("documentation","Irona-inspired 914.4 mm research prototype. Masses assumed. See README.md for validation scope.")
    UsdGeom.Scope.Define(stage,"/Irona/Looks")
    mats={k:make_material(stage,k,v) for k,v in COLORS.items()}
    ground_mat=UsdShade.Material.Define(stage,"/Irona/Looks/contact")
    phys=UsdPhysics.MaterialAPI.Apply(ground_mat.GetPrim())
    phys.CreateStaticFrictionAttr(.9);phys.CreateDynamicFrictionAttr(.8);phys.CreateRestitutionAttr(0)
    UsdGeom.Scope.Define(stage,"/Irona/Links")
    for name,body in LINKS.items():
        xf=UsdGeom.Xform.Define(stage,"/Irona/Links/"+name)
        xf.AddTranslateOp().Set(Gf.Vec3d(*body["position"]))
        p=xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(p).CreateRigidBodyEnabledAttr(True)
        mass=UsdPhysics.MassAPI.Apply(p)
        mass.CreateMassAttr(body["mass"])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*body["com"]))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*body["inertia"]))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1,0,0,0))
        if name=="pelvis":
            UsdPhysics.ArticulationRootAPI.Apply(p)
            p.AddAppliedSchema("PhysxArticulationAPI")
            attr(p,"physxArticulation:enabledSelfCollisions",Sdf.ValueTypeNames.Bool,True)
            attr(p,"physxArticulation:solverPositionIterationCount",Sdf.ValueTypeNames.Int,32)
            attr(p,"physxArticulation:solverVelocityIterationCount",Sdf.ValueTypeNames.Int,8)
        if body["collision"]:
            c=UsdGeom.Cube.Define(stage,str(p.GetPath())+"/collision")
            c.CreateSizeAttr(1)
            c.AddTranslateOp().Set(Gf.Vec3d(*body["com"]))
            c.AddScaleOp().Set(Gf.Vec3f(*body["size"]))
            c.CreateVisibilityAttr("invisible")
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())
            c.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
            attr(c.GetPrim(),"physxCollision:contactOffset",Sdf.ValueTypeNames.Float,.002)
            attr(c.GetPrim(),"physxCollision:restOffset",Sdf.ValueTypeNames.Float,0.0)
            UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(ground_mat,materialPurpose="physics")
    # Filter close relatives, including zero-length multi-axis actuator carriers.
    graph={n:set() for n in LINKS}
    for j in JOINTS:
        graph[j["parent"]].add(j["child"]); graph[j["child"]].add(j["parent"])
    for n in LINKS:
        seen={n}; frontier={n}
        for _ in range(3):
            frontier=set().union(*(graph[x] for x in frontier))-seen
            seen|=frontier
        targets=[Sdf.Path("/Irona/Links/"+x) for x in sorted(seen-{n})]
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/Irona/Links/"+n)).CreateFilteredPairsRel().SetTargets(targets)
    UsdGeom.Scope.Define(stage,"/Irona/Joints")
    for j in JOINTS:
        cls=UsdPhysics.PrismaticJoint if j["kind"]=="prismatic" else UsdPhysics.RevoluteJoint
        obj=cls.Define(stage,"/Irona/Joints/"+j["name"])
        obj.CreateBody0Rel().SetTargets(["/Irona/Links/"+j["parent"]])
        obj.CreateBody1Rel().SetTargets(["/Irona/Links/"+j["child"]])
        anchor=np.asarray(j["anchor"])
        obj.CreateLocalPos0Attr(Gf.Vec3f(*(anchor-LINKS[j["parent"]]["position"])))
        obj.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
        obj.CreateLocalRot0Attr(Gf.Quatf(1,0,0,0));obj.CreateLocalRot1Attr(Gf.Quatf(1,0,0,0))
        obj.CreateAxisAttr(j["axis"])
        obj.CreateLowerLimitAttr(j["limits"][0]);obj.CreateUpperLimitAttr(j["limits"][1])
        obj.CreateCollisionEnabledAttr(False)
        dtype="linear" if j["kind"]=="prismatic" else "angular"
        drive=UsdPhysics.DriveAPI.Apply(obj.GetPrim(),dtype)
        drive.CreateTypeAttr("force")
        # USD angular stiffness/damping use degrees; controller uses Nm/rad.
        factor=1 if dtype=="linear" else math.pi/180
        drive.CreateStiffnessAttr(j["kp"]*factor)
        drive.CreateDampingAttr(j["kd"]*factor)
        drive.CreateMaxForceAttr(j["effort"])
        drive.CreateTargetPositionAttr(0)
        obj.GetPrim().AddAppliedSchema("PhysxJointAPI")
        attr(obj.GetPrim(),"physxJoint:jointFriction",Sdf.ValueTypeNames.Float,j["friction"])
        obj.GetPrim().CreateAttribute("irona:actuator",Sdf.ValueTypeNames.String).Set(j["actuator"])
        attr(obj.GetPrim(),"physxJoint:maxJointVelocity",Sdf.ValueTypeNames.Float,
             j["velocity"] if dtype=="linear" else math.degrees(j["velocity"]))
    for jet in THRUSTERS:
        p=UsdGeom.Xform.Define(stage,"/Irona/Links/"+jet["body"]+"/"+jet["name"])
        p.AddTranslateOp().Set(Gf.Vec3d(*jet["point"]))
        p.GetPrim().CreateAttribute("irona:maxThrustN",Sdf.ValueTypeNames.Float).Set(jet["max_force"])
        p.GetPrim().CreateAttribute("irona:jetDirection",Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(0,0,1))
        p.GetPrim().CreateAttribute("irona:gimbalConeDeg",Sdf.ValueTypeNames.Float).Set(25)

    from sensors import author_sensors
    author_sensors(stage)
    print_rows=[]; render_parts=[]; assembly=cq.Assembly(name="Irona_914_exterior_mm")
    occupied={n:[] for n in LINKS}
    for idx,part in enumerate(PARTS):
        n,body,sh,col=part["name"],part["body"],part["shape"],part["material"]
        verts,faces=mesh_arrays(sh)
        local=verts*.001-np.array(LINKS[body]["position"])
        UsdGeom.Scope.Define(stage,"/Irona/Links/"+body+"/Visuals")
        mesh_prim(stage,"/Irona/Links/"+body+"/Visuals/"+n,local,faces,mats[col])
        render_parts.append(dict(name=n,body=body,material=col,vertices=verts.tolist(),faces=faces.tolist()))
        # Trim buried decorative material to make real mating seats on each link.
        # The visible union is identical, but separate prints no longer interpenetrate.
        bbox=sh.BoundingBox()
        def overlaps(other):
            b=other.BoundingBox()
            return all([bbox.xmax>b.xmin,b.xmax>bbox.xmin,bbox.ymax>b.ymin,b.ymax>bbox.ymin,bbox.zmax>b.zmin,b.zmax>bbox.zmin])
        prior=[x for x in occupied[body] if overlaps(x)]
        seated=sh.cut(*prior) if prior else sh
        occupied[body].append(sh)
        if seated.Volume()<.05:
            continue
        assert seated.isValid(),n+" mating seat"
        assembly.add(seated,name=n,color=cq.Color(*COLORS[col]))
        for k,piece in enumerate(split_shape(seated,part["split"])):
            b=piece.BoundingBox()
            offset=np.array([b.xmin,b.ymin,b.zmin])
            normalized=piece.translate(tuple(-offset))
            filename=f"{idx+1:03d}_{n}_{k+1:02d}.stl"
            cq.exporters.export(normalized,str(OUT/"print/stl"/filename),tolerance=.12,angularTolerance=.15)
            print_rows.append(dict(file=filename,assembly=body,part=n,color=col,role=part["role"],
                                   size_x_mm=round(b.xlen,3),size_y_mm=round(b.ylen,3),size_z_mm=round(b.zlen,3),
                                   assembly_tx_mm=round(offset[0],6),assembly_ty_mm=round(offset[1],6),assembly_tz_mm=round(offset[2],6),
                                   volume_cm3=round(piece.Volume()/1000,4),units="mm"))
        if idx%20==0: print("Geometry exported:",idx+1,"/",len(PARTS),flush=True)
    assembly.save(str(OUT/"cad/irona_exterior.step"))
    stage.GetRootLayer().Save()
    stage.GetRootLayer().Export(str(OUT/"assets/irona.usd"))
    with (OUT/"print/parts.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(print_rows[0]));w.writeheader();w.writerows(print_rows)
    (OUT/"config/robot.json").write_text(json.dumps(configuration(),indent=2))
    (OUT/"config/visual_meshes.json").write_text(json.dumps(render_parts,separators=(",",":")))
    (OUT/"print/assembly_transforms.json").write_text(json.dumps(print_rows,indent=2))
    print("Exterior STL files:",len(print_rows),flush=True)

def build_display_armature():
    """Static neutral-pose display support. Not a powered robot frame."""
    for f in (OUT/"print/display_armature").glob("display_core_*.stl"):f.unlink()
    rods=[cylinder(8,340,(0,0,650)),cylinder(8,150,(0,0,480),"Y"),cylinder(8,344,(0,0,703),"Y")]
    for sign in [-1,1]:
        rods += [cylinder(8,454,(0,sign*75,253)),cylinder(7,294,(0,sign*197,546)),cylinder(7,25,(0,sign*184.5,703),"Y")]
        for y,z in [(sign*75,480),(sign*75,460),(sign*75,265),(sign*75,70),(sign*197,703),(sign*197,553),(sign*197,417)]:
            rods.append(sphere(13,(0,y,z)))
        rods.append(box((60,42,5),(15,sign*75,27.5),2))
    full=rods[0].fuse(*rods[1:]).clean()
    cq.exporters.export(full,str(OUT/"cad/static_display_armature.step"))
    rows=[]
    for i,s in enumerate(split_shape(full,"YZ")):
        b=s.BoundingBox(); offset=(b.xmin,b.ymin,b.zmin)
        fn=f"display_core_{i+1:02d}.stl"
        cq.exporters.export(s.translate(tuple(-x for x in offset)),str(OUT/"print/display_armature"/fn))
        rows.append(dict(file=fn,translation_mm=offset,size_mm=(b.xlen,b.ylen,b.zlen)))
    # Glue-on seam reinforcement strips, printed independently of the exterior.
    strap=box((40,12,2.4),(20,6,1.2),1)
    cq.exporters.export(strap,str(OUT/"print/display_armature/seam_strap_print_as_needed.stl"))
    sleeve=tube(11,36,(0,0,18),thickness=2.8)
    cq.exporters.export(sleeve,str(OUT/"print/display_armature/16mm_core_sleeve_print_as_needed.stl"))
    (OUT/"print/display_armature/transforms.json").write_text(json.dumps(rows,indent=2))

def build_scene():
    stage=Usd.Stage.CreateNew(str(OUT/"assets/scene.usda"))
    world=UsdGeom.Xform.Define(stage,"/World");stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,"Z")
    robot=UsdGeom.Xform.Define(stage,"/World/Irona")
    robot.GetPrim().GetReferences().AddReference("irona.usd")
    scene=UsdPhysics.Scene.Define(stage,"/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
    scene.GetPrim().AddAppliedSchema("PhysxSceneAPI")
    attr(scene.GetPrim(),"physxScene:timeStepsPerSecond",Sdf.ValueTypeNames.UInt,int(round(1/DT)))
    attr(scene.GetPrim(),"physxScene:solverType",Sdf.ValueTypeNames.Token,"TGS")
    attr(scene.GetPrim(),"physxScene:enableGPUDynamics",Sdf.ValueTypeNames.Bool,False)
    ground=UsdGeom.Cube.Define(stage,"/World/Ground")
    ground.CreateSizeAttr(1);ground.AddTranslateOp().Set(Gf.Vec3d(0,0,-.05));ground.AddScaleOp().Set(Gf.Vec3f(20,20,.1))
    ground.CreateDisplayColorAttr([Gf.Vec3f(.16,.19,.22)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    material=UsdShade.Material.Define(stage,"/World/GroundMaterial")
    m=UsdPhysics.MaterialAPI.Apply(material.GetPrim());m.CreateStaticFrictionAttr(.9);m.CreateDynamicFrictionAttr(.8);m.CreateRestitutionAttr(0)
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material,materialPurpose="physics")
    light=UsdLux.DomeLight.Define(stage,"/World/Sky");light.CreateIntensityAttr(700)
    sun=UsdLux.DistantLight.Define(stage,"/World/Key");sun.CreateIntensityAttr(1800)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-30,-30,25));sun.CreateAngleAttr(10)
    camera=UsdGeom.Camera.Define(stage,"/World/Camera")
    eye=Gf.Vec3d(1.6,-1.8,1.05);target=Gf.Vec3d(0,0,.47)
    view=Gf.Matrix4d().SetLookAt(eye,target,Gf.Vec3d(0,0,1)).GetInverse()
    camera.AddTransformOp().Set(view);camera.CreateFocalLengthAttr(55)
    from arena import add_arena
    add_arena(stage)
    stage.GetRootLayer().Save()

if __name__=="__main__":
    make_geometry()
    build_usd_and_prints()
    build_display_armature()
    from clean_print_meshes import clean
    clean()
    build_scene()
    print("Assets complete.",flush=True)
