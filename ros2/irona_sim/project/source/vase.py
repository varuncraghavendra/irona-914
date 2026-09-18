"""Graspable target object: a rigid vase standing on a static pedestal.

The colour is deliberately outside the arena palette so the head camera can
segment it by hue alone, without ground-truth pose. The geometry is a
two-cylinder proxy, not a scanned asset; mass and friction are demo values.
"""
import numpy as np

VASE_PATH='/World/Vase'
PEDESTAL_PATH='/World/VasePedestal'
MATERIAL_PATH='/World/VaseMaterial'
VASE_COLOR=(.93,.04,.55)      # magenta, hue ~326 deg; nothing else in the scene is close
BODY_RADIUS=.032
BODY_HEIGHT=.170
NECK_RADIUS=.020
NECK_HEIGHT=.045
MASS=.12
STANDOFF=.008
HUE_WINDOW=(295.,352.)        # magenta/cerise glaze; nothing else in the scene is close


class ObjectModel:
    """What the detector, the arm and the mission need to know about the target.

    The procedural vase and the scanned asset in house.py differ in size, so the
    scene builder fills this in once at start-up and everything downstream reads
    it instead of hard-coded numbers.
    """

    def __init__(self,radius=BODY_RADIUS,height=BODY_HEIGHT,grasp_fraction=.5,
                 standoff=STANDOFF,hue=HUE_WINDOW,name='vase'):
        self.radius=radius;self.height=height;self.grasp_fraction=grasp_fraction
        self.standoff=standoff;self.hue=hue;self.name=name

    @property
    def grasp_local(self):
        """Palm-relative grasp point: the vase axis stands this far ahead of the
        hand origin, level with the middle of the curled fingers. The standoff
        keeps a few millimetres of estimate error from pushing the vase over."""
        return np.array([.012+self.radius+self.standoff,0,-.050])

    def grasp_height(self,base_z):
        return float(base_z)+self.grasp_fraction*self.height

    def describe(self):
        return dict(name=self.name,radius_m=round(self.radius,4),height_m=round(self.height,4),
                    grasp_fraction=self.grasp_fraction,hue_window=list(self.hue))


MODEL=ObjectModel()
GRASP_LOCAL=MODEL.grasp_local   # kept for callers that only need the default shape


def _collision(prim,material=MATERIAL_PATH):
    from pxr import PhysxSchema,Sdf,UsdPhysics
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    physx=PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx.CreateContactOffsetAttr(.003);physx.CreateRestOffsetAttr(0.0)
    if material:prim.CreateRelationship('material:binding:physics',False).SetTargets([Sdf.Path(material)])


def add_vase(stage,position=(1.0,-.10),pedestal_height=.40,cylinders=True):
    """Add the pedestal and the vase; return the vase's resting centre in world.

    position is the (x,y) of the vase axis. The vase base rests on the pedestal top.
    """
    from pxr import Gf,PhysxSchema,UsdGeom,UsdPhysics,UsdShade
    x,y=float(position[0]),float(position[1])
    material=UsdShade.Material.Define(stage,MATERIAL_PATH).GetPrim()
    physics_material=UsdPhysics.MaterialAPI.Apply(material)
    physics_material.CreateStaticFrictionAttr(1.2)
    physics_material.CreateDynamicFrictionAttr(1.05)
    physics_material.CreateRestitutionAttr(0.0)

    pedestal=UsdGeom.Cube.Define(stage,PEDESTAL_PATH)
    pedestal.CreateSizeAttr(1)
    pedestal.AddTranslateOp().Set(Gf.Vec3d(x,y,pedestal_height/2))
    pedestal.AddScaleOp().Set(Gf.Vec3f(.14,.14,pedestal_height))
    pedestal.CreateDisplayColorAttr([Gf.Vec3f(.28,.29,.32)])
    _collision(pedestal.GetPrim(),material=None)

    centre=np.array([x,y,pedestal_height+BODY_HEIGHT/2])
    vase=UsdGeom.Xform.Define(stage,VASE_PATH)
    vase.AddTranslateOp().Set(Gf.Vec3d(*centre))
    vase.AddOrientOp().Set(Gf.Quatf(1,0,0,0))
    vase.AddScaleOp().Set(Gf.Vec3f(1,1,1))
    prim=vase.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
    mass_api=UsdPhysics.MassAPI.Apply(prim);mass_api.CreateMassAttr(MASS)
    radial=MASS*(3*BODY_RADIUS**2+BODY_HEIGHT**2)/12
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(radial,radial,MASS*BODY_RADIUS**2/2))
    body=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    body.CreateSolverPositionIterationCountAttr(32)
    body.CreateSolverVelocityIterationCountAttr(8)
    body.CreateSleepThresholdAttr(0.0)
    body.CreateMaxDepenetrationVelocityAttr(.5)

    for name,radius,height,offset in [('body',BODY_RADIUS,BODY_HEIGHT,0.0),
                                      ('neck',NECK_RADIUS,NECK_HEIGHT,(BODY_HEIGHT+NECK_HEIGHT)/2)]:
        if cylinders:
            geom=UsdGeom.Cylinder.Define(stage,VASE_PATH+'/'+name)
            geom.CreateRadiusAttr(radius);geom.CreateHeightAttr(height);geom.CreateAxisAttr('Z')
            geom.CreateExtentAttr([(-radius,-radius,-height/2),(radius,radius,height/2)])
        else:
            # Box fallback: PhysX represents cylinders with custom geometry, which a
            # given build can have disabled.
            geom=UsdGeom.Cube.Define(stage,VASE_PATH+'/'+name)
            geom.CreateSizeAttr(1);geom.AddScaleOp().Set(Gf.Vec3f(2*radius,2*radius,height))
        geom.AddTranslateOp().Set(Gf.Vec3d(0,0,offset))
        geom.CreateDisplayColorAttr([Gf.Vec3f(*VASE_COLOR)])
        _collision(geom.GetPrim())
    return centre
