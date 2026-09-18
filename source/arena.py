"""Simple local geometry for observing depth and 360-degree lidar returns."""
from pxr import UsdGeom,UsdPhysics,Gf

def add_arena(stage):
    for name,pos,size,color in [
        ('Front',[2.4,0,.8],[.20,2.0,1.6],[.18,.48,.58]),
        ('Rear',[-2.0,0,.7],[.15,1.8,1.4],[.70,.33,.15]),
        ('Left',[0,2.0,.9],[1.8,.15,1.8],[.32,.52,.30]),
        ('Right',[0,-2.0,.6],[1.7,.15,1.2],[.50,.32,.60]),
        ('DepthTarget',[1.1,.45,.4],[.30,.30,.8],[.90,.62,.12])]:
        c=UsdGeom.Cube.Define(stage,'/World/PerceptionArena/'+name)
        c.CreateSizeAttr(1);c.AddTranslateOp().Set(Gf.Vec3d(*pos));c.AddScaleOp().Set(Gf.Vec3f(*size))
        c.CreateDisplayColorAttr([Gf.Vec3f(*color)]);UsdPhysics.CollisionAPI.Apply(c.GetPrim())
