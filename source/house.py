"""Furnished-room scene: NVIDIA residential assets, a real vase, real physics.

arena.py builds a perception test rig out of coloured blocks. This builds the
demo scene instead: an interior room shell, a coffee table, a scanned vase and
a plant from the Isaac Sim asset library, with collision and mass authored onto
the props the robot has to touch.

Assets stream from the Isaac Sim asset root over HTTPS, so the first run with a
cold Omniverse cache is slow.
"""
import time

import numpy as np

from vase import MODEL, MATERIAL_PATH

RESIDENTIAL = '/NVIDIA/Assets/ArchVis/Residential'
TABLE = RESIDENTIAL + '/Furniture/CoffeeTables/Midtown.usd'
VASE = RESIDENTIAL + '/Decor/Vases/BrassVase.usd'
PLANT = RESIDENTIAL + '/Plants/Plant_01.usd'
SOFA = RESIDENTIAL + '/Furniture/Sofas/Moline.usd'
RUG = RESIDENTIAL + '/Decor/Rugs/Rug_01.usd'
SHELF = RESIDENTIAL + '/Furniture/Bookshelves/Costello.usd'
PICTURE = RESIDENTIAL + '/Decor/Pictures/FramedPicture_Square02.usd'

ROOM_PATH = '/World/Room'
TABLE_PATH = '/World/CoffeeTable'
VASE_PATH = '/World/Vase'
PLANT_PATH = '/World/Plant'

# Room shell, in metres: the robot starts at the origin and walks along +X, so
# the lane from x=-1 to x=+1.3 at y=0 is kept clear of furniture.
ROOM_SIZE = (6.4, 5.0, 2.6)
ROOM_CENTRE = (1.2, 0.35)
WALL_THICKNESS = .10
FLOOR_COLOUR = (.40, .30, .21)
WALL_COLOUR = (.80, .78, .74)
TABLE_TOP = .362                 # Midtown coffee table, measured from the asset
VASE_GLAZE = (.86, .05, .52)     # ceramic glaze; see _tint for why this is not brass


def _settle(updates=3):
    """Pump the app so deferred stage work lands before anything is measured.

    Referencing a centimetre asset into a metre stage makes Isaac Sim's metrics
    assembler append an `xformOp:scale:unitsResolve` of 0.01 - but only on a later
    update. Measure a prop before that and the bounding box is 100x too large,
    while the render (which waits for it) shows a speck. Do not apply a unit
    conversion here as well; the assembler already owns that.
    """
    import omni.kit.app
    app = omni.kit.app.get_app()
    for _ in range(updates):
        app.update()


def _reference(stage, path, url, translate=(0, 0, 0), rotate_z=0.0, scale=1.0):
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, UsdGeom
    add_reference_to_stage(url, path)
    stage.Load(stage.GetPrimAtPath(path).GetPath())
    prim = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    prim.ClearXformOpOrder()
    prim.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translate]))
    if rotate_z:
        prim.AddRotateZOp().Set(float(rotate_z))
    prim.AddScaleOp().Set(Gf.Vec3f(float(scale), float(scale), float(scale)))
    _settle()
    return stage.GetPrimAtPath(path)


def _meshes(stage, path):
    from pxr import Usd, UsdGeom
    return [prim for prim in Usd.PrimRange(stage.GetPrimAtPath(path)) if prim.IsA(UsdGeom.Mesh)]


def _collide(stage, path, approximation='convexDecomposition', material=None):
    from pxr import PhysxSchema, Sdf, UsdPhysics
    meshes = _meshes(stage, path)
    for prim in meshes:
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
        physx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        physx.CreateContactOffsetAttr(.004)
        physx.CreateRestOffsetAttr(0.0)
        if material:
            prim.CreateRelationship('material:binding:physics', False).SetTargets([Sdf.Path(material)])
    return len(meshes)


def _bounds(stage, path):
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=True)
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.array([float(v) for v in box.GetMin()]), np.array([float(v) for v in box.GetMax()])


def _tint(stage, path, colour, name):
    """Bind one flat material over a referenced prop.

    The asset ships with a brass finish. A saturated ceramic glaze is just as
    real a vase and gives the colour detector something separable to work with
    in a furnished room, where brass reads the same as the wooden furniture.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    material = UsdShade.Material.Define(stage, '/World/Looks/' + name)
    shader = UsdShade.Shader.Define(stage, '/World/Looks/' + name + '/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.35)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    for prim in _meshes(stage, path):
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.UnbindAllBindings()
        binding.Bind(material, UsdShade.Tokens.strongerThanDescendants)
        UsdGeom.Mesh(prim).CreateDisplayColorAttr([Gf.Vec3f(*colour)])


def add_lighting(stage, height=2.45, area=(3.6, 3.6), intensity=2200.0):
    """Interior lighting.

    A dome light does nothing useful inside a closed room: the ceiling blocks it
    and the camera sees a black frame. These are ceiling panels over the walking
    lane plus a soft fill, which is what makes the vase readable to the detector.
    """
    from pxr import Gf, UsdLux
    panels = []
    for i, (x, y) in enumerate([(0.6, 0.0), (2.0, 0.0), (1.3, 1.6)]):
        light = UsdLux.RectLight.Define(stage, f'/World/HouseLights/Panel{i}')
        light.CreateWidthAttr(area[0]);light.CreateHeightAttr(area[1])
        light.CreateIntensityAttr(intensity if i < 2 else intensity * .5)
        light.CreateColorAttr(Gf.Vec3f(1.0, .97, .92))
        # A RectLight already emits along its own -Z, which is down here. Do not
        # add a 180 degree flip: that aims the panel at the ceiling.
        light.AddTranslateOp().Set(Gf.Vec3d(x, y, height))
        panels.append(light)
    fill = UsdLux.SphereLight.Define(stage, '/World/HouseLights/Fill')
    fill.CreateRadiusAttr(.35);fill.CreateIntensityAttr(intensity * .25)
    fill.AddTranslateOp().Set(Gf.Vec3d(.2, -1.1, 1.7))
    return panels


def wait_for_assets(timeout=180.0):
    """Block until every referenced asset has streamed in.

    The residential props come from the Isaac asset server over HTTPS. Rendering
    before they arrive gives a correct-looking USD stage and an empty picture.
    """
    import omni.kit.app
    from isaacsim.core.utils.stage import is_stage_loading
    app = omni.kit.app.get_app()
    start = time.time()
    while is_stage_loading() and time.time() - start < timeout:
        app.update()
    return time.time() - start


def add_room(stage, size=ROOM_SIZE, centre=ROOM_CENTRE, thickness=WALL_THICKNESS):
    """Floor and four walls, built as boxes so the walkable lane is known exactly.

    The shipped room environments put counters and fixtures across the middle of
    the floor, which leaves no predictable lane for a 914 mm robot to walk. This
    shell is plain, and every real prop is placed explicitly below.
    """
    from pxr import Gf, UsdGeom, UsdPhysics
    width, depth, height = size
    pieces = [('Floor', (centre[0], centre[1], -thickness / 2), (width, depth, thickness), FLOOR_COLOUR),
              ('WallFar', (centre[0] + width / 2, centre[1], height / 2), (thickness, depth, height), WALL_COLOUR),
              ('WallNear', (centre[0] - width / 2, centre[1], height / 2), (thickness, depth, height), WALL_COLOUR),
              ('WallLeft', (centre[0], centre[1] + depth / 2, height / 2), (width, thickness, height), WALL_COLOUR),
              ('WallRight', (centre[0], centre[1] - depth / 2, height / 2), (width, thickness, height), WALL_COLOUR)]
    UsdGeom.Xform.Define(stage, ROOM_PATH)
    for name, position, scale, colour in pieces:
        box = UsdGeom.Cube.Define(stage, f'{ROOM_PATH}/{name}')
        box.CreateSizeAttr(1)
        box.AddTranslateOp().Set(Gf.Vec3d(*position))
        box.AddScaleOp().Set(Gf.Vec3f(*scale))
        box.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
        UsdPhysics.CollisionAPI.Apply(box.GetPrim()).CreateCollisionEnabledAttr(True)
    return pieces


def build_house(stage, assets_root, table_position=(1.45, -.10), vase_scale=.72,
                vase_inset=.13, vase_mass=.15, plant=True, furniture=True):
    """Compose the living room and return the grasp target description.

    table_position is where the coffee table's centre goes. The vase stands
    vase_inset behind the table's near edge so the robot can reach it without
    walking its toes into the table.
    """
    from pxr import Gf, PhysxSchema, UsdPhysics, UsdShade

    physics_material = UsdShade.Material.Define(stage, MATERIAL_PATH).GetPrim()
    surface = UsdPhysics.MaterialAPI.Apply(physics_material)
    surface.CreateStaticFrictionAttr(1.1)
    surface.CreateDynamicFrictionAttr(.95)
    surface.CreateRestitutionAttr(0.0)

    add_room(stage)
    add_lighting(stage)

    _reference(stage, TABLE_PATH, _url(assets_root, TABLE),
               translate=(table_position[0], table_position[1], 0.0), rotate_z=90.0)
    _collide(stage, TABLE_PATH, approximation='convexDecomposition')
    wait_for_assets()
    _settle()
    table_low, table_high = _bounds(stage, TABLE_PATH)

    # Vase: dynamic rigid body with convex-hull collision on every mesh part.
    vase_x = table_low[0] + vase_inset
    _reference(stage, VASE_PATH, _url(assets_root, VASE),
               translate=(vase_x, table_position[1], table_high[2]), scale=vase_scale)
    prim = stage.GetPrimAtPath(VASE_PATH)
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(vase_mass)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    body.CreateSolverPositionIterationCountAttr(32)
    body.CreateSolverVelocityIterationCountAttr(8)
    # The scanned vase is a convex hull standing on an approximated table top, and
    # a hull whose underside is not quite flat rocks microscopically. With a zero
    # sleep threshold it never settles, and the rocking integrates into creep: the
    # vase walked 87 mm across the table during a run before the robot ever
    # touched it, which moves the target out from under the grasp. Damping plus a
    # threshold it can actually fall below lets it come to rest and stay there.
    body.CreateSleepThresholdAttr(.0005)
    body.CreateLinearDampingAttr(.60)
    body.CreateAngularDampingAttr(.80)
    body.CreateMaxDepenetrationVelocityAttr(.5)
    _collide(stage, VASE_PATH, approximation='convexHull', material=MATERIAL_PATH)
    _tint(stage, VASE_PATH, VASE_GLAZE, 'vase_glaze')
    wait_for_assets()
    _settle()
    low, high = _bounds(stage, VASE_PATH)
    # Drop the vase onto the measured table top: the placement above used the
    # table bounds, which the assembler may have adjusted since.
    from pxr import Gf as _Gf
    lift = float(table_high[2] - low[2])
    translate_op = [op for op in UsdGeomXformable(stage, VASE_PATH).GetOrderedXformOps()
                    if op.GetOpName() == 'xformOp:translate'][0]
    current = translate_op.Get()
    translate_op.Set(_Gf.Vec3d(current[0], current[1], current[2] + lift))
    _settle()
    low, high = _bounds(stage, VASE_PATH)
    MODEL.radius = float(max(high[0] - low[0], high[1] - low[1]) / 2)
    MODEL.height = float(high[2] - low[2])
    MODEL.name = 'vase'
    centre = np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2] + MODEL.height / 2])

    if plant:
        _reference(stage, PLANT_PATH, _url(assets_root, PLANT),
                   translate=(table_high[0] - .20, table_position[1] + .34, table_high[2]), scale=.55)
        _collide(stage, PLANT_PATH, approximation='convexHull')

    if furniture:
        # Everything here sits outside the x=-1..1.3, |y|<0.45 walking lane.
        _reference(stage, '/World/Rug', _url(assets_root, RUG), translate=(1.3, .25, .002), scale=.75)
        # The rug's carpet MDL is missing from the asset server, which renders it
        # flat red. A plain wool colour is closer to the intent and keeps the
        # detector's hue window clear.
        _tint(stage, '/World/Rug', (.30, .32, .36), 'rug_wool')
        _reference(stage, '/World/Sofa', _url(assets_root, SOFA), translate=(1.35, 1.75, 0.0), rotate_z=180.0)
        _collide(stage, '/World/Sofa', approximation='convexHull')
        _reference(stage, '/World/Shelf', _url(assets_root, SHELF), translate=(3.9, -1.2, 0.0), rotate_z=-90.0)
        _collide(stage, '/World/Shelf', approximation='convexHull')
        _reference(stage, '/World/Picture', _url(assets_root, PICTURE),
                   translate=(4.34, .6, 1.5), rotate_z=-90.0)

    waited = wait_for_assets()

    return dict(vase_path=VASE_PATH, centre=centre, base_z=float(low[2]), asset_wait_s=round(waited, 1),
                radius=MODEL.radius, height=MODEL.height,
                grasp_z=MODEL.grasp_height(float(low[2])),
                table_edge=float(table_low[0]), table_top=float(table_high[2]))


def UsdGeomXformable(stage, path):
    from pxr import UsdGeom
    return UsdGeom.Xformable(stage.GetPrimAtPath(path))


def _url(root, path):
    return root.rstrip('/') + path
