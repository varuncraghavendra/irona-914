"""Cross-format contract checks for the hands + perception revision."""
import ast,json,math,xml.etree.ElementTree as ET
import numpy as np
from pxr import Usd,UsdGeom,Gf
from model import ROOT,LINKS,JOINTS,NAMES,INDEX,HEIGHT,forward_kinematics
from sensors import CAMERAS,LIDAR,OPTICAL_R,sensor_frames
from control import HOME,hand_pose,LO,HI,TargetLimiter,MAX_VEL
from ros_bridge import graph_specs

def main():
    report={'runtime_verified':{'isaac_sim':False,'ros2':False,'rviz':False},'checks':{}}
    stage=Usd.Stage.Open(str(ROOT/'assets/irona.usd'));assert stage
    scene=Usd.Stage.Open(str(ROOT/'assets/scene_ros.usda'));assert not scene.GetCompositionErrors()
    allnames=[p.GetName() for p in stage.Traverse()]
    assert not any(any(key in n for key in ['eye_socket','eye_white','eye_pupil','eyebrow','smile','mouth_tooth','hair_roll']) for n in allnames)
    report['checks']['facial_features_removed']=True
    assert len(JOINTS)==62 and len(LINKS)==63
    for side in ['left','right']:
        for finger in ['index','middle','ring','little']:
            assert len([j for j in JOINTS if j['name'].startswith(f'{side}_{finger}_')])==3
        assert len([j for j in JOINTS if j['name'].startswith(f'{side}_thumb_')])==4
    report['checks']['five_fingers_per_hand_16_dof_each']=True
    for closure in np.linspace(0,1,11):
        q=hand_pose(HOME,closure);assert np.all(q>=LO-1e-9) and np.all(q<=HI+1e-9)
    limiter=TargetLimiter(.001);old=limiter.q.copy();new=limiter.update(hand_pose(HOME,1))
    assert np.all(np.abs(new-old)<=MAX_VEL*.001+1e-12)
    report['checks']['hand_targets_and_velocity_limits']=True
    # USD camera's world forward/right/down must match ROS optical basis exactly.
    cache=UsdGeom.XformCache()
    for name,cfg in CAMERAS.items():
        p=stage.GetPrimAtPath('/Irona/Links/head/Sensors/'+name);c=UsdGeom.Camera(p);assert c
        M=cache.GetLocalToWorldTransform(p)
        for usd_axis,ros_axis in [([0,0,-1],[0,0,1]),([1,0,0],[1,0,0]),([0,-1,0],[0,1,0])]:
            assert np.allclose(M.TransformDir(Gf.Vec3d(*usd_axis)),OPTICAL_R@ros_axis,atol=1e-6)
        assert np.allclose(np.array(M.ExtractTranslation()),np.array(LINKS['head']['position'])+cfg['position'])
        for aperture,angle in [(c.GetHorizontalApertureAttr().Get(),cfg['fov_deg'][0]),(c.GetVerticalApertureAttr().Get(),cfg['fov_deg'][1])]:
            assert abs(np.degrees(2*np.arctan(aperture/(2*c.GetFocalLengthAttr().Get())))-angle)<1e-5
    report['checks']['camera_usd_and_ros_optical_axes_and_fov']=True
    p=stage.GetPrimAtPath('/Irona/Links/head/Sensors/lidar');assert p.GetTypeName()=='OmniLidar'
    assert p.GetAttribute('omni:sensor:Core:validEndAzimuthDeg').Get()==360
    assert p.GetAttribute('omni:sensor:tickRate').Get()==10
    az=p.GetAttribute('omni:sensor:Core:emitterState:s001:azimuthDeg').Get();el=p.GetAttribute('omni:sensor:Core:emitterState:s001:elevationDeg').Get()
    assert len(az)==len(el)==32 and min(el)==-15 and max(el)==15
    report['checks']['native_lidar_32_channels_360_by_30_deg']=True
    urdf=ET.parse(ROOT/'ros2/irona_sim/urdf/irona.urdf').getroot()
    js={j.get('name'):j for j in urdf.findall('joint')};ls={l.get('name') for l in urdf.findall('link')}
    for j in JOINTS:
        u=js[j['name']];assert u.find('parent').get('link')==j['parent'] and u.find('child').get('link')==j['child']
        expected=np.array(LINKS[j['child']]['position'])-LINKS[j['parent']]['position']
        assert np.allclose(np.fromstring(u.find('origin').get('xyz'),sep=' '),expected)
        assert float(u.find('limit').get('effort'))==j['effort']
        assert u.get('type')==j['kind']
    for f in sensor_frames():
        assert f['name'] in ls
        origin=js[f['name']+'_fixed'].find('origin');assert np.allclose(np.fromstring(origin.get('xyz'),sep=' '),f['xyz'])
        assert np.allclose(np.fromstring(origin.get('rpy'),sep=' '),f['rpy'])
    for mesh in urdf.findall('.//mesh'):
        assert (ROOT/'ros2/irona_sim'/mesh.get('filename').split('package://irona_sim/')[1]).is_file()
    report['checks']['urdf_usd_kinematic_and_limit_agreement']=True
    spec=graph_specs();assert len(spec)==4
    gstage=Usd.Stage.Open(str(ROOT/'assets/ros2_bridge.usda'))
    for graph in spec:
        for name,typ in graph['nodes']:
            assert gstage.GetPrimAtPath(graph['path']+'/'+name).GetAttribute('node:type').Get()==typ
        for a,b in graph['connections']:
            dn,port=b.split('.',1);assert gstage.GetPrimAtPath(graph['path']+'/'+dn).GetAttribute(port).GetConnections()
    report['checks']['authored_ros_graph_node_and_connection_contract']=True
    for path in (ROOT/'source').glob('*.py'):ast.parse(path.read_text())
    for path in (ROOT/'ros2').rglob('*.py'):ast.parse(path.read_text())
    import yaml
    rviz=yaml.safe_load((ROOT/'ros2/irona_sim/rviz/irona.rviz').read_text())
    assert rviz['Visualization Manager']['Global Options']['Fixed Frame']=='world'
    report['checks']['python_syntax_rviz_yaml']=True
    report['limitations']=['Graph structure, sensor schemas and transforms checked offline; node execution, RTX output, DDS transport and RViz displays untested here.',
       'Independent MuJoCo checks use simplified collisions. USD joint friction is PhysX-specific; no claim of friction equivalence.',
       'Nominal sensor extrinsics and pinhole intrinsics are design values; replace with measured hardware calibration.',
       'CAD joint pieces are geometry prototypes; full-range shell clearance and hardware fit are not certified.']
    (ROOT/'validation/revision_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
