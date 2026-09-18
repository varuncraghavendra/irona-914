"""Generate the RViz URDF/meshes and relocatable ROS package runtime from USD sources."""
import json,math,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from model import ROOT,LINKS,JOINTS
from sensors import sensor_frames
from ros_bridge import author_graph_usd

PKG=ROOT/'ros2/irona_sim'

def v(x):return ' '.join(f'{float(a):.10g}' for a in x)

def main():
    visual=json.loads((ROOT/'config/visual_meshes.json').read_text())
    grouped={n:[] for n in LINKS}
    for part in visual:
        mesh=trimesh.Trimesh(vertices=np.array(part['vertices'])*.001-LINKS[part['body']]['position'],faces=part['faces'],process=False)
        grouped[part['body']].append(mesh)
    robot=ET.Element('robot',name='irona_914')
    for name,body in LINKS.items():
        link=ET.SubElement(robot,'link',name=name)
        inertia=ET.SubElement(link,'inertial');ET.SubElement(inertia,'origin',xyz=v(body['com']),rpy='0 0 0')
        ET.SubElement(inertia,'mass',value=str(body['mass']));ix,iy,iz=body['inertia']
        ET.SubElement(inertia,'inertia',ixx=str(ix),iyy=str(iy),izz=str(iz),ixy='0',ixz='0',iyz='0')
        if grouped[name]:
            mesh=trimesh.util.concatenate(grouped[name]);mesh.export(PKG/'meshes'/f'{name}.stl')
            visual=ET.SubElement(link,'visual');geom=ET.SubElement(visual,'geometry')
            ET.SubElement(geom,'mesh',filename=f'package://irona_sim/meshes/{name}.stl')
            mat=ET.SubElement(visual,'material',name='silver_'+name);ET.SubElement(mat,'color',rgba='.58 .66 .71 1')
        if body['collision']:
            col=ET.SubElement(link,'collision');ET.SubElement(col,'origin',xyz=v(body['com']),rpy='0 0 0')
            geom=ET.SubElement(col,'geometry');ET.SubElement(geom,'box',size=v(body['size']))
    for j in JOINTS:
        joint=ET.SubElement(robot,'joint',name=j['name'],type=j['kind'])
        ET.SubElement(joint,'parent',link=j['parent']);ET.SubElement(joint,'child',link=j['child'])
        ET.SubElement(joint,'origin',xyz=v(np.array(LINKS[j['child']]['position'])-LINKS[j['parent']]['position']),rpy='0 0 0')
        ET.SubElement(joint,'axis',xyz={'X':'1 0 0','Y':'0 1 0','Z':'0 0 1'}[j['axis']])
        limits=j['limits'] if j['kind']=='prismatic' else np.deg2rad(j['limits'])
        ET.SubElement(joint,'limit',lower=str(limits[0]),upper=str(limits[1]),effort=str(j['effort']),velocity=str(j['velocity']))
    for f in sensor_frames():
        ET.SubElement(robot,'link',name=f['name'])
        j=ET.SubElement(robot,'joint',name=f['name']+'_fixed',type='fixed')
        ET.SubElement(j,'parent',link=f['parent']);ET.SubElement(j,'child',link=f['name'])
        ET.SubElement(j,'origin',xyz=v(f['xyz']),rpy=v(f['rpy']))
    ET.indent(robot);ET.ElementTree(robot).write(PKG/'urdf/irona.urdf',encoding='unicode',xml_declaration=True)
    author_graph_usd(ROOT/'assets/ros2_bridge.usda')
    from pxr import Usd,UsdGeom,Sdf
    for asset in [ROOT/"assets/irona.usda",ROOT/"assets/irona.usd"]:
        robot_stage=Usd.Stage.Open(str(asset))
        robot_stage.GetPrimAtPath("/Irona/Links/head/Sensors/lidar").CreateAttribute("omni:sensor:tickRate",Sdf.ValueTypeNames.Float,custom=False).Set(10.0)
        robot_stage.GetRootLayer().Save()
    stage=Usd.Stage.CreateNew(str(ROOT/'assets/scene_ros.usda'))
    stage.GetRootLayer().subLayerPaths=['ros2_bridge.usda','scene.usda']
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'));stage.GetRootLayer().Save()
    import csv
    with (ROOT/'config/joint_limits.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['name','parent','child','axis','lower_deg','upper_deg','torque_cap_Nm','speed_cap_rad_s','kp_Nm_per_rad','kd_Nms_per_rad','physx_friction','actuator_basis'])
        for j in JOINTS:writer.writerow([j['name'],j['parent'],j['child'],j['axis'],*j['limits'],j['effort'],j['velocity'],j['kp'],j['kd'],j['friction'],j['actuator']])
    dest=PKG/'project'
    for folder in ['source','assets','config','validation']:(dest/folder).mkdir(exist_ok=True,parents=True)
    for f in (ROOT/'source').glob('*.py'):shutil.copy2(f,dest/'source'/f.name)
    for f in (ROOT/'assets').glob('*.usd*'):shutil.copy2(f,dest/'assets'/f.name)
    for name in ['robot.json','sensors.json','joint_limits.csv']:shutil.copy2(ROOT/'config'/name,dest/'config'/name)
    print(f'URDF: {len(LINKS)} dynamic links, {len(JOINTS)} movable joints, {len(sensor_frames())} fixed sensor frames')

if __name__=='__main__':main()
