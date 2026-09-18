"""Render the actual CAD triangulation; no AI concept imagery is used."""
import json,os
from pathlib import Path
import numpy as np
import trimesh
os.environ.setdefault("MUJOCO_GL","egl")
import mujoco
import xml.etree.ElementTree as ET
from PIL import Image,ImageDraw,ImageFont
from model import ROOT,LINKS,TOTAL_MASS,JOINTS

COLORS={"silver":(.58,.66,.71),"bright_silver":(.76,.81,.83),"graphite":(.060,.082,.105),
        "ivory":(.93,.92,.86),"copper":(.67,.17,.095),"black":(.013,.020,.025),
        "cyan":(.10,.72,.86),"rubber":(.07,.075,.085),"red":(.50,.07,.065)}

def load_scene():
    parts=json.loads((ROOT/"config/visual_meshes.json").read_text())
    root=ET.Element("mujoco",model="Irona_CAD_render")
    visual=ET.SubElement(root,"visual")
    ET.SubElement(visual,"global",offwidth="1800",offheight="1800")
    ET.SubElement(visual,"quality",offsamples="4",shadowsize="4096")
    ET.SubElement(visual,"headlight",ambient=".3 .3 .3",diffuse=".7 .7 .7",specular=".25 .25 .25")
    assets=ET.SubElement(root,"asset");world=ET.SubElement(root,"worldbody")
    ET.SubElement(assets,"texture",type="skybox",builtin="flat",rgb1=".925 .934 .932",rgb2=".925 .934 .932",width="512",height="512")
    for p in parts:
        vs=np.asarray(p["vertices"])*.001;fs=np.asarray(p["faces"],dtype=int)
        normals=trimesh.Trimesh(vertices=vs,faces=fs,process=False).vertex_normals
        ET.SubElement(assets,"mesh",name=p["name"],vertex=" ".join(f"{x:.7f}" for x in vs.ravel()),face=" ".join(str(x) for x in fs.ravel()),
                      normal=" ".join(f"{x:.6f}" for x in normals.ravel()),smoothnormal="true")
        ET.SubElement(world,"geom",type="mesh",mesh=p["name"],rgba=" ".join(str(v) for v in (*COLORS[p["material"]],1)),contype="0",conaffinity="0")
    ET.SubElement(world,"light",pos="2 -3 4",dir="-1 1 -2",directional="true",diffuse=".75 .75 .75",ambient=".15 .15 .15",castshadow="true")
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    return (m,d),None,None

def snapshot(ren,win,eye,size=(900,1200),scale=.525,target=(0,0,.4572)):
    m,d=ren
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=target
    delta=np.asarray(eye)-target
    camera.azimuth=180+np.degrees(np.arctan2(delta[1],delta[0]))
    camera.elevation=-np.degrees(np.arctan2(delta[2],np.linalg.norm(delta[:2])))
    camera.distance=2*scale/np.tan(np.deg2rad(45/2))/2
    # Fovy is the vertical field of view for a free camera.
    m.vis.global_.fovy=45
    r=mujoco.Renderer(m,height=size[1],width=size[0]);r.update_scene(d,camera)
    img=Image.fromarray(r.render().copy()).convert("RGB");r.close()
    return img

def font(size,bold=False):
    p="/usr/share/fonts/truetype/dejavu/DejaVuSans"+("-Bold" if bold else "")+".ttf"
    return ImageFont.truetype(p,size)

def main():
    ren,win,actors=load_scene()
    iso=snapshot(ren,win,(2,-3,1.25),(1100,1220),.555)
    front=snapshot(ren,win,(4,0,.4572),(660,1120),.52)
    side=snapshot(ren,win,(0,-4,.4572),(600,1120),.52)
    iso.save(ROOT/"previews/irona_cad.png")
    sheet=Image.new("RGB",(2500,1740),(236,239,237));d=ImageDraw.Draw(sheet)
    d.text((90,62),"IRONA / 914",fill="#182931",font=font(84,True))
    d.text((95,168),"FIVE-FINGER HANDS / RGB-D VISION / 360-DEGREE 3D LIDAR",fill="#52666d",font=font(24))
    d.line((95,224,2400,224),fill="#a7b7b5",width=2)
    sheet.paste(iso,(30,250));sheet.paste(front,(1140,352));sheet.paste(side,(1810,352))
    d=ImageDraw.Draw(sheet)
    d.text((1260,278),"FRONT",fill="#50666b",font=font(22,True));d.text((1970,278),"SIDE",fill="#50666b",font=font(22,True))
    d.text((95,1503),"914.4 mm",fill="#182931",font=font(38,True))
    d.text((95,1558),"EXACT REST HEIGHT",fill="#52666d",font=font(19))
    d.text((485,1503),"62 joints",fill="#182931",font=font(38,True))
    d.text((485,1558),"FLOATING-BASE ARTICULATION",fill="#52666d",font=font(19))
    d.text((1050,1503),"RGB-D + 3D lidar",fill="#182931",font=font(38,True))
    d.text((1050,1558),"NATIVE USD SENSOR PRIMS + ROS 2",fill="#52666d",font=font(19))
    d.text((1740,1503),"USD + STEP + STL",fill="#182931",font=font(38,True))
    d.text((1740,1558),"EDITABLE DESIGN & PRINTABLE EXTERIOR",fill="#52666d",font=font(19))
    d.line((95,1620,2400,1620),fill="#a7b7b5",width=2)
    d.text((95,1652),"ACTUAL CAD GEOMETRY  /  Irona-inspired research prototype  /  Printed parts are an exterior and static display kit.",fill="#52666d",font=font(22))
    sheet.save(ROOT/"previews/irona_design_sheet.png")
    pass

if __name__=="__main__":main()
