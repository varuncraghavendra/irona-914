"""Render actual hand CAD at measured open/closed MuJoCo states."""
import json,os
os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np
import mujoco
from PIL import Image,ImageDraw
from render_previews import load_scene,font
from render_dynamics import quatmat
from model import ROOT,LINKS,forward_kinematics


def main():
    (m,d),_,_=load_scene();parts=json.loads((ROOT/'config/visual_meshes.json').read_text())
    p0=m.geom_pos.copy();R0=[quatmat(q) for q in m.geom_quat]
    trace=json.loads((ROOT/'validation/hands_trajectory.json').read_text())
    opt=mujoco.MjvOption();opt.geomgroup[1]=0
    selected=['left_hand','left_wrist_pitch_link']
    for i,part in enumerate(parts):
        if part['body'] not in selected and not any(part['body'].startswith('left_'+f+'_') for f in ['index','middle','ring','little','thumb']):m.geom_group[i]=1
    panel=[];renderer=mujoco.Renderer(m,width=1100,height=1000)
    for t in [0,3.99]:
        row=min(trace,key=lambda r:abs(r['t']-t));frames=forward_kinematics(row['q'],row['root'],quatmat(row['root_quat_wxyz']))
        for i,part in enumerate(parts):
            n=part['body'];p,R=frames[n];m.geom_pos[i]=p+R@(p0[i]-LINKS[n]['position']);mujoco.mju_mat2Quat(m.geom_quat[i],(R@R0[i]).ravel())
        mujoco.mj_forward(m,d)
        cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:]=frames['left_hand'][0]+[.01,0,-.049];cam.distance=.245;cam.azimuth=195;cam.elevation=-12
        renderer.update_scene(d,cam,scene_option=opt);panel.append(Image.fromarray(renderer.render().copy()))
    renderer.close()
    out=Image.new('RGB',(2380,1340),(236,239,237));draw=ImageDraw.Draw(out)
    draw.text((70,45),'IRONA / FIVE-FINGER HAND',fill='#182931',font=font(58,True))
    draw.text((74,124),'16 joints per hand  /  separate phalanges, knuckles and contact pads',fill='#52666d',font=font(25))
    for i,img in enumerate(panel):out.paste(img,(65+i*1150,225))
    draw=ImageDraw.Draw(out)
    for x,label in [(80,'OPEN'),(1230,'CLOSING')]:draw.text((x,192),label,fill='#50666b',font=font(23,True))
    draw.text((75,1265),'Actual CAD at measured MuJoCo states. Hardware linkage fit and real grasping are not validated.',fill='#52666d',font=font(24))
    out.save(ROOT/'previews/irona_hands_detail.png')
if __name__=='__main__':main()
