"""Replay measured MuJoCo states using the actual CAD meshes, with force glyphs.

The source trajectories come from mujoco_check.py. No poses are fabricated here.
Requires ffmpeg and an EGL-capable MuJoCo rendering environment.
"""
import subprocess,json,math
import numpy as np
from PIL import Image,ImageDraw
from render_previews import load_scene,font
from model import ROOT,LINKS,THRUSTERS,forward_kinematics
import mujoco

def quatmat(q):
    R=np.zeros(9);mujoco.mju_quat2Mat(R,np.asarray(q));return R.reshape(3,3)

def axis_frame(z):
    z=np.asarray(z)/max(np.linalg.norm(z),1e-9)
    ref=np.array([1,0,0]) if abs(z[0])<.9 else np.array([0,1,0])
    y=np.cross(z,ref);y/=np.linalg.norm(y);x=np.cross(y,z)
    return np.column_stack([x,y,z])

def add_geom(scene,typ,size,pos,R,color):
    g=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g,typ,np.asarray(size,dtype=float),np.asarray(pos,dtype=float),np.asarray(R,dtype=float).ravel(),np.asarray(color,dtype=np.float32))
    scene.ngeom+=1

def main():
    (m,d),_,_=load_scene()
    parts=json.loads((ROOT/"config/visual_meshes.json").read_text())
    base_pos=m.geom_pos.copy();base_R=[quatmat(q) for q in m.geom_quat]
    w,h,fps=480,640,12
    renderer=mujoco.Renderer(m,height=h,width=w)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth=130;camera.elevation=-12
    out=ROOT/"previews/irona_dynamics.mp4"
    proc=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{w}x{h}","-r",str(fps),"-i","-","-an","-vcodec","libx264","-crf","22","-pix_fmt","yuv420p","-movflags","+faststart",str(out)],stdin=subprocess.PIPE)
    try:
        for mode in ["hands","walk","flight"]:
            trace=json.loads((ROOT/f"validation/{mode}_trajectory.json").read_text())
            times=np.array([s["t"] for s in trace])
            duration=round(times[-1]+.008)
            camera.distance=1.45 if mode=="hands" else (1.55 if mode=="walk" else 2.28)
            camera.lookat[:]=[0,0,.44] if mode=="hands" else ([.22,0,.455] if mode=="walk" else [0,0,.76])
            for t in {"hands":np.linspace(0,7.99,36),"walk":np.linspace(8,19.99,36),"flight":np.linspace(0,15.99,48)}[mode]:
                row=trace[min(np.searchsorted(times,t),len(trace)-1)]
                frames=forward_kinematics(row["q"],row["root"],quatmat(row["root_quat_wxyz"]))
                for i,part in enumerate(parts):
                    n=part["body"];p,R=frames[n]
                    m.geom_pos[i]=p+R@(base_pos[i]-LINKS[n]["position"])
                    mujoco.mju_mat2Quat(m.geom_quat[i],(R@base_R[i]).ravel())
                mujoco.mj_forward(m,d);renderer.update_scene(d,camera)
                # A neutral ground reference and grid make contact and lift visible.
                add_geom(renderer.scene,mujoco.mjtGeom.mjGEOM_BOX,[2,2,.002],[0,0,-.004],np.eye(3),[.82,.86,.85,1])
                for v in np.arange(-1,1.001,.10):
                    add_geom(renderer.scene,mujoco.mjtGeom.mjGEOM_BOX,[1,.0005,.0005],[0,v,-.001],np.eye(3),[.66,.73,.72,1])
                    add_geom(renderer.scene,mujoco.mjtGeom.mjGEOM_BOX,[.0005,1,.0005],[v,0,-.001],np.eye(3),[.66,.73,.72,1])
                # Glyphs show thrust magnitude; exhaust-fluid simulation is not implied.
                if mode=="flight":
                    for j,force in zip(THRUSTERS,row["jet_norms"]):
                        if force<2:continue
                        p,R=frames[j["body"]];tip=p+R@j["point"]
                        length=.10*math.sqrt(force/50)
                        add_geom(renderer.scene,mujoco.mjtGeom.mjGEOM_CAPSULE,[.010,length/2,0],tip-R[:,2]*length/2,R,[1,.39,.05,.80])
                img=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(img)
                draw.rectangle((0,0,w,104),fill=(236,239,237))
                draw.text((26,17),"IRONA / 914",font=font(24,True),fill="#182931")
                draw.text((26,60),f"{mode.upper()}  |  MuJoCo physics check  |  t = {t:04.1f} s",font=font(12),fill="#50666b")
                draw.rectangle((0,h-52,w,h),fill=(236,239,237))
                draw.text((26,h-40),"Accelerated MuJoCo replay • Isaac Sim untested",font=font(11),fill="#50666b")
                if mode=="flight":draw.text((26,h-22),"Orange glyphs indicate thrust; no combustion or exhaust CFD",font=font(12),fill="#50666b")
                proc.stdin.write(np.asarray(img).tobytes())
            print("Rendered",mode,flush=True)
    finally:
        proc.stdin.close();proc.wait();renderer.close()
    if proc.returncode:raise RuntimeError("ffmpeg failed")
    print(out)

if __name__=="__main__":main()
