"""Package the entire revision, excluding caches/build output, with file hashes."""
import json,hashlib,zipfile
from model import ROOT

def main():
    excluded={'__pycache__','build','install','log','.git'}
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(x in excluded for x in p.relative_to(ROOT).parts) and p.name!='PROJECT_MANIFEST.json')
    manifest={'project':'Irona 914','version':'2.0.0','height_mm':914.4,
              'isaac_sim_runtime_tested':False,'ros2_runtime_tested':False,'rviz_runtime_tested':False,
              'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    out=ROOT/'PROJECT_MANIFEST.json';out.write_text(json.dumps(manifest,indent=2));files.append(out)
    archive=ROOT.parent/'Irona_914_IsaacSim_5x.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(ROOT.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip();assert bad is None,bad
    print(json.dumps({'archive':str(archive),'files':len(files),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
