"""Remove degenerate CAD tessellation triangles and weld sub-micron seams."""
import json
import numpy as np
import trimesh
from model import ROOT

def clean():
    rows=[];fail=[]
    for p in sorted((ROOT/"print").rglob("*.stl")):
        m=trimesh.load_mesh(p,process=True)
        before=len(m.faces)
        m.merge_vertices(digits_vertex=5)  # 0.00001 mm, far below FDM resolution
        m.update_faces(m.nondegenerate_faces());m.update_faces(m.unique_faces());m.remove_unreferenced_vertices()
        m.fix_normals(multibody=True)
        if not m.is_watertight:
            trimesh.repair.fill_holes(m)
        if not m.is_watertight:
            fail.append(p.name)
        m.export(p)
        rows.append(dict(file=p.name,faces_before=before,faces_after=len(m.faces),watertight=bool(m.is_watertight)))
    (ROOT/"validation/mesh_cleanup.json").write_text(json.dumps(rows,indent=2))
    print("Print meshes:",len(rows),"remaining open meshes:",fail)
    if fail:raise RuntimeError("Open print meshes remain")

if __name__=="__main__":clean()
