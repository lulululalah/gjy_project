"""按锚点 STP 注入铆钉 -> 导出注入后 STP + Polyscope 截图。

注入位置完全来自 --anchors 指定的锚点 STP（不含任何启发式选面）。锚点当前由
临时脚本 scripts/make_anchors.py 生成，后期改为 DL 模型预测，本脚本无需改动。

用法（需 OCC 环境）：
    /opt/anaconda3/envs/occ/bin/python scripts/inject_rivets_view.py \
        --step "data/Xian Y-20 v3.step" \
        --anchors "data/Xian Y-20 v3_anchors.stp" \
        --out-step y20_rivets.stp --out y20_rivets.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Compound, topods
from OCC.Core.TopTools import TopTools_IndexedMapOfShape, TopTools_ListIteratorOfListOfShape
from OCC.Core.gp import gp_Ax2, gp_Pnt

from brep_defeature.occ.anchors import read_anchors_stp, resolve_anchor
from brep_defeature.occ.extract import read_step


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def face_map(shape):
    fmap = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, fmap)
    return fmap


def faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(topods.Face(exp.Current()))
        exp.Next()
    return out


def bbox(shape):
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    return box.Get()


def make_rivet(pnt, normal, radius, height):
    ax = gp_Ax2(pnt, normal)
    cyl = BRepPrimAPI_MakeCylinder(ax, radius, height).Shape()
    top = gp_Pnt(pnt.X() + normal.X() * height,
                 pnt.Y() + normal.Y() * height,
                 pnt.Z() + normal.Z() * height)
    sph = BRepPrimAPI_MakeSphere(gp_Ax2(top, normal), radius).Shape()
    return BRepAlgoAPI_Fuse(cyl, sph).Shape()


def tessellate(shape, lin_defl):
    BRepMesh_IncrementalMesh(shape, lin_defl, False, 0.5, True)
    fmap = face_map(shape)
    verts, tris, tri_face = [], [], []
    for fid in range(1, fmap.Size() + 1):
        face = topods.Face(fmap.FindKey(fid))
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        base = len(verts)
        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i).Transformed(trsf)
            verts.append((p.X(), p.Y(), p.Z()))
        rev = face.Orientation() == 1
        for i in range(1, tri.NbTriangles() + 1):
            n1, n2, n3 = tri.Triangle(i).Get()
            if rev:
                n1, n3 = n3, n1
            tris.append((base + n1 - 1, base + n2 - 1, base + n3 - 1))
            tri_face.append(fid)
    return np.asarray(verts, float), np.asarray(tris, int), np.asarray(tri_face, int)


def rivet_face_ids(fuse, tools_compound, result_fmap):
    ids = set()
    for tf in faces(tools_compound):
        for getter in (fuse.Modified, fuse.Generated):
            try:
                lst = getter(tf)
            except RuntimeError:
                continue
            it = TopTools_ListIteratorOfListOfShape(lst)
            while it.More():
                sub = it.Value()
                if sub.ShapeType() == TopAbs_FACE:
                    fid = result_fmap.FindIndex(topods.Face(sub))
                    if fid > 0:
                        ids.add(fid)
                it.Next()
        fid = result_fmap.FindIndex(tf)
        if fid > 0:
            ids.add(fid)
    return ids


def export_step(shape, path):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(path) != 1:
        raise IOError(f"导出 STP 失败: {path}")


# --------------------------------------------------------------------------- #
def render(verts, tris, tri_is_rivet, out_png, center, diag, vert_axis):
    colors = np.tile(np.array([0.72, 0.74, 0.78]), (len(tris), 1))
    colors[tri_is_rivet] = np.array([0.90, 0.15, 0.12])
    up_name = ["x_up", "y_up", "z_up"][vert_axis]
    offset = np.full(3, 0.85)
    offset[vert_axis] = 1.0
    cam = np.asarray(center) + offset * diag * 1.0
    try:
        import polyscope as ps

        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_up_dir(up_name)
        ps.set_window_size(1600, 1000)
        m = ps.register_surface_mesh("model", verts, tris, smooth_shade=True)
        m.add_color_quantity("part", colors, defined_on="faces", enabled=True)
        ps.look_at(tuple(cam), tuple(center))
        ps.screenshot(out_png, transparent_bg=False)
        log(f"  [polyscope] 已保存 {out_png}")
        return "polyscope"
    except Exception as e:
        log(f"  [polyscope 失败 -> matplotlib] {type(e).__name__}: {e}")
        return _render_matplotlib(verts, tris, tri_is_rivet, out_png)


def _render_matplotlib(verts, tris, tri_is_rivet, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")
    fc = np.tile(np.array([0.72, 0.74, 0.78, 1.0]), (len(tris), 1))
    fc[tri_is_rivet] = np.array([0.90, 0.15, 0.12, 1.0])
    coll = Poly3DCollection(verts[tris], facecolors=fc, edgecolors="none")
    ax.add_collection3d(coll)
    mn, mx = verts.min(0), verts.max(0)
    ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
    ax.set_box_aspect(mx - mn)
    ax.view_init(elev=60, azim=-60)
    ax.set_axis_off()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    log(f"  [matplotlib] 已保存 {out_png}")
    return "matplotlib"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="按锚点 STP 注入铆钉并导出 STP + 截图")
    ap.add_argument("--step", required=True)
    ap.add_argument("--anchors", required=True, help="锚点 STP（注入位置，唯一来源）")
    ap.add_argument("--out", default="rivets.png", help="截图 PNG")
    ap.add_argument("--out-step", default=None, help="注入后模型 STP")
    ap.add_argument("--radius-frac", type=float, default=0.004, help="铆钉半径 / 包围盒对角线")
    args = ap.parse_args()

    log(f"读取模型 {args.step}")
    shape = read_step(args.step)
    log(f"  面数 {face_map(shape).Size()}")
    xmin, ymin, zmin, xmax, ymax, zmax = bbox(shape)
    ext = [xmax - xmin, ymax - ymin, zmax - zmin]
    center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    vert_axis = int(np.argmin(ext))
    diag = math.sqrt(sum(e * e for e in ext))
    radius = diag * args.radius_frac

    anchors = read_anchors_stp(args.anchors)
    log(f"读取锚点 {len(anchors)} 个 <- {args.anchors}")

    comp = TopoDS_Compound()
    bld = BRep_Builder()
    bld.MakeCompound(comp)
    placed = 0
    for xyz in anchors:
        pnt, normal = resolve_anchor(shape, xyz)
        if pnt is None:
            continue
        bld.Add(comp, make_rivet(pnt, normal, radius, radius * 0.5))
        placed += 1
    log(f"  在锚点处放置铆钉刀具 {placed} 个（半径 {radius:.3f}）")

    mode, rivet_ids = "fused", set()
    try:
        fuse = BRepAlgoAPI_Fuse(shape, comp)
        fuse.Build()
        if not fuse.IsDone():
            raise RuntimeError("Fuse 未完成")
        result = fuse.Shape()
        result_fmap = face_map(result)
        rivet_ids = rivet_face_ids(fuse, comp, result_fmap)
        log(f"  Fuse 成功：结果面数 {result_fmap.Size()}，铆钉面 {len(rivet_ids)}")
    except Exception as e:
        log(f"  Fuse 失败 -> 叠加渲染: {type(e).__name__}: {e}")
        mode = "overlay"

    if mode == "fused":
        if args.out_step:
            export_step(result, args.out_step)
            log(f"  注入后模型已导出 -> {args.out_step}")
        verts, tris, tri_face = tessellate(result, diag * 0.0008)
        tri_is_rivet = np.isin(tri_face, list(rivet_ids)) if rivet_ids else np.zeros(len(tris), bool)
    else:
        v1, t1, _ = tessellate(shape, diag * 0.0008)
        v2, t2, _ = tessellate(comp, diag * 0.0008)
        verts = np.vstack([v1, v2])
        tris = np.vstack([t1, t2 + len(v1)])
        tri_is_rivet = np.concatenate([np.zeros(len(t1), bool), np.ones(len(t2), bool)])

    log(f"  三角化：顶点 {len(verts)}，三角面 {len(tris)}，铆钉三角 {int(tri_is_rivet.sum())}")
    backend = render(verts, tris, tri_is_rivet, args.out, center, diag, vert_axis)
    log(f"完成（mode={mode}, render={backend}）-> {args.out}")


if __name__ == "__main__":
    main()
