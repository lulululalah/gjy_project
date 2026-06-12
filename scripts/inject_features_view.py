"""把「半球铆钉 STP」融入飞机 BRep -> 导出融合后 STP + Polyscope 截图。

飞机是**多实体 compound**（机身/机翼/发动机/尾翼等各为独立 solid）。若把整机当
一个布尔参数去 Fuse，会把原本相交的相邻 solid 合并、甚至丢面 -> 机身/机头丢失。

正确做法：把每个铆钉只 Fuse 进它所在的那个 host solid，其余 solid 原样保留，
最后重组 compound。这样所有零件都不丢，只有收到铆钉的 solid 被修改。

着色：飞机蒙皮无球面，故“球面 = 铆钉”，据此着色，无需依赖 BOP 历史。

用法（需 OCC 环境）：
    /opt/anaconda3/envs/occ/bin/python scripts/inject_features_view.py \
        --step "data/Xian Y-20 v3.step" \
        --features "data/Xian Y-20 v3_features.stp" \
        --out-step y20_features.stp --out y20_features.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GeomAbs import GeomAbs_Sphere
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Compound, topods
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.gp import gp_Pnt

from brep_defeature.occ.extract import read_step

BODY_COLOR = (0.72, 0.74, 0.78)
RIVET_COLOR = (0.12, 0.45, 0.95)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def face_map(shape):
    fmap = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, fmap)
    return fmap


def solids(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        out.append(topods.Solid(exp.Current()))
        exp.Next()
    return out


def feature_solids(shape):
    return solids(shape)


def bbox(shape):
    b = Bnd_Box()
    brepbndlib.Add(shape, b)
    return b


def bbox_center(shape):
    g = bbox(shape).Get()
    return gp_Pnt((g[0] + g[3]) / 2, (g[1] + g[4]) / 2, (g[2] + g[5]) / 2)


def is_sphere(face):
    return BRepAdaptor_Surface(face).GetType() == GeomAbs_Sphere


def tessellate(shape, lin_defl):
    BRepMesh_IncrementalMesh(shape, lin_defl, False, 0.5, True)
    fmap = face_map(shape)
    verts, tris, tri_rivet = [], [], []
    for fid in range(1, fmap.Size() + 1):
        face = topods.Face(fmap.FindKey(fid))
        rivet = is_sphere(face)
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
            tri_rivet.append(rivet)
    return np.asarray(verts, float), np.asarray(tris, int), np.asarray(tri_rivet, bool)


def export_step(shape, path):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(path) != 1:
        raise IOError(f"导出 STP 失败: {path}")


def fuse_per_solid(air_solids, feats):
    """每个铆钉只 Fuse 进它的 host solid，其余 solid 原样保留，重组 compound。"""
    boxes = []
    for s in air_solids:
        b = bbox(s)
        b.Enlarge(1e-6)
        boxes.append(b)

    # 铆钉 -> host solid 下标
    groups = {}
    for fs in feature_solids(feats):
        c = bbox_center(fs)
        cands = [i for i, b in enumerate(boxes) if not b.IsOut(c)] or list(range(len(air_solids)))
        best, bestd = None, 1e18
        v = BRepBuilderAPI_MakeVertex(c).Vertex()
        for i in cands:
            d = BRepExtrema_DistShapeShape(v, air_solids[i])
            if d.IsDone() and d.Value() < bestd:
                bestd, best = d.Value(), i
        if best is not None:
            groups.setdefault(best, []).append(fs)

    comp = TopoDS_Compound()
    bld = BRep_Builder()
    bld.MakeCompound(comp)
    n_applied = 0
    for i, s in enumerate(air_solids):
        if i in groups:
            tools = TopoDS_Compound()
            bld.MakeCompound(tools)
            for fs in groups[i]:
                bld.Add(tools, fs)
            fuse = BRepAlgoAPI_Fuse(s, tools)
            fuse.Build()
            if fuse.IsDone() and BRepCheck_Analyzer(fuse.Shape()).IsValid():
                bld.Add(comp, fuse.Shape())
                n_applied += len(groups[i])
                continue
            log(f"  solid#{i} Fuse 失败/无效，保留原件（{len(groups[i])} 个铆钉未施加）")
        bld.Add(comp, s)
    return comp, n_applied


def render(verts, tris, tri_rivet, out_png, center, diag, vert_axis):
    colors = np.tile(np.array(BODY_COLOR), (len(tris), 1))
    colors[tri_rivet] = np.array(RIVET_COLOR)
    up_name = ["x_up", "y_up", "z_up"][vert_axis]
    offset = np.full(3, 0.85)
    offset[vert_axis] = 1.0
    cam = np.asarray(center) + offset * diag * 0.72
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
    except Exception as e:  # noqa: BLE001
        log(f"  [polyscope 失败 -> matplotlib] {type(e).__name__}: {e}")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection="3d")
        fc = np.concatenate([colors, np.ones((len(colors), 1))], axis=1)
        ax.add_collection3d(Poly3DCollection(verts[tris], facecolors=fc, edgecolors="none"))
        mn, mx = verts.min(0), verts.max(0)
        ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
        ax.set_box_aspect(mx - mn)
        ax.view_init(elev=60, azim=-60)
        ax.set_axis_off()
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
        log(f"  [matplotlib] 已保存 {out_png}")
        return "matplotlib"


def main():
    ap = argparse.ArgumentParser(description="逐 solid 融入半球铆钉并导出 STP + 截图")
    ap.add_argument("--step", required=True)
    ap.add_argument("--features", required=True, help="半球铆钉 STP（实体 compound）")
    ap.add_argument("--out", default="features.png")
    ap.add_argument("--out-step", default=None)
    args = ap.parse_args()

    log(f"读取飞机 {args.step}")
    shape = read_step(args.step)
    air = solids(shape)
    g = bbox(shape).Get()
    ext = [g[3] - g[0], g[4] - g[1], g[5] - g[2]]
    center = [(g[0] + g[3]) / 2, (g[1] + g[4]) / 2, (g[2] + g[5]) / 2]
    vert_axis = int(np.argmin(ext))
    diag = math.sqrt(sum(e * e for e in ext))
    log(f"  飞机 solid 数 {len(air)}，面数 {face_map(shape).Size()}")

    feats = read_step(args.features)
    log(f"  铆钉数 {len(feature_solids(feats))}")

    result, n_applied = fuse_per_solid(air, feats)
    n_solid_out = len(solids(result))
    valid = BRepCheck_Analyzer(result).IsValid()
    log(f"  融合：solid {len(air)} -> {n_solid_out}（应相等），施加铆钉 {n_applied}，有效={valid}")

    if args.out_step:
        export_step(result, args.out_step)
        log(f"  融合后模型已导出 -> {args.out_step}")

    verts, tris, tri_rivet = tessellate(result, diag * 0.0008)
    log(f"  三角化：顶点 {len(verts)}，三角面 {len(tris)}，铆钉三角 {int(tri_rivet.sum())}")
    backend = render(verts, tris, tri_rivet, args.out, center, diag, vert_axis)
    log(f"完成（render={backend}）-> {args.out}")


if __name__ == "__main__":
    main()
