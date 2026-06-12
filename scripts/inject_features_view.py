"""把「凸起特征 STP」整体 Fuse 进飞机 BRep -> 导出融合后 STP + Polyscope 截图。

输入 --features 是一组已定位的凸起实体（钉子/半球/梯台）的 STP（compound）。
本脚本只做：读飞机 + 读凸起 -> 布尔并(Fuse) -> 凸起融入原拓扑 -> 导出/渲染。
不含任何选位/启发式逻辑（那部分在 make_features.py，后期由 DL 替换）。

渲染按凸起面的曲面类型着色，直观区分形状：圆柱(钉身)/球(半球·钉头)/圆锥(梯台)。

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

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Sphere
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods
from OCC.Core.TopTools import TopTools_IndexedMapOfShape, TopTools_ListIteratorOfListOfShape

from brep_defeature.occ.extract import read_step

# 类别 -> 颜色（0=机身，1=圆柱/钉身，2=球/半球·钉头，3=圆锥/梯台）
CAT_COLOR = {
    0: (0.72, 0.74, 0.78),
    1: (0.90, 0.15, 0.12),
    2: (0.12, 0.45, 0.95),
    3: (0.96, 0.62, 0.10),
}


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


def feature_face_ids(fuse, feats_shape, result_fmap):
    ids = set()
    for tf in faces(feats_shape):
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


def face_category(face):
    """凸起面按曲面类型分类着色。"""
    t = BRepAdaptor_Surface(face).GetType()
    if t == GeomAbs_Cylinder:
        return 1
    if t == GeomAbs_Sphere:
        return 2
    if t == GeomAbs_Cone:
        return 3
    return 1


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


def export_step(shape, path):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(path) != 1:
        raise IOError(f"导出 STP 失败: {path}")


def render(verts, tris, tri_cat, out_png, center, diag, vert_axis):
    colors = np.array([CAT_COLOR[int(c)] for c in tri_cat], dtype=float)
    up_name = ["x_up", "y_up", "z_up"][vert_axis]
    offset = np.full(3, 0.85)
    offset[vert_axis] = 1.0
    cam = np.asarray(center) + offset * diag * 0.62
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
    ap = argparse.ArgumentParser(description="把凸起特征 STP 融入飞机 BRep 并导出 + 截图")
    ap.add_argument("--step", required=True)
    ap.add_argument("--features", required=True, help="凸起特征 STP（实体 compound）")
    ap.add_argument("--out", default="features.png")
    ap.add_argument("--out-step", default=None, help="融合后模型 STP")
    args = ap.parse_args()

    log(f"读取飞机 {args.step}")
    shape = read_step(args.step)
    n0 = face_map(shape).Size()
    xmin, ymin, zmin, xmax, ymax, zmax = bbox(shape)
    ext = [xmax - xmin, ymax - ymin, zmax - zmin]
    center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    vert_axis = int(np.argmin(ext))
    diag = math.sqrt(sum(e * e for e in ext))

    feats = read_step(args.features)
    n_feat_solids = len(faces(feats))
    log(f"  飞机面数 {n0}；凸起特征 STP 面数 {n_feat_solids}")

    fuse = BRepAlgoAPI_Fuse(shape, feats)
    fuse.Build()
    if not fuse.IsDone():
        log("Fuse 失败")
        sys.exit(1)
    result = fuse.Shape()
    result_fmap = face_map(result)
    feat_ids = feature_face_ids(fuse, feats, result_fmap)
    valid = BRepCheck_Analyzer(result).IsValid()
    log(f"  Fuse 完成：面数 {n0} -> {result_fmap.Size()}（凸起面 {len(feat_ids)}），有效实体={valid}")

    # 凸起面按曲面类型分类
    face_cat = {}
    for fid in feat_ids:
        face_cat[fid] = face_category(topods.Face(result_fmap.FindKey(fid)))

    if args.out_step:
        export_step(result, args.out_step)
        log(f"  融合后模型已导出 -> {args.out_step}")

    verts, tris, tri_face = tessellate(result, diag * 0.0008)
    tri_cat = np.array([face_cat.get(int(f), 0) for f in tri_face], dtype=int)
    n_by_cat = {c: int(np.sum(tri_cat == c)) for c in (1, 2, 3)}
    log(f"  三角化：顶点 {len(verts)}，三角面 {len(tris)}，凸起三角按类别 {n_by_cat}")
    backend = render(verts, tris, tri_cat, args.out, center, diag, vert_axis)
    log(f"完成（render={backend}）-> {args.out}")


if __name__ == "__main__":
    main()
