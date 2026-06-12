"""在飞机模型机翼/接缝处注入铆钉，并用 Polyscope 自动截图。

用法（需 OCC 环境）：
    /opt/anaconda3/envs/occ/bin/python scripts/inject_rivets_view.py \
        --step "data/Xian Y-20 v3.step" --out rivets.png --n 30

流程：
  1. 读取 STEP，分析包围盒，自动挑选「机翼/尾翼」区域的近水平大面作为宿主。
  2. 在这些面靠近边界（接缝）处采样点，建圆柱+球冠铆钉刀具。
  3. 一次性 Fuse（失败则退化为「叠加铆钉几何」渲染，仍能出图）。
  4. 三角化结果，按面着色（铆钉=红，机身=灰），Polyscope 截图（无 GL 则 matplotlib）。
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Compound, topods
from OCC.Core.TopTools import TopTools_IndexedMapOfShape, TopTools_ListIteratorOfListOfShape
from OCC.Core.TopExp import topexp
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt


# --------------------------------------------------------------------------- #
def log(*a):
    print(*a, file=sys.stderr, flush=True)


def read_step(path: str):
    reader = STEPControl_Reader()
    if reader.ReadFile(path) != 1:
        raise IOError(f"读取失败: {path}")
    reader.TransferRoots()
    return reader.OneShape()


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


def face_area(face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return props.Mass()


def face_centroid_normal(face):
    """面参数中点的 (点, 外法向)。"""
    ad = BRepAdaptor_Surface(face)
    u = 0.5 * (ad.FirstUParameter() + ad.LastUParameter())
    v = 0.5 * (ad.FirstVParameter() + ad.LastVParameter())
    props = BRepLProp_SLProps(ad, u, v, 1, 1e-6)
    if not props.IsNormalDefined():
        return None, None
    p = props.Value()
    n = props.Normal()
    if face.Orientation() == 1:
        n.Reverse()
    return p, n


def bbox(shape):
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    return box.Get()  # xmin,ymin,zmin,xmax,ymax,zmax


# --------------------------------------------------------------------------- #
# 宿主面选择：机翼/尾翼区域的近水平大面
# --------------------------------------------------------------------------- #
def select_hosts(shape, max_hosts=12):
    xmin, ymin, zmin, xmax, ymax, zmax = bbox(shape)
    ext = [xmax - xmin, ymax - ymin, zmax - zmin]
    vert_axis = int(np.argmin(ext))          # 高度方向（最小跨度）
    horiz_axes = [i for i in range(3) if i != vert_axis]
    # 在两个水平轴里，跨度较大的更可能是翼展方向
    span_axis = horiz_axes[int(np.argmax([ext[i] for i in horiz_axes]))]
    center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    half_span = ext[span_axis] / 2
    log(f"  bbox ext={[round(e,1) for e in ext]} vert_axis={vert_axis} span_axis={span_axis}")

    areas = [face_area(f) for f in faces(shape)]
    if not areas:
        return []
    area_thr = np.percentile(areas, 75)      # 取较大的面

    cand = []
    for f, a in zip(faces(shape), areas):
        if a < area_thr:
            continue
        p, n = face_centroid_normal(f)
        if p is None:
            continue
        coords = [p.X(), p.Y(), p.Z()]
        nrm = [n.X(), n.Y(), n.Z()]
        # 近水平：法向主要沿竖直轴
        if abs(nrm[vert_axis]) < 0.6:
            continue
        # 外翼区域：沿翼展方向偏离中心
        span_off = abs(coords[span_axis] - center[span_axis]) / (half_span + 1e-9)
        if span_off < 0.2:                   # 排除机身中线附近
            continue
        cand.append((span_off * a, f, a, span_off))
    cand.sort(key=lambda t: -t[0])
    chosen = cand[:max_hosts]
    log(f"  候选宿主面 {len(cand)} 个，选用 {len(chosen)} 个 (area_thr={area_thr:.1f})")
    return [c[1] for c in chosen]


def sample_points_near_seams(face, n_per_face=4):
    """在面参数域靠近边界（接缝）与内部采样若干 (点, 法向)。"""
    ad = BRepAdaptor_Surface(face)
    u0, u1 = ad.FirstUParameter(), ad.LastUParameter()
    v0, v1 = ad.FirstVParameter(), ad.LastVParameter()
    # 靠近 4 条参数边界的点 + 中心
    fracs = [(0.12, 0.5), (0.88, 0.5), (0.5, 0.12), (0.5, 0.88), (0.5, 0.5)]
    pts = []
    for fu, fv in fracs[: n_per_face + 1]:
        u = u0 + fu * (u1 - u0)
        v = v0 + fv * (v1 - v0)
        try:
            props = BRepLProp_SLProps(ad, u, v, 1, 1e-6)
            if not props.IsNormalDefined():
                continue
            p = props.Value()
            n = props.Normal()
            if face.Orientation() == 1:
                n.Reverse()
            pts.append((p, n))
        except RuntimeError:
            continue
    return pts


def make_rivet(pnt, normal, radius, height):
    ax = gp_Ax2(pnt, normal)
    cyl = BRepPrimAPI_MakeCylinder(ax, radius, height).Shape()
    top = gp_Pnt(pnt.X() + normal.X() * height,
                 pnt.Y() + normal.Y() * height,
                 pnt.Z() + normal.Z() * height)
    sph = BRepPrimAPI_MakeSphere(gp_Ax2(top, normal), radius).Shape()
    return BRepAlgoAPI_Fuse(cyl, sph).Shape()


# --------------------------------------------------------------------------- #
# 三角化
# --------------------------------------------------------------------------- #
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
        reversed_face = face.Orientation() == 1
        for i in range(1, tri.NbTriangles() + 1):
            n1, n2, n3 = tri.Triangle(i).Get()
            if reversed_face:
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


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #
def render(verts, tris, tri_is_rivet, out_png, center, diag, vert_axis):
    colors = np.tile(np.array([0.72, 0.74, 0.78]), (len(tris), 1))  # 机身灰
    colors[tri_is_rivet] = np.array([0.90, 0.15, 0.12])            # 铆钉红
    up_name = ["x_up", "y_up", "z_up"][vert_axis]
    # 斜俯视：竖直分量朝上，两个水平分量给出 3/4 视角，能看到平面投影
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
    except Exception as e:  # 无 GL 上下文等
        log(f"  [polyscope 失败 -> matplotlib] {type(e).__name__}: {e}")
        return _render_matplotlib(verts, tris, tri_is_rivet, out_png)


def _render_matplotlib(verts, tris, tri_is_rivet, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")
    poly = verts[tris]
    fc = np.tile(np.array([0.72, 0.74, 0.78, 1.0]), (len(tris), 1))
    fc[tri_is_rivet] = np.array([0.90, 0.15, 0.12, 1.0])
    coll = Poly3DCollection(poly, facecolors=fc, edgecolors="none", linewidths=0)
    coll.set_sort_zpos(0)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True)
    ap.add_argument("--out", default="rivets.png")
    ap.add_argument("--n", type=int, default=30, help="目标铆钉数")
    ap.add_argument("--hosts", type=int, default=12)
    ap.add_argument("--radius-frac", type=float, default=0.004, help="铆钉半径 / 包围盒对角线")
    args = ap.parse_args()

    log(f"读取 {args.step}")
    shape = read_step(args.step)
    nfaces = face_map(shape).Size()
    log(f"  面数 {nfaces}")
    xmin, ymin, zmin, xmax, ymax, zmax = bbox(shape)
    ext = [xmax - xmin, ymax - ymin, zmax - zmin]
    center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    vert_axis = int(np.argmin(ext))
    diag = math.sqrt(sum(e * e for e in ext))
    radius = diag * args.radius_frac
    log(f"  对角线 {diag:.1f}，铆钉半径 {radius:.3f}")

    hosts = select_hosts(shape, max_hosts=args.hosts)
    if not hosts:
        log("未找到合适宿主面，退出")
        sys.exit(1)

    # 采样点 -> 铆钉刀具（汇入一个 compound，单次 Fuse）
    comp = TopoDS_Compound()
    bld = BRep_Builder()
    bld.MakeCompound(comp)
    placed = 0
    per_face = max(1, args.n // len(hosts))
    for f in hosts:
        for p, n in sample_points_near_seams(f, n_per_face=per_face):
            bld.Add(comp, make_rivet(p, n, radius, radius * 0.5))
            placed += 1
            if placed >= args.n:
                break
        if placed >= args.n:
            break
    log(f"  放置铆钉刀具 {placed} 个")

    # 尝试真实 Fuse
    mode = "fused"
    rivet_ids = set()
    try:
        fuse = BRepAlgoAPI_Fuse(shape, comp)
        fuse.Build()
        if fuse.IsDone():
            result = fuse.Shape()
            result_fmap = face_map(result)
            rivet_ids = rivet_face_ids(fuse, comp, result_fmap)
            log(f"  Fuse 成功，结果面数 {result_fmap.Size()}，铆钉面 {len(rivet_ids)}")
        else:
            raise RuntimeError("Fuse 未完成")
    except Exception as e:
        log(f"  Fuse 失败 -> 叠加渲染: {type(e).__name__}: {e}")
        mode = "overlay"

    if mode == "fused":
        verts, tris, tri_face = tessellate(result, diag * 0.0008)
        tri_is_rivet = np.isin(tri_face, list(rivet_ids)) if rivet_ids else np.zeros(len(tris), bool)
    else:
        # 叠加：机身 + 铆钉刀具分别三角化
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
