"""【临时脚手架】生成「半球铆钉」特征 STP，沿大面板按规律成排、左右对称。

⚠️ 启发式占位实现，仅用于在 DL 模型就绪前产出示例特征文件。
   后期由「深度学习预测的铆钉位置」替换，注入管线读取的特征 STP 接口不变。

只生成**半球**铆钉。排布规律：取若干最大的面板（机翼/机身侧板/尾翼皮），
在每块面板上沿其**长轴**方向、靠近**一侧边缘**铺一排铆钉。点取自该面的三角网格
（保证落在已裁剪蒙皮上）。因左右机翼/尾翼是对称面，自然得到左右对称的铆钉行。

输出 STP = 一组已定位在飞机表面上的半球实体（compound），注入脚本整体 Fuse 进飞机。

用法：
    /opt/anaconda3/envs/occ/bin/python scripts/make_features.py \
        --step "data/Xian Y-20 v3.step" --out "data/Xian Y-20 v3_features.stp"
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Compound, topods
from OCC.Core.gp import gp_Dir, gp_Pnt, gp_Vec

from brep_defeature.occ.extract import read_step
from brep_defeature.occ.features import make_hemisphere


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(topods.Face(exp.Current()))
        exp.Next()
    return out


def _triangle_samples(face):
    """单个面 -> [(质心 ndarray[3], 外法向 gp_Dir, 面积)]，质心保证落在已裁剪蒙皮上。"""
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation(face, loc)
    if tri is None:
        return []
    trsf = loc.Transformation()
    nodes = {i: tri.Node(i).Transformed(trsf) for i in range(1, tri.NbNodes() + 1)}
    rev = face.Orientation() == 1
    out = []
    for i in range(1, tri.NbTriangles() + 1):
        a, b, c = tri.Triangle(i).Get()
        p1, p2, p3 = nodes[a], nodes[b], nodes[c]
        cr = gp_Vec(p1, p2).Crossed(gp_Vec(p1, p3))
        m = cr.Magnitude()
        if m < 1e-12:
            continue
        cen = np.array([(p1.X() + p2.X() + p3.X()) / 3,
                        (p1.Y() + p2.Y() + p3.Y()) / 3,
                        (p1.Z() + p2.Z() + p3.Z()) / 3])
        nrm = gp_Dir(cr)
        if rev:
            nrm.Reverse()
        out.append((cen, nrm, 0.5 * m))
    return out


def _face_row(samples, per_row, vert_axis, edge_frac=0.78):
    """在一块面板上沿长轴铺一排：靠近一侧边缘、沿长轴等距取点。

    samples: [(cen[3], normal, area)]。返回 [(gp_Pnt, gp_Dir)]。
    """
    if len(samples) < per_row:
        per_row = max(1, len(samples))
    cens = np.array([s[0] for s in samples])
    horiz = [i for i in range(3) if i != vert_axis]
    pts2d = cens[:, horiz]                       # 投到水平面
    mean = pts2d.mean(axis=0)
    cov = np.cov((pts2d - mean).T)
    w, V = np.linalg.eigh(cov)
    long_dir = V[:, int(np.argmax(w))]           # 长轴（沿展向）
    chord_dir = V[:, int(np.argmin(w))]          # 短轴（弦向）
    proj_long = (pts2d - mean) @ long_dir
    proj_chord = (pts2d - mean) @ chord_dir

    # 取靠近某一侧边缘的带状区域（≈后缘/侧边）
    thresh = np.quantile(proj_chord, edge_frac)
    band = np.where(proj_chord >= thresh)[0]
    if len(band) < per_row:
        band = np.argsort(proj_chord)[-max(per_row, len(band)):]

    # 沿长轴等距挑 per_row 个
    band = band[np.argsort(proj_long[band])]
    lo, hi = proj_long[band].min(), proj_long[band].max()
    targets = np.linspace(lo, hi, per_row)
    chosen, used = [], set()
    for t in targets:
        order = band[np.argsort(np.abs(proj_long[band] - t))]
        for idx in order:
            if idx not in used:
                used.add(idx)
                chosen.append(idx)
                break
    out = []
    for idx in chosen:
        c = cens[idx]
        out.append((gp_Pnt(c[0], c[1], c[2]), samples[idx][1]))
    return out


def row_sites(shape, max_panels=6, per_row=9, min_area_pct=80):
    """在若干最大面板上各铺一排铆钉。返回 (gp_Pnt, gp_Dir) 列表。"""
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    g = box.Get()
    ext = [g[3] - g[0], g[4] - g[1], g[5] - g[2]]
    diag = math.sqrt(sum(e * e for e in ext))
    vert_axis = int(np.argmin(ext))

    BRepMesh_IncrementalMesh(shape, diag * 0.001, False, 0.5, True)

    panels = []
    for f in _faces(shape):
        s = _triangle_samples(f)
        if len(s) < per_row:
            continue
        area = sum(t[2] for t in s)
        panels.append((area, s))
    if not panels:
        return []
    thr = np.percentile([p[0] for p in panels], min_area_pct)
    panels = [p for p in panels if p[0] >= thr]
    panels.sort(key=lambda p: -p[0])
    panels = panels[:max_panels]

    sites = []
    for _, s in panels:
        sites.extend(_face_row(s, per_row, vert_axis))
    return sites


def main():
    ap = argparse.ArgumentParser(description="【临时】生成半球铆钉特征 STP（成排对称，待 DL 替换）")
    ap.add_argument("--step", required=True)
    ap.add_argument("--out", required=True, help="输出半球铆钉特征 STP")
    ap.add_argument("--panels", type=int, default=6, help="布置铆钉排的最大面板数")
    ap.add_argument("--per-row", type=int, default=9, help="每排铆钉数")
    ap.add_argument("--size-frac", type=float, default=0.0024,
                    help="半球半径 / 包围盒对角线（约为之前的 1/5）")
    args = ap.parse_args()

    shape = read_step(args.step)
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    g = box.Get()
    diag = math.sqrt((g[3] - g[0]) ** 2 + (g[4] - g[1]) ** 2 + (g[5] - g[2]) ** 2)
    radius = diag * args.size_frac

    sites = row_sites(shape, max_panels=args.panels, per_row=args.per_row)
    if not sites:
        log("未生成任何铆钉位置")
        sys.exit(1)

    comp = TopoDS_Compound()
    from OCC.Core.BRep import BRep_Builder

    bld = BRep_Builder()
    bld.MakeCompound(comp)
    for pnt, normal in sites:
        bld.Add(comp, make_hemisphere(pnt, normal, radius))

    writer = STEPControl_Writer()
    writer.Transfer(comp, STEPControl_AsIs)
    if writer.Write(args.out) != 1:
        raise IOError(f"写入失败: {args.out}")
    log(f"已写出 {len(sites)} 个半球铆钉（半径 {radius:.4f}，{args.panels} 排）-> {args.out}")


if __name__ == "__main__":
    main()
