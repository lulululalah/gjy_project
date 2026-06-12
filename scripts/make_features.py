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


def _rep_normal(samples):
    """面板的面积加权代表法向 [nx,ny,nz]。"""
    acc = np.zeros(3)
    A = 0.0
    for cen, n, a in samples:
        acc += a * np.array([n.X(), n.Y(), n.Z()])
        A += a
    norm = np.linalg.norm(acc)
    return acc / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])


def _trailing_row(samples, span_axis, length_axis, aft_sign, per_row, band_frac=0.16):
    """在面板的**后缘**（length 轴靠 aft 一侧）沿 span 轴铺一排。

    samples: [(cen[3], normal, area)]。返回 [(gp_Pnt, gp_Dir)]。
    """
    cens = np.array([s[0] for s in samples])
    proj_span = cens[:, span_axis]
    proj_len = cens[:, length_axis]
    lo, hi = proj_len.min(), proj_len.max()
    rng = hi - lo + 1e-12
    if aft_sign >= 0:                                  # 后缘在 length 轴正向
        band = np.where(proj_len >= hi - band_frac * rng)[0]
    else:
        band = np.where(proj_len <= lo + band_frac * rng)[0]
    if len(band) < per_row:
        order = np.argsort(proj_len)
        band = order[-per_row:] if aft_sign >= 0 else order[:per_row]

    band = band[np.argsort(proj_span[band])]
    s_lo, s_hi = proj_span[band].min(), proj_span[band].max()
    targets = np.linspace(s_lo, s_hi, per_row)
    chosen, used = [], set()
    for t in targets:
        for idx in band[np.argsort(np.abs(proj_span[band] - t))]:
            if idx not in used:
                used.add(idx)
                chosen.append(idx)
                break
    return [(gp_Pnt(*cens[i]), samples[i][1]) for i in chosen]


def row_sites(shape, max_panels=4, per_row=10):
    """在若干最大的近水平面板（机翼/平尾）后缘各铺一排。返回 (gp_Pnt, gp_Dir)。"""
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    g = box.Get()
    ext = [g[3] - g[0], g[4] - g[1], g[5] - g[2]]
    center = [(g[0] + g[3]) / 2, (g[1] + g[4]) / 2, (g[2] + g[5]) / 2]
    diag = math.sqrt(sum(e * e for e in ext))
    vert_axis = int(np.argmin(ext))
    horiz = [i for i in range(3) if i != vert_axis]
    span_axis = horiz[int(np.argmax([ext[i] for i in horiz]))]   # 翼展（较长水平轴）
    length_axis = horiz[1 - horiz.index(span_axis)]              # 机身纵轴（较短水平轴）

    BRepMesh_IncrementalMesh(shape, diag * 0.001, False, 0.5, True)

    panels = []
    top_pt = None  # (vert坐标, length坐标) 用于判定 aft 方向（竖尾最高点在尾部）
    for f in _faces(shape):
        s = _triangle_samples(f)
        if len(s) < per_row:
            continue
        for cen, _, _ in s:
            if top_pt is None or cen[vert_axis] > top_pt[0]:
                top_pt = (cen[vert_axis], cen[length_axis])
        panels.append((sum(t[2] for t in s), s, _rep_normal(s)))
    if not panels:
        return []

    aft_sign = 1.0 if (top_pt is not None and top_pt[1] >= center[length_axis]) else -1.0

    # 只取近水平的大面板（机翼/平尾蒙皮），按面积取前 max_panels
    near_h = [p for p in panels if abs(p[2][vert_axis]) > 0.6]
    near_h.sort(key=lambda p: -p[0])
    near_h = near_h[:max_panels]

    sites = []
    for _, s, _ in near_h:
        sites.extend(_trailing_row(s, span_axis, length_axis, aft_sign, per_row))
    return sites


def main():
    ap = argparse.ArgumentParser(description="【临时】生成半球铆钉特征 STP（成排对称，待 DL 替换）")
    ap.add_argument("--step", required=True)
    ap.add_argument("--out", required=True, help="输出半球铆钉特征 STP")
    ap.add_argument("--panels", type=int, default=4, help="布置铆钉排的最大面板数（机翼/平尾）")
    ap.add_argument("--per-row", type=int, default=10, help="每排铆钉数")
    ap.add_argument("--size-frac", type=float, default=0.0012,
                    help="半球半径 / 包围盒对角线（再缩小为之前的 1/2）")
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
