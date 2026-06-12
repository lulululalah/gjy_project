"""【临时脚手架】生成「微小凸起特征」STP（混合 钉子/半球/梯台）。

⚠️ 启发式占位实现，仅用于在 DL 模型就绪前产出示例特征文件。
   后期由「深度学习预测的凸起位置/类型/尺寸」替换，注入管线读取的特征 STP 接口不变。

输出 STP = 一组**已定位在飞机表面上的凸起实体**（compound）。注入脚本把它整体
Fuse 进飞机 BRep，凸起即融入原拓扑。

用法：
    /opt/anaconda3/envs/occ/bin/python scripts/make_features.py \
        --step "data/Xian Y-20 v3.step" --out "data/Xian Y-20 v3_features.stp" --n 45
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from OCC.Core.BRep import BRep_Builder, BRep_Tool
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
from brep_defeature.occ.features import FEATURE_KINDS, make_protrusion


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
    """对单个面取其三角网格 -> [(area, 质心 gp_Pnt, 外法向 gp_Dir)]。

    用三角形质心保证点**落在实际（已裁剪）蒙皮上**，不会跑到参数域裁剪区外
    （那正是之前凸起“悬空”的原因）。需事先对 shape 做过 BRepMesh。
    """
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
        cen = gp_Pnt((p1.X() + p2.X() + p3.X()) / 3,
                     (p1.Y() + p2.Y() + p3.Y()) / 3,
                     (p1.Z() + p2.Z() + p3.Z()) / 3)
        nrm = gp_Dir(cr)
        if rev:
            nrm.Reverse()
        out.append((0.5 * m, cen, nrm))
    return out


def heuristic_sites(shape, n=45, max_hosts=16):
    """启发式选位：近水平大面 + 偏外侧；点取自三角网格质心（保证在蒙皮上）。

    返回 (质心 gp_Pnt, 外法向 gp_Dir) 列表。
    """
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    ext = [xmax - xmin, ymax - ymin, zmax - zmin]
    center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    diag = math.sqrt(sum(e * e for e in ext))
    vert_axis = int(np.argmin(ext))
    horiz = [i for i in range(3) if i != vert_axis]
    span_axis = horiz[int(np.argmax([ext[i] for i in horiz]))]
    half_span = ext[span_axis] / 2

    BRepMesh_IncrementalMesh(shape, diag * 0.001, False, 0.5, True)

    # 每个面：累计面积 + 代表三角（最大）+ 全部三角样本
    reps = []
    for f in _faces(shape):
        samples = _triangle_samples(f)
        if not samples:
            continue
        samples.sort(key=lambda t: -t[0])
        total = sum(s[0] for s in samples)
        reps.append((total, f, samples[0], samples))
    if not reps:
        return []
    thr = np.percentile([r[0] for r in reps], 75)

    cand = []
    for total, f, rep, samples in reps:
        if total < thr:
            continue
        _, cen, nrm = rep
        if abs([nrm.X(), nrm.Y(), nrm.Z()][vert_axis]) < 0.6:   # 近水平
            continue
        coords = [cen.X(), cen.Y(), cen.Z()]
        span_off = abs(coords[span_axis] - center[span_axis]) / (half_span + 1e-9)
        if span_off < 0.2:                                      # 偏外侧
            continue
        cand.append((span_off * total, samples))
    cand.sort(key=lambda t: -t[0])
    hosts = cand[:max_hosts]

    per_face = max(1, n // max(len(hosts), 1))
    sites = []
    for _, samples in hosts:
        stride = max(1, len(samples) // per_face)               # 在面上铺开取点
        for _, cen, nrm in samples[::stride][:per_face]:
            sites.append((cen, nrm))
            if len(sites) >= n:
                break
        if len(sites) >= n:
            break
    return sites


def main():
    ap = argparse.ArgumentParser(description="【临时】生成微小凸起特征 STP（待 DL 替换）")
    ap.add_argument("--step", required=True)
    ap.add_argument("--out", required=True, help="输出凸起特征 STP")
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--hosts", type=int, default=16)
    ap.add_argument("--size-frac", type=float, default=0.004, help="凸起基准尺寸 / 包围盒对角线")
    args = ap.parse_args()

    shape = read_step(args.step)
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
    size = diag * args.size_frac

    sites = heuristic_sites(shape, n=args.n, max_hosts=args.hosts)
    if not sites:
        log("未生成任何凸起位置")
        sys.exit(1)

    comp = TopoDS_Compound()
    bld = BRep_Builder()
    bld.MakeCompound(comp)
    counts = {k: 0 for k in FEATURE_KINDS}
    for i, (pnt, normal) in enumerate(sites):
        kind = FEATURE_KINDS[i % len(FEATURE_KINDS)]  # 三类轮流（占位策略）
        try:
            solid = make_protrusion(kind, pnt, normal, size)
        except Exception as e:  # noqa: BLE001
            log(f"  跳过 1 个 {kind}: {type(e).__name__}")
            continue
        bld.Add(comp, solid)
        counts[kind] += 1

    writer = STEPControl_Writer()
    writer.Transfer(comp, STEPControl_AsIs)
    if writer.Write(args.out) != 1:
        raise IOError(f"写入失败: {args.out}")
    log(f"已写出凸起特征 -> {args.out}  (基准尺寸 {size:.3f}; 计数 {counts})")


if __name__ == "__main__":
    main()
