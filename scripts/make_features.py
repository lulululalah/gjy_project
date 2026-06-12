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

from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.GProp import GProp_GProps
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound, topods

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


def _area(face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return props.Mass()


def _props_at(face, fu, fv):
    ad = BRepAdaptor_Surface(face)
    u = ad.FirstUParameter() + fu * (ad.LastUParameter() - ad.FirstUParameter())
    v = ad.FirstVParameter() + fv * (ad.LastVParameter() - ad.FirstVParameter())
    props = BRepLProp_SLProps(ad, u, v, 1, 1e-6)
    if not props.IsNormalDefined():
        return None, None
    p = props.Value()
    n = props.Normal()
    if face.Orientation() == 1:
        n.Reverse()
    return p, n


def heuristic_sites(shape, n=45, max_hosts=16):
    """启发式：近水平大面 + 偏外侧；在参数边界（≈接缝）取点。返回 (pnt, normal) 列表。"""
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    ext = [xmax - xmin, ymax - ymin, zmax - zmin]
    center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    vert_axis = int(np.argmin(ext))
    horiz = [i for i in range(3) if i != vert_axis]
    span_axis = horiz[int(np.argmax([ext[i] for i in horiz]))]
    half_span = ext[span_axis] / 2

    fs = _faces(shape)
    areas = [_area(f) for f in fs]
    if not areas:
        return []
    thr = np.percentile(areas, 75)
    cand = []
    for f, a in zip(fs, areas):
        if a < thr:
            continue
        p, nrm = _props_at(f, 0.5, 0.5)
        if p is None:
            continue
        if abs([nrm.X(), nrm.Y(), nrm.Z()][vert_axis]) < 0.6:
            continue
        coords = [p.X(), p.Y(), p.Z()]
        span_off = abs(coords[span_axis] - center[span_axis]) / (half_span + 1e-9)
        if span_off < 0.2:
            continue
        cand.append((span_off * a, f))
    cand.sort(key=lambda t: -t[0])
    hosts = [c[1] for c in cand[:max_hosts]]

    fracs = [(0.15, 0.5), (0.85, 0.5), (0.5, 0.15), (0.5, 0.85), (0.5, 0.5)]
    per_face = max(1, n // max(len(hosts), 1))
    sites = []
    for f in hosts:
        for fu, fv in fracs[:per_face]:
            p, nrm = _props_at(f, fu, fv)
            if p is not None:
                sites.append((p, nrm))
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
