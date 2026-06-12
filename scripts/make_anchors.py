"""【临时脚手架】生成铆钉注入锚点 STP。

⚠️ 这是启发式占位实现，仅用于在 DL 模型就绪前产出一份示例锚点文件。
   后期将由「纯深度学习预测的铆钉位置」替换本脚本，注入管线（inject_rivets_view.py）
   读取的锚点 STP 接口保持不变。

启发式规则（将被废弃）：选近水平的大面、偏外侧区域，在其参数域边界附近取点。

用法：
    /opt/anaconda3/envs/occ/bin/python scripts/make_anchors.py \
        --step "data/Xian Y-20 v3.step" --out "data/Xian Y-20 v3_anchors.stp" --n 44
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods

from brep_defeature.occ.anchors import write_anchors_stp
from brep_defeature.occ.extract import read_step


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


def _centroid_normal(face):
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


def _seam_points(face, n_per_face):
    """参数域靠近 4 条边界 + 中心处取点（近似接缝）。"""
    ad = BRepAdaptor_Surface(face)
    u0, u1 = ad.FirstUParameter(), ad.LastUParameter()
    v0, v1 = ad.FirstVParameter(), ad.LastVParameter()
    fracs = [(0.12, 0.5), (0.88, 0.5), (0.5, 0.12), (0.5, 0.88), (0.5, 0.5)]
    pts = []
    for fu, fv in fracs[: n_per_face + 1]:
        u = u0 + fu * (u1 - u0)
        v = v0 + fv * (v1 - v0)
        try:
            props = BRepLProp_SLProps(ad, u, v, 1, 1e-6)
            if props.IsNormalDefined():
                p = props.Value()
                pts.append((p.X(), p.Y(), p.Z()))
        except RuntimeError:
            continue
    return pts


def heuristic_anchor_points(shape, n=44, max_hosts=16):
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
        p, nrm = _centroid_normal(f)
        if p is None:
            continue
        coords = [p.X(), p.Y(), p.Z()]
        if abs([nrm.X(), nrm.Y(), nrm.Z()][vert_axis]) < 0.6:
            continue
        span_off = abs(coords[span_axis] - center[span_axis]) / (half_span + 1e-9)
        if span_off < 0.2:
            continue
        cand.append((span_off * a, f))
    cand.sort(key=lambda t: -t[0])
    hosts = [c[1] for c in cand[:max_hosts]]

    per_face = max(1, n // max(len(hosts), 1))
    pts = []
    for f in hosts:
        pts.extend(_seam_points(f, per_face))
        if len(pts) >= n:
            break
    return pts[:n]


def main():
    ap = argparse.ArgumentParser(description="【临时】生成铆钉注入锚点 STP（待 DL 替换）")
    ap.add_argument("--step", required=True)
    ap.add_argument("--out", required=True, help="输出锚点 STP 路径")
    ap.add_argument("--n", type=int, default=44)
    ap.add_argument("--hosts", type=int, default=16)
    args = ap.parse_args()

    shape = read_step(args.step)
    pts = heuristic_anchor_points(shape, n=args.n, max_hosts=args.hosts)
    if not pts:
        log("未生成任何锚点")
        sys.exit(1)
    write_anchors_stp(pts, args.out)
    log(f"已写出 {len(pts)} 个锚点 -> {args.out}")


if __name__ == "__main__":
    main()
