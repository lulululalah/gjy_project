"""据 Rhino 记录的 JSON，在 OCC 里把铆钉锚点【压印】进 BRep 拓扑（不做布尔）。

原理（参数域 + 拓扑，正是所需）：
  对每个记录点，找到它所在的面；把锚点投影得到该面的参数 (u,v)；在**面的参数域**里
  画一个半径对应世界尺度的小圆(Geom2d_Circle)，据此在该面上造一条 pcurve 边 -> 闭合 wire；
  用 BRepFeat_SplitShape 把该面按这个 wire 切开 -> 面上多出一个圆盘子面 + 内环。
  全程无布尔、无悬空体；子面/内环直接进入 BRep 拓扑，且必落在面上。

之后：每个铆钉即一个圆盘子面，可被识别(GNN)、可被去除(删子面并补平)。

用法（OCC 环境）：
    /opt/anaconda3/envs/occ/bin/python scripts/imprint_from_records.py \
        --step "data/747sp.stp" --records ~/Desktop/rhino_rivets.json \
        --out-step y747_imprinted.stp
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
)
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepFeat import BRepFeat_SplitShape
from OCC.Core.BRepLib import breplib
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.Geom2d import Geom2d_Circle
from OCC.Core.gp import gp_Ax2d, gp_Dir2d, gp_Pnt, gp_Pnt2d
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TopoDS import topods

from brep_defeature.occ.extract import build_face_map, read_step


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def nearest_face_uv(faces, shape, c):
    """返回 (face_index1based, face, u, v)：离点 c 最近的面及其参数。"""
    target = gp_Pnt(*c)
    v = BRepBuilderAPI_MakeVertex(target).Vertex()
    best = None
    for i, face in enumerate(faces, start=1):
        d = BRepExtrema_DistShapeShape(v, face)
        if d.IsDone() and d.NbSolution() >= 1:
            val = d.Value()
            if best is None or val < best[0]:
                best = (val, i, face)
    if best is None:
        return None
    _, fid, face = best
    surf = BRep_Tool.Surface(face)
    proj = GeomAPI_ProjectPointOnSurf(target, surf)
    if proj.NbPoints() < 1:
        return None
    u, vv = proj.LowerDistanceParameters()
    return fid, face, u, vv


def make_uv_circle_wire(face, u, v, world_radius):
    """在面的参数域内造一个半径≈world_radius 的圆，返回贴在面上的 wire。"""
    surf = BRep_Tool.Surface(face)
    props = BRepLProp_SLProps(BRepAdaptor_Surface(face), u, v, 1, 1e-6)
    du = props.D1U().Magnitude() if props.IsTangentUDefined() else 1.0
    dv = props.D1V().Magnitude() if props.IsTangentVDefined() else 1.0
    scale = 0.5 * (du + dv)
    if scale < 1e-9:
        scale = 1.0
    r_uv = world_radius / scale
    circ2d = Geom2d_Circle(gp_Ax2d(gp_Pnt2d(u, v), gp_Dir2d(1.0, 0.0)), r_uv)
    edge = BRepBuilderAPI_MakeEdge(circ2d, surf).Edge()
    breplib.BuildCurves3d(edge)                       # 据 pcurve 生成 3D 曲线
    return BRepBuilderAPI_MakeWire(edge).Wire()


def main():
    ap = argparse.ArgumentParser(description="据记录 JSON 在 OCC 里压印铆钉(无布尔)")
    ap.add_argument("--step", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out-step", required=True)
    args = ap.parse_args()

    shape = read_step(args.step)
    fmap = build_face_map(shape)
    faces = [topods.Face(fmap.FindKey(i)) for i in range(1, fmap.Size() + 1)]
    n0 = fmap.Size()
    log(f"原模型面数 {n0}")

    with open(args.records, "r", encoding="utf-8") as f:
        recs = json.load(f)["rivets"]
    log(f"记录 {len(recs)} 个 <- {args.records}")

    splitter = BRepFeat_SplitShape(shape)
    added = 0
    for rec in recs:
        c = rec["center"]
        r = float(rec.get("radius", 0.0))
        got = nearest_face_uv(faces, shape, c)
        if got is None:
            continue
        _, face, u, v = got
        try:
            wire = make_uv_circle_wire(face, u, v, r)
            splitter.Add(wire, face)
            added += 1
        except Exception as e:  # noqa: BLE001
            log(f"  跳过 1 个: {type(e).__name__}: {e}")
    log(f"加入压印 wire {added} 个")

    splitter.Build()
    if not splitter.IsDone():
        log("SplitShape 未完成，退出")
        sys.exit(1)
    result = splitter.Shape()
    n1 = build_face_map(result).Size()
    log(f"压印完成：面数 {n0} -> {n1}（应 > 原值，多出的即铆钉圆盘子面）")

    w = STEPControl_Writer()
    w.Transfer(result, STEPControl_AsIs)
    if w.Write(args.out_step) != 1:
        raise IOError("写出失败")
    log(f"已导出 -> {args.out_step}")


if __name__ == "__main__":
    main()
