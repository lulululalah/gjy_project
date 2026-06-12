"""用 Rhino 放钉记录的 JSON，在 OCC 里稳健地把铆钉融入原模型 + 导出 + 截图。

为什么不在 Rhino 里融合：Rhino 的整体 BooleanUnion 对这些导入实体不稳定（会丢件）。
改为：Rhino 只负责交互选位、记录每钉的 中心/法向/形状/半径 到 JSON；这里用 OCC 的
逐 solid 布尔（已在 Y-20 上验证可靠）做真正的融合。

用法（OCC 环境）：
    /opt/anaconda3/envs/occ/bin/python scripts/fuse_from_records.py \
        --step "data/Xian Y-20 v3.step" \
        --records ~/Desktop/rhino_rivets.json \
        --out-step y20_fused.stp --out y20_fused.png

注意：JSON 里的坐标须与 --step 模型同一坐标系（即在该模型上放的钉）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.gp import gp_Dir, gp_Pnt

from brep_defeature.occ.extract import read_step
from brep_defeature.occ.features import make_protrusion
from inject_features_view import bbox, export_step, face_map, fuse_per_solid, render, solids, tessellate


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description="按 Rhino 记录 JSON 在 OCC 里稳健融合铆钉")
    ap.add_argument("--step", required=True, help="原始模型 STP（与记录同坐标系）")
    ap.add_argument("--records", required=True, help="Rhino 放钉记录 JSON")
    ap.add_argument("--out-step", default=None)
    ap.add_argument("--out", default="fused.png")
    args = ap.parse_args()

    shape = read_step(args.step)
    air = solids(shape)
    log(f"原模型 solid 数 {len(air)}，面数 {face_map(shape).Size()}")

    with open(args.records, "r", encoding="utf-8") as f:
        recs = json.load(f)["rivets"]
    log(f"读取放钉记录 {len(recs)} 个 <- {args.records}")

    comp = TopoDS_Compound()
    bld = BRep_Builder()
    bld.MakeCompound(comp)
    built = 0
    for r in recs:
        c, n = r["center"], r["normal"]
        rad = float(r.get("radius", 0.0))
        kind = r.get("shape", "dome")
        kind = {"dome": "hemisphere"}.get(kind, kind)  # Rhino 用 dome，OCC 用 hemisphere
        try:
            solid = make_protrusion(kind, gp_Pnt(*c), gp_Dir(*n), rad)
            bld.Add(comp, solid)
            built += 1
        except Exception as e:  # noqa: BLE001
            log(f"  跳过 1 个 {kind}: {type(e).__name__}: {e}")
    log(f"构造铆钉实体 {built} 个")

    if air:
        result, n_applied = fuse_per_solid(air, comp)
        n_out = len(solids(result))
        log(f"逐 solid 融合：solid {len(air)} -> {n_out}（应相等），施加铆钉 {n_applied}")
    else:
        # 模型是开壳（无 solid）：整体 Fuse 兜底
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse

        log("模型无 solid（开壳），改为整体 Fuse")
        fuse = BRepAlgoAPI_Fuse(shape, comp)
        fuse.Build()
        result = fuse.Shape() if fuse.IsDone() else shape
        n_applied = built if fuse.IsDone() else 0

    if args.out_step:
        export_step(result, args.out_step)
        log(f"融合后模型 -> {args.out_step}")

    g = bbox(shape).Get()
    ext = [g[3] - g[0], g[4] - g[1], g[5] - g[2]]
    center = [(g[0] + g[3]) / 2, (g[1] + g[4]) / 2, (g[2] + g[5]) / 2]
    vert_axis = int(np.argmin(ext))
    diag = math.sqrt(sum(e * e for e in ext))
    verts, tris, tri_rivet = tessellate(result, diag * 0.0008)
    render(verts, tris, tri_rivet, args.out, center, diag, vert_axis)
    log(f"截图 -> {args.out}")


if __name__ == "__main__":
    main()
