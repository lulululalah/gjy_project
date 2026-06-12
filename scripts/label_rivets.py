"""把已生成铆钉的模型标注出来，产出深度学习的「预先已知真值」。

原理（注入即真值）：飞机蒙皮没有球面，所以融合后模型里**凡是球面 = 半球铆钉**。
据此把每个面打上三层标签，并按图连通性聚成实例：
    semantic   : RIVET / BACKGROUND
    instance_id: 每个铆钉（连通的铆钉面）一个 id
    operation  : REMOVE_PROTRUSION / KEEP

输出：
    <模型>_labeled.json  —— 带标签的 FaceGraph（可直接喂给 train/infer，是 GNN 的真值）
    <模型>_labels.csv    —— 逐面标签，便于人工核对
可选 --png 渲染：高亮被标为铆钉的面，用于肉眼确认标注正确。

⚠️ 需要 OCC 环境：
    /opt/anaconda3/envs/occ/bin/python scripts/label_rivets.py y20_features.stp
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brep_defeature.occ.extract import step_to_graph
from brep_defeature.schema import SURFACE_TYPE_INDEX, Operation, Semantic


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def label_rivets(graph, rivet_surface="sphere"):
    """按曲面类型标注铆钉面 + 连通聚实例。原地修改并返回 graph。"""
    target = SURFACE_TYPE_INDEX[rivet_surface]
    idx = graph.id_to_index()

    is_rivet = [int(n.feats.get("surface_type", -1)) == target for n in graph.nodes]

    # 连通分量：相邻且同为铆钉面的归一个实例
    uf = _UF(len(graph.nodes))
    for e in graph.edges:
        a, b = idx.get(e.src), idx.get(e.dst)
        if a is not None and b is not None and is_rivet[a] and is_rivet[b]:
            uf.union(a, b)

    inst_of = {}
    next_id = 0
    for i, n in enumerate(graph.nodes):
        if is_rivet[i]:
            root = uf.find(i)
            if root not in inst_of:
                inst_of[root] = next_id
                next_id += 1
            n.semantic = int(Semantic.RIVET)
            n.instance_id = inst_of[root]
            n.operation = int(Operation.REMOVE_PROTRUSION)
        else:
            n.semantic = int(Semantic.BACKGROUND)
            n.instance_id = -1
            n.operation = int(Operation.KEEP)

    # 边的 same_instance 真值
    for e in graph.edges:
        a, b = idx.get(e.src), idx.get(e.dst)
        e.same_instance = int(
            a is not None and b is not None
            and is_rivet[a] and is_rivet[b]
            and graph.nodes[a].instance_id == graph.nodes[b].instance_id
        )

    n_rivet = sum(is_rivet)
    return graph, n_rivet, next_id


def write_csv(graph, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["face_id", "semantic", "instance_id", "operation", "surface_type"])
        for n in graph.nodes:
            w.writerow([n.face_id, n.semantic, n.instance_id, n.operation,
                        int(n.feats.get("surface_type", -1))])


def main():
    ap = argparse.ArgumentParser(description="标注铆钉面，产出 GNN 真值（FaceGraph JSON）")
    ap.add_argument("step", help="已融合铆钉的 STP")
    ap.add_argument("--surface", default="sphere", choices=["sphere", "cone"],
                    help="铆钉对应的曲面类型（半球=sphere，梯台=cone）")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--png", default=None, help="可选：高亮标注结果并截图")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.step))[0]
    out_json = args.out_json or f"{stem}_labeled.json"
    out_csv = args.out_csv or f"{stem}_labels.csv"

    print(f"读取 {args.step} ...", file=sys.stderr)
    graph = step_to_graph(args.step, model_id=stem)
    graph, n_rivet, n_inst = label_rivets(graph, rivet_surface=args.surface)

    graph.to_json(out_json)
    write_csv(graph, out_csv)
    print(f"面数 {graph.num_nodes}，标为铆钉的面 {n_rivet}，铆钉实例 {n_inst}")
    print(f"真值 FaceGraph -> {out_json}")
    print(f"逐面标签   -> {out_csv}")

    if args.png:
        _render_labeled(args.step, graph, args.png)
        print(f"标注可视化 -> {args.png}")


def _render_labeled(step_path, graph, png):
    """高亮被标为铆钉的面（按 face_id 对齐三角网格）。"""
    import numpy as np
    from inject_features_view import bbox, render, tessellate
    from brep_defeature.occ.extract import read_step

    shape = read_step(step_path)
    g = bbox(shape).Get()
    ext = [g[3] - g[0], g[4] - g[1], g[5] - g[2]]
    center = [(g[0] + g[3]) / 2, (g[1] + g[4]) / 2, (g[2] + g[5]) / 2]
    vert_axis = int(np.argmin(ext))
    diag = float(np.sqrt(sum(e * e for e in ext)))

    rivet_fids = {n.face_id for n in graph.nodes if n.semantic == int(Semantic.RIVET)}
    verts, tris, _ = tessellate(shape, diag * 0.0008)
    # tessellate 的 tri_face 与 extract 的 face_id 同序（都按 face map 1-based）
    from inject_features_view import face_map
    fmap = face_map(shape)
    # 重新取每个三角所属 face_id
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods
    tri_rivet = []
    for fid in range(1, fmap.Size() + 1):
        face = topods.Face(fmap.FindKey(fid))
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None:
            continue
        tri_rivet.extend([fid in rivet_fids] * tri.NbTriangles())
    render(verts, tris, np.asarray(tri_rivet, bool), png, center, diag, vert_axis)


if __name__ == "__main__":
    main()
