"""OCC-free 合成面图生成器。

不依赖 OpenCASCADE，纯程序化地构造「背景面网格 + 植入的微小特征」面图，
并直接给出三层真值标签。用途：

1. 在没有 OCC 的机器上端到端开发/测试整个 ML 管线。
2. 提供「注入即真值」范式的快速验证：合成器扮演 occ.inject 的角色，
   产出与真实数据相同的 FaceGraph 契约。

设计要点：单个特征**不能仅靠单面阈值区分**——特征面的尺度/曲率与背景有重叠，
必须结合边的拓扑上下文（凸/凹、面积比）才能稳定区分，从而真正考验图/窗口机制。
"""

from __future__ import annotations

import math
import random
from typing import Optional

from ..schema import (
    CURVE_TYPE_INDEX,
    SURFACE_TYPE_INDEX,
    EdgeRecord,
    FaceGraph,
    FaceNode,
    Operation,
    Semantic,
)


# --------------------------------------------------------------------------- #
# 构图累加器
# --------------------------------------------------------------------------- #
class _Builder:
    def __init__(self, model_id: str, rng: random.Random):
        self.model_id = model_id
        self.rng = rng
        self.nodes: dict[int, FaceNode] = {}
        self.edges: list[EdgeRecord] = []
        self._next_id = 0
        self._next_instance = 0

    def add_face(self, **kw) -> int:
        fid = self._next_id
        self._next_id += 1
        node = FaceNode(
            face_id=fid,
            feats=kw.pop("feats"),
            semantic=int(kw.pop("semantic", Semantic.BACKGROUND)),
            instance_id=int(kw.pop("instance_id", -1)),
            operation=int(kw.pop("operation", Operation.KEEP)),
        )
        self.nodes[fid] = node
        return fid

    def add_edge(self, a: int, b: int, feats: dict, same_instance: int = 0) -> None:
        self.edges.append(EdgeRecord(src=a, dst=b, feats=feats, same_instance=same_instance))

    def new_instance(self) -> int:
        i = self._next_instance
        self._next_instance += 1
        return i

    def finalize(self) -> FaceGraph:
        # rel_area（相对总面积）与 neighbor_area_ratio（局部对比）后处理
        total_area = sum(n.feats["_abs_area"] for n in self.nodes.values()) or 1.0
        neigh: dict[int, list[int]] = {fid: [] for fid in self.nodes}
        for e in self.edges:
            neigh[e.src].append(e.dst)
            neigh[e.dst].append(e.src)
        for fid, node in self.nodes.items():
            node.feats["rel_area"] = node.feats["_abs_area"] / total_area
            areas = [self.nodes[m].feats["_abs_area"] for m in neigh[fid]]
            mean_n = sum(areas) / len(areas) if areas else node.feats["_abs_area"]
            ratio = node.feats["_abs_area"] / mean_n if mean_n > 1e-9 else 1.0
            node.feats["neighbor_area_ratio"] = ratio
        for node in self.nodes.values():
            node.feats.pop("_abs_area", None)
        return FaceGraph(
            model_id=self.model_id,
            nodes=list(self.nodes.values()),
            edges=self.edges,
            meta={"synthetic": True},
        )


# --------------------------------------------------------------------------- #
# 特征零件
# --------------------------------------------------------------------------- #
def _jit(rng: random.Random, base: float, frac: float) -> float:
    return base * (1.0 + rng.uniform(-frac, frac))


def _plane_face_feats(rng: random.Random, abs_area: float) -> dict:
    nz = _jit(rng, 1.0, 0.02)
    nx = rng.uniform(-0.05, 0.05)
    ny = rng.uniform(-0.05, 0.05)
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    return {
        "_abs_area": abs_area,
        "log_area": math.log1p(abs_area),
        "compactness": _jit(rng, 1.3, 0.2),
        "nx": nx / norm, "ny": ny / norm, "nz": nz / norm,
        "mean_curvature": abs(rng.gauss(0.0, 0.01)),
        "gauss_curvature": abs(rng.gauss(0.0, 0.005)),
        "radius": 0.0,
        "min_bbox_dim": _jit(rng, math.sqrt(abs_area), 0.3),
        "num_wires": 1,
        "inner_wire_count": 0,
        "min_inner_wire_len": 0.0,
        "max_inner_wire_len": 0.0,
        "num_edges": 4,
        "surface_type": SURFACE_TYPE_INDEX["plane"],
    }


def _curved_small_feats(rng: random.Random, surf: str, radius: float) -> dict:
    abs_area = _jit(rng, 2 * math.pi * radius * radius, 0.4)
    nx, ny, nz = (rng.gauss(0, 1) for _ in range(3))
    norm = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    return {
        "_abs_area": abs_area,
        "log_area": math.log1p(abs_area),
        "compactness": _jit(rng, 1.1, 0.25),
        "nx": nx / norm, "ny": ny / norm, "nz": nz / norm,
        "mean_curvature": _jit(rng, 1.0 / (2 * radius), 0.2),
        "gauss_curvature": _jit(rng, 1.0 / (radius * radius), 0.3) if surf == "sphere" else abs(rng.gauss(0, 0.02)),
        "radius": _jit(rng, radius, 0.15),
        "min_bbox_dim": _jit(rng, radius, 0.3),
        "num_wires": 1,
        "inner_wire_count": 0,
        "min_inner_wire_len": 0.0,
        "max_inner_wire_len": 0.0,
        "num_edges": rng.randint(2, 4),
        "surface_type": SURFACE_TYPE_INDEX[surf],
    }


def _edge_feats(rng: random.Random, kind: str, area_ratio: float, shared_len: float) -> dict:
    """kind ∈ {smooth, convex, concave}。"""
    if kind == "convex":
        et, dih, curve = 1.0, _jit(rng, 1.2, 0.3), "circle"
    elif kind == "concave":
        et, dih, curve = -1.0, -_jit(rng, 1.2, 0.3), "circle"
    else:
        et, dih, curve = 0.0, rng.gauss(0.0, 0.05), "line"
    return {
        "edge_type": et,
        "dihedral_angle": dih,
        "shared_len": shared_len,
        "rel_shared_len": shared_len / 40.0,
        "area_ratio": area_ratio,
        "curve_type": CURVE_TYPE_INDEX[curve],
    }


def _add_rivet(b: _Builder, host: int) -> None:
    """凸起：1 顶面(球冠) + 2~4 侧面(圆柱)，相互凸连，整体凸连到宿主。"""
    inst = b.new_instance()
    radius = b.rng.uniform(0.15, 0.45)
    host_area = b.nodes[host].feats["_abs_area"]

    cap = b.add_face(
        feats=_curved_small_feats(b.rng, "sphere", radius),
        semantic=Semantic.RIVET, instance_id=inst, operation=Operation.REMOVE_PROTRUSION,
    )
    sides = []
    for _ in range(b.rng.randint(2, 4)):
        s = b.add_face(
            feats=_curved_small_feats(b.rng, "cylinder", radius),
            semantic=Semantic.RIVET, instance_id=inst, operation=Operation.REMOVE_PROTRUSION,
        )
        sides.append(s)

    small_area = b.nodes[cap].feats["_abs_area"]
    # cap - side（实例内，凸）
    for s in sides:
        b.add_edge(cap, s, _edge_feats(b.rng, "convex", 0.8, _jit(b.rng, radius, 0.3)), same_instance=1)
    # side - side（实例内，凸）
    for i in range(len(sides)):
        b.add_edge(sides[i], sides[(i + 1) % len(sides)],
                   _edge_feats(b.rng, "convex", 1.0, _jit(b.rng, radius, 0.3)), same_instance=1)
    # side - host（跨实例边界，凸，面积比悬殊）
    for s in sides:
        b.add_edge(s, host,
                   _edge_feats(b.rng, "convex", small_area / host_area, _jit(b.rng, 2 * math.pi * radius, 0.3)),
                   same_instance=0)


def _add_hole(b: _Builder, host: int) -> None:
    """孔：圆柱内壁(1~3) + 可选底面，内壁凹连到宿主；宿主内环数 +1。"""
    inst = b.new_instance()
    radius = b.rng.uniform(0.2, 0.5)
    host_node = b.nodes[host]
    host_area = host_node.feats["_abs_area"]

    walls = []
    for _ in range(b.rng.randint(1, 3)):
        w = b.add_face(
            feats=_curved_small_feats(b.rng, "cylinder", radius),
            semantic=Semantic.HOLE, instance_id=inst, operation=Operation.FILL_HOLE,
        )
        walls.append(w)
    bottom = None
    if b.rng.random() < 0.5:  # 盲孔
        bf = _curved_small_feats(b.rng, "plane", radius)
        bf["surface_type"] = SURFACE_TYPE_INDEX["plane"]
        bottom = b.add_face(feats=bf, semantic=Semantic.HOLE, instance_id=inst, operation=Operation.FILL_HOLE)

    small_area = b.nodes[walls[0]].feats["_abs_area"]
    circ = 2 * math.pi * radius
    for i in range(len(walls)):
        b.add_edge(walls[i], walls[(i + 1) % len(walls)],
                   _edge_feats(b.rng, "smooth", 1.0, _jit(b.rng, radius, 0.3)), same_instance=1)
    if bottom is not None:
        for w in walls:
            b.add_edge(w, bottom, _edge_feats(b.rng, "concave", 0.9, _jit(b.rng, radius, 0.3)), same_instance=1)
    # 内壁 - 宿主（跨边界，凹）
    for w in walls:
        b.add_edge(w, host, _edge_feats(b.rng, "concave", small_area / host_area, _jit(b.rng, circ, 0.3)),
                   same_instance=0)

    # 宿主内环 +1
    host_node.feats["inner_wire_count"] += 1
    host_node.feats["num_wires"] += 1
    host_node.feats["min_inner_wire_len"] = circ
    host_node.feats["max_inner_wire_len"] = max(host_node.feats["max_inner_wire_len"], circ)


# --------------------------------------------------------------------------- #
# 顶层 API
# --------------------------------------------------------------------------- #
def synth_model(
    model_id: str,
    rng: random.Random,
    grid: int = 5,
    n_rivets: int = 4,
    n_holes: int = 3,
) -> FaceGraph:
    """生成一张含特征的面图。

    背景为 grid×grid 的平面面网格（4-邻接），随机选宿主面植入铆钉/孔。
    """
    b = _Builder(model_id, rng)

    # 背景网格
    cell_area = rng.uniform(60.0, 140.0)
    ids = [[b.add_face(feats=_plane_face_feats(rng, _jit(rng, cell_area, 0.2)),
                       semantic=Semantic.BACKGROUND) for _ in range(grid)] for _ in range(grid)]
    for r in range(grid):
        for c in range(grid):
            if c + 1 < grid:
                b.add_edge(ids[r][c], ids[r][c + 1],
                           _edge_feats(rng, "smooth", 1.0, _jit(rng, math.sqrt(cell_area), 0.15)))
            if r + 1 < grid:
                b.add_edge(ids[r][c], ids[r + 1][c],
                           _edge_feats(rng, "smooth", 1.0, _jit(rng, math.sqrt(cell_area), 0.15)))

    flat = [ids[r][c] for r in range(grid) for c in range(grid)]
    for _ in range(n_rivets):
        _add_rivet(b, rng.choice(flat))
    for _ in range(n_holes):
        _add_hole(b, rng.choice(flat))

    return b.finalize()


def synth_dataset(n_models: int = 40, seed: int = 0, **kw) -> list[FaceGraph]:
    """生成一批互不相同的合成模型。"""
    master = random.Random(seed)
    graphs = []
    for i in range(n_models):
        rng = random.Random(master.randint(0, 2**31 - 1))
        grid = kw.get("grid", rng.randint(4, 6))
        n_rivets = kw.get("n_rivets", rng.randint(2, 6))
        n_holes = kw.get("n_holes", rng.randint(1, 5))
        graphs.append(synth_model(f"synth_{i:04d}", rng, grid=grid, n_rivets=n_rivets, n_holes=n_holes))
    return graphs
