"""数据契约：FaceGraph + 三层标签。

这是几何后端（OCC）与 ML 管线之间唯一的接口。无论 FaceGraph 来自真实 STEP
模型（occ.extract）还是合成生成器（synth），下游训练/推理代码都只依赖本模块。

三层标签对应 GNN 的三个任务：
- semantic:  面的特征类别（背景 / 铆钉 / 孔 / ...）
- instance:  同一物理特征的面共享一个实例 id（-1 表示背景，不属于任何实例）
- operation: 该面对应的去除操作类型（保留 / 删凸起 / 填孔 / ...）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# 枚举：语义类别、几何类型、操作类型
# --------------------------------------------------------------------------- #
class Semantic(IntEnum):
    BACKGROUND = 0
    RIVET = 1
    HOLE = 2
    WINDOW = 3
    DECAL = 4
    FILLET = 5


class Operation(IntEnum):
    KEEP = 0
    REMOVE_PROTRUSION = 1  # 铆钉 / 凸台：删除凸起并补平
    FILL_HOLE = 2          # 孔 / 窗口：删除内壁面并封闭宿主内环
    MERGE_DECAL = 3        # 贴花：与宿主面合并
    REMOVE_FILLET = 4      # 小倒角：删除过渡面并延伸相邻面求交


# 几何类型与 OCC GeomAbs_SurfaceType / GeomAbs_CurveType 对齐的精简集合。
SURFACE_TYPES = ["plane", "cylinder", "cone", "sphere", "torus", "bspline", "other"]
CURVE_TYPES = ["line", "circle", "ellipse", "bspline", "other"]

SURFACE_TYPE_INDEX = {name: i for i, name in enumerate(SURFACE_TYPES)}
CURVE_TYPE_INDEX = {name: i for i, name in enumerate(CURVE_TYPES)}


# --------------------------------------------------------------------------- #
# 特征列定义（单一事实来源）
#
# 连续特征参与归一化；类别特征（surface_type / curve_type）在 to_pyg 时 one-hot，
# 不参与归一化。节点/边的最终特征矩阵 = [归一化连续块 | one-hot 类别块]。
# --------------------------------------------------------------------------- #
NODE_CONT_FEATURES = [
    "rel_area",            # 面积 / 模型总面积
    "log_area",            # log(1+绝对面积)，提供尺度信息
    "compactness",         # 周长^2 / (4*pi*面积)
    "nx", "ny", "nz",      # 中心点法向
    "mean_curvature",
    "gauss_curvature",
    "radius",              # 圆柱/球/锥/环面半径，平面为 0
    "min_bbox_dim",        # 包围盒最小边长（对微小特征敏感）
    "num_wires",
    "inner_wire_count",
    "min_inner_wire_len",
    "max_inner_wire_len",
    "num_edges",
    "neighbor_area_ratio", # 本面面积 / 邻居面积均值（局部对比）
]
NODE_CAT_FEATURES = [("surface_type", len(SURFACE_TYPES))]

EDGE_CONT_FEATURES = [
    "edge_type",       # 凸=+1 凹=-1 光滑=0（离散但作连续输入）
    "dihedral_angle",  # 二面角（弧度，带符号）
    "shared_len",      # 共享边长度（绝对）
    "rel_shared_len",  # 共享边长度（相对模型尺度）
    "area_ratio",      # 边两侧面积 min/max
]
EDGE_CAT_FEATURES = [("curve_type", len(CURVE_TYPES))]

# 对外暴露的“完整列名”（连续 + 类别原始值），合成器/提取器需要全部填充。
NODE_FEATURES = NODE_CONT_FEATURES + [name for name, _ in NODE_CAT_FEATURES]
EDGE_FEATURES = EDGE_CONT_FEATURES + [name for name, _ in EDGE_CAT_FEATURES]

NODE_FEATURE_DIM = len(NODE_CONT_FEATURES) + sum(c for _, c in NODE_CAT_FEATURES)
EDGE_FEATURE_DIM = len(EDGE_CONT_FEATURES) + sum(c for _, c in EDGE_CAT_FEATURES)


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class FaceNode:
    """一个 BRep 面对应一个图节点。"""

    face_id: int
    feats: dict = field(default_factory=dict)  # 见 NODE_FEATURES
    semantic: int = int(Semantic.BACKGROUND)
    instance_id: int = -1
    operation: int = int(Operation.KEEP)

    def vector(self) -> list[float]:
        """按 NODE_CONT_FEATURES + 类别原值 的顺序取出（类别留作原始索引）。"""
        cont = [float(self.feats.get(name, 0.0)) for name in NODE_CONT_FEATURES]
        cat = [float(self.feats.get(name, 0.0)) for name, _ in NODE_CAT_FEATURES]
        return cont + cat


@dataclass
class EdgeRecord:
    """一条 BRep edge 对应图中一条无向边（存为两条有向边）。"""

    src: int  # face_id
    dst: int  # face_id
    feats: dict = field(default_factory=dict)  # 见 EDGE_FEATURES
    same_instance: int = 0  # 真值：两端面是否属于同一特征实例（实例边界头的标签）

    def vector(self) -> list[float]:
        cont = [float(self.feats.get(name, 0.0)) for name in EDGE_CONT_FEATURES]
        cat = [float(self.feats.get(name, 0.0)) for name, _ in EDGE_CAT_FEATURES]
        return cont + cat


@dataclass
class FaceGraph:
    """一个模型 = 一张面图。"""

    model_id: str
    nodes: list[FaceNode] = field(default_factory=list)
    edges: list[EdgeRecord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # -- 便捷访问 ----------------------------------------------------------- #
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def id_to_index(self) -> dict[int, int]:
        return {n.face_id: i for i, n in enumerate(self.nodes)}

    # -- JSON 序列化 -------------------------------------------------------- #
    def to_json(self, path: str) -> None:
        payload = {
            "model_id": self.model_id,
            "meta": self.meta,
            "nodes": [
                {
                    "face_id": n.face_id,
                    "feats": n.feats,
                    "semantic": int(n.semantic),
                    "instance_id": int(n.instance_id),
                    "operation": int(n.operation),
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "feats": e.feats,
                    "same_instance": int(e.same_instance),
                }
                for e in self.edges
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "FaceGraph":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        nodes = [
            FaceNode(
                face_id=n["face_id"],
                feats=n.get("feats", {}),
                semantic=n.get("semantic", 0),
                instance_id=n.get("instance_id", -1),
                operation=n.get("operation", 0),
            )
            for n in payload["nodes"]
        ]
        edges = [
            EdgeRecord(
                src=e["src"],
                dst=e["dst"],
                feats=e.get("feats", {}),
                same_instance=e.get("same_instance", 0),
            )
            for e in payload["edges"]
        ]
        return cls(
            model_id=payload["model_id"],
            nodes=nodes,
            edges=edges,
            meta=payload.get("meta", {}),
        )

    # -- 数值矩阵（供归一化统计与 PyG 转换使用） --------------------------- #
    def node_cont_matrix(self) -> np.ndarray:
        """仅连续特征块，形状 [N, len(NODE_CONT_FEATURES)]。"""
        if not self.nodes:
            return np.zeros((0, len(NODE_CONT_FEATURES)), dtype=np.float32)
        rows = [[float(n.feats.get(name, 0.0)) for name in NODE_CONT_FEATURES] for n in self.nodes]
        return np.asarray(rows, dtype=np.float32)

    def edge_cont_matrix(self) -> np.ndarray:
        if not self.edges:
            return np.zeros((0, len(EDGE_CONT_FEATURES)), dtype=np.float32)
        rows = [[float(e.feats.get(name, 0.0)) for name in EDGE_CONT_FEATURES] for e in self.edges]
        return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------------------- #
# 归一化统计
# --------------------------------------------------------------------------- #
@dataclass
class NormStats:
    """连续特征的均值/标准差，仅作用于连续块。"""

    node_mean: np.ndarray
    node_std: np.ndarray
    edge_mean: np.ndarray
    edge_std: np.ndarray

    @classmethod
    def fit(cls, graphs: list[FaceGraph], eps: float = 1e-6) -> "NormStats":
        node_rows = [g.node_cont_matrix() for g in graphs if g.num_nodes > 0]
        edge_rows = [g.edge_cont_matrix() for g in graphs if len(g.edges) > 0]
        node_all = (
            np.concatenate(node_rows, axis=0)
            if node_rows
            else np.zeros((1, len(NODE_CONT_FEATURES)), dtype=np.float32)
        )
        edge_all = (
            np.concatenate(edge_rows, axis=0)
            if edge_rows
            else np.zeros((1, len(EDGE_CONT_FEATURES)), dtype=np.float32)
        )
        return cls(
            node_mean=node_all.mean(axis=0),
            node_std=node_all.std(axis=0) + eps,
            edge_mean=edge_all.mean(axis=0),
            edge_std=edge_all.std(axis=0) + eps,
        )

    def save(self, path: str) -> None:
        np.savez(
            path,
            node_mean=self.node_mean,
            node_std=self.node_std,
            edge_mean=self.edge_mean,
            edge_std=self.edge_std,
        )

    @classmethod
    def load(cls, path: str) -> "NormStats":
        d = np.load(path)
        return cls(
            node_mean=d["node_mean"],
            node_std=d["node_std"],
            edge_mean=d["edge_mean"],
            edge_std=d["edge_std"],
        )


def _one_hot(indices: np.ndarray, cardinality: int) -> np.ndarray:
    indices = np.clip(indices.astype(int), 0, cardinality - 1)
    out = np.zeros((indices.shape[0], cardinality), dtype=np.float32)
    if indices.shape[0]:
        out[np.arange(indices.shape[0]), indices] = 1.0
    return out


def build_node_matrix(graph: FaceGraph, stats: Optional[NormStats]) -> np.ndarray:
    """[N, NODE_FEATURE_DIM] = [归一化连续块 | one-hot(surface_type)]。"""
    cont = graph.node_cont_matrix()
    if stats is not None and cont.shape[0]:
        cont = (cont - stats.node_mean) / stats.node_std
    surf_idx = np.asarray(
        [int(n.feats.get("surface_type", SURFACE_TYPE_INDEX["other"])) for n in graph.nodes],
        dtype=int,
    )
    cat = _one_hot(surf_idx, len(SURFACE_TYPES))
    if cont.shape[0] == 0:
        return np.zeros((0, NODE_FEATURE_DIM), dtype=np.float32)
    return np.concatenate([cont, cat], axis=1).astype(np.float32)


def build_edge_matrix(graph: FaceGraph, stats: Optional[NormStats]) -> np.ndarray:
    """[E, EDGE_FEATURE_DIM] = [归一化连续块 | one-hot(curve_type)]。"""
    cont = graph.edge_cont_matrix()
    if stats is not None and cont.shape[0]:
        cont = (cont - stats.edge_mean) / stats.edge_std
    curve_idx = np.asarray(
        [int(e.feats.get("curve_type", CURVE_TYPE_INDEX["other"])) for e in graph.edges],
        dtype=int,
    )
    cat = _one_hot(curve_idx, len(CURVE_TYPES))
    if cont.shape[0] == 0:
        return np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)
    return np.concatenate([cont, cat], axis=1).astype(np.float32)
