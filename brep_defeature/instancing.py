"""实例化：逐边亲和度 -> 连通分量 -> 特征实例。

输入逐面语义预测 + 逐边「同实例」概率，输出每个特征面的实例 id 及实例级操作。
只在「特征面」（语义≠背景）之间按预测同实例的边做并查集合并；每个连通分量
即一个特征实例。实例操作 = 成员面操作预测的多数票。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .schema import Operation, Semantic


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class Instance:
    instance_id: int
    faces: list[int]          # 节点下标
    semantic: int
    operation: int


def predict_instances(
    num_nodes: int,
    sem_pred: np.ndarray,           # [N]
    op_pred: np.ndarray,            # [N]
    edge_index: np.ndarray,         # [2, E] 有向
    edge_pair_id: np.ndarray,       # [E] 指向原无向边
    edge_same_prob: np.ndarray,     # [E] P(same_instance)
    threshold: float = 0.5,
) -> tuple[np.ndarray, list[Instance]]:
    """返回 (inst_ids[N]，实例列表)。背景面 inst_id = -1。"""
    is_feature = sem_pred != int(Semantic.BACKGROUND)
    uf = _UnionFind(num_nodes)

    # 把有向边按原无向边聚合（两方向概率取均值）
    pair_prob: dict[int, list[float]] = {}
    pair_ends: dict[int, tuple[int, int]] = {}
    for col in range(edge_index.shape[1]):
        pid = int(edge_pair_id[col])
        a, b = int(edge_index[0, col]), int(edge_index[1, col])
        pair_prob.setdefault(pid, []).append(float(edge_same_prob[col]))
        pair_ends[pid] = (a, b)

    for pid, probs in pair_prob.items():
        a, b = pair_ends[pid]
        if not (is_feature[a] and is_feature[b]):
            continue
        if np.mean(probs) >= threshold:
            uf.union(a, b)

    # 收集特征面的连通分量
    comp: dict[int, list[int]] = {}
    for i in range(num_nodes):
        if not is_feature[i]:
            continue
        comp.setdefault(uf.find(i), []).append(i)

    inst_ids = np.full(num_nodes, -1, dtype=int)
    instances: list[Instance] = []
    for iid, (_, faces) in enumerate(sorted(comp.items())):
        for f in faces:
            inst_ids[f] = iid
        sem = Counter(int(sem_pred[f]) for f in faces).most_common(1)[0][0]
        op = Counter(int(op_pred[f]) for f in faces).most_common(1)[0][0]
        instances.append(Instance(instance_id=iid, faces=sorted(faces), semantic=sem, operation=op))
    return inst_ids, instances


def gt_instances(instance_id: np.ndarray) -> list[list[int]]:
    """从真值 instance_id 数组提取实例的面集合（忽略 -1 背景）。"""
    comp: dict[int, list[int]] = {}
    for i, iid in enumerate(instance_id):
        if int(iid) < 0:
            continue
        comp.setdefault(int(iid), []).append(i)
    return [sorted(v) for _, v in sorted(comp.items())]
