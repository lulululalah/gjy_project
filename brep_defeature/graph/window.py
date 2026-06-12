"""局部窗口机制：以面为种子的 k-hop 子图采样。

微小特征是局部的，判断一个面只需其 k 跳邻域。训练时把每个（采样到的）面
变成一个以它为中心的 k-hop 子图样本：
- 子图中心面是预测目标（其语义/操作标签）
- 子图其余面提供局部上下文
- 背景种子下采样，缓解类别不平衡

推理可直接用浅层 GNN 整图前向（层数 = 感受野半径），与训练的局部性一致。
"""

from __future__ import annotations

import random
from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from ..schema import Semantic


def sample_windows(
    data: Data,
    k: int = 2,
    background_keep: float = 0.15,
    max_windows: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> list[Data]:
    """把一张整图拆成多个 k-hop 窗口子图。

    Args:
        data: 整图（to_pyg_data 产物）。
        k: 窗口半径（跳数）。
        background_keep: 背景中心面的保留比例（下采样）。
        max_windows: 限制窗口总数（None 不限）。
        rng: 随机源（确定性测试用）。

    Returns:
        子图列表；每个子图带 center_index（中心面在子图内的下标）
        与 center_sem / center_op（中心面标签）。
    """
    rng = rng or random.Random(0)
    n = data.num_nodes
    seeds: list[int] = []
    for i in range(n):
        is_bg = int(data.y_sem[i]) == int(Semantic.BACKGROUND)
        if is_bg and rng.random() > background_keep:
            continue
        seeds.append(i)
    rng.shuffle(seeds)
    if max_windows is not None:
        seeds = seeds[:max_windows]

    windows: list[Data] = []
    for seed in seeds:
        sub = _extract_window(data, seed, k)
        if sub is not None:
            windows.append(sub)
    return windows


def _extract_window(data: Data, seed: int, k: int) -> Optional[Data]:
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=seed,
        num_hops=k,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )
    if subset.numel() == 0:
        return None

    sub = Data(
        x=data.x[subset],
        edge_index=edge_index,
        edge_attr=data.edge_attr[edge_mask] if data.edge_attr is not None else None,
    )
    sub.y_sem = data.y_sem[subset]
    sub.y_op = data.y_op[subset]
    sub.instance_id = data.instance_id[subset]
    sub.edge_y = data.edge_y[edge_mask]
    # mapping[0] 是 seed 在子图中的新下标
    sub.center_index = int(mapping[0].item())
    sub.center_sem = int(data.y_sem[seed].item())
    sub.center_op = int(data.y_op[seed].item())
    return sub
