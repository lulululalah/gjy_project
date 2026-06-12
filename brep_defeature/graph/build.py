"""FaceGraph -> torch_geometric.data.Data。

无向 BRep 边展开为两条有向边（消息双向传播）；边特征与 same_instance 标签
对两条有向边相同。Data 上携带三套监督信号：
- y_sem    : [N]  逐面语义类别
- y_op     : [N]  逐面操作类型
- edge_y   : [E2] 逐（有向）边 same_instance（实例边界头标签）
另存 edge_pair_id [E2] 用于把有向边对回连接到原无向边（评估/连通分量用）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from ..schema import (
    FaceGraph,
    NormStats,
    build_edge_matrix,
    build_node_matrix,
)


def to_pyg_data(graph: FaceGraph, stats: Optional[NormStats] = None) -> Data:
    idx = graph.id_to_index()

    x = torch.from_numpy(build_node_matrix(graph, stats))
    y_sem = torch.tensor([n.semantic for n in graph.nodes], dtype=torch.long)
    y_op = torch.tensor([n.operation for n in graph.nodes], dtype=torch.long)
    inst = torch.tensor([n.instance_id for n in graph.nodes], dtype=torch.long)

    edge_feat = build_edge_matrix(graph, stats)  # [E, F] 每条无向边一行

    src_list: list[int] = []
    dst_list: list[int] = []
    attr_rows: list[np.ndarray] = []
    edge_y: list[int] = []
    pair_id: list[int] = []

    for e_i, e in enumerate(graph.edges):
        if e.src not in idx or e.dst not in idx:
            continue
        u, v = idx[e.src], idx[e.dst]
        row = edge_feat[e_i] if edge_feat.shape[0] else np.zeros(0, dtype=np.float32)
        # 两个方向
        for a, b in ((u, v), (v, u)):
            src_list.append(a)
            dst_list.append(b)
            attr_rows.append(row)
            edge_y.append(e.same_instance)
            pair_id.append(e_i)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.from_numpy(np.asarray(attr_rows, dtype=np.float32))
        edge_y_t = torch.tensor(edge_y, dtype=torch.long)
        pair_id_t = torch.tensor(pair_id, dtype=torch.long)
    else:
        from ..schema import EDGE_FEATURE_DIM

        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)
        edge_y_t = torch.zeros((0,), dtype=torch.long)
        pair_id_t = torch.zeros((0,), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.y_sem = y_sem
    data.y_op = y_op
    data.instance_id = inst
    data.edge_y = edge_y_t
    data.edge_pair_id = pair_id_t
    data.model_id = graph.model_id
    return data
