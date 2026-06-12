"""多任务 GNN：共享 NNConv 编码器 + 三个任务头。

      x, edge_attr
          │
   ┌──────▼───────────────────┐
   │  NNConv ×L (含边特征)      │  -> 节点 embedding h [N, H]
   │  BatchNorm + ReLU + 残差   │
   └──────┬──────────┬─────────┘
   语义头  │   操作头  │   实例边界头
   [N,Cs]      [N,Co]      [E,2]  (concat h_i,h_j,edge_attr)

- 语义头/操作头：逐面分类。
- 实例边界头：逐（有向）边二分类「是否同一实例」。推理时对预测为同实例的
  特征面边做连通分量，得到特征实例（见 instancing.py）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv

from ..schema import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, Operation, Semantic


class DefeatureGNN(nn.Module):
    def __init__(
        self,
        node_dim: int = NODE_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        hidden: int = 64,
        num_layers: int = 3,
        num_semantic: int = len(Semantic),
        num_operation: int = len(Operation),
        nn_hidden: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = nn.Linear(node_dim, hidden)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            edge_nn = nn.Sequential(
                nn.Linear(edge_dim, nn_hidden),
                nn.ReLU(),
                nn.Linear(nn_hidden, hidden * hidden),
            )
            self.convs.append(NNConv(hidden, hidden, edge_nn, aggr="mean"))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.semantic_head = nn.Linear(hidden, num_semantic)
        self.operation_head = nn.Linear(hidden, num_operation)
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def encode(self, x, edge_index, edge_attr):
        h = self.input_proj(x)
        h = F.relu(h)
        for conv, bn in zip(self.convs, self.bns):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if h.shape == h_in.shape:
                h = h + h_in
        return h

    def edge_logits(self, h, edge_index, edge_attr):
        src, dst = edge_index
        feat = torch.cat([h[src], h[dst], edge_attr], dim=1)
        return self.edge_head(feat)

    def forward(self, data):
        h = self.encode(data.x, data.edge_index, data.edge_attr)
        out = {
            "semantic": self.semantic_head(h),                       # [N, Cs]
            "operation": self.operation_head(h),                     # [N, Co]
            "edge": self.edge_logits(h, data.edge_index, data.edge_attr),  # [E, 2]
        }
        return out
