"""推理：FaceGraph -> 逐面语义/操作 + 特征实例（带 face_id 映射）。

输出可直接喂给 occ.defeature 执行几何去除（按实例与操作分组）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .graph.build import to_pyg_data
from .instancing import Instance, predict_instances
from .models.gnn import DefeatureGNN
from .schema import FaceGraph, NormStats


@dataclass
class InferResult:
    face_ids: list[int]
    semantic: np.ndarray          # [N]
    operation: np.ndarray         # [N]
    instance_ids: np.ndarray      # [N] (-1 背景)
    instances: list[Instance]     # 面下标按节点 index

    def instance_face_ids(self) -> list[dict]:
        """实例 -> {instance_id, semantic, operation, face_ids}（用 face_id）。"""
        out = []
        for inst in self.instances:
            out.append(
                {
                    "instance_id": inst.instance_id,
                    "semantic": inst.semantic,
                    "operation": inst.operation,
                    "face_ids": [self.face_ids[i] for i in inst.faces],
                }
            )
        return out


def load_model(weights_path: str, hidden: int = 64, num_layers: int = 3, device: str = "cpu") -> DefeatureGNN:
    model = DefeatureGNN(hidden=hidden, num_layers=num_layers)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def infer(model: DefeatureGNN, graph: FaceGraph, stats: NormStats, threshold: float = 0.5) -> InferResult:
    data = to_pyg_data(graph, stats)
    out = model(data)
    sem = out["semantic"].argmax(1).numpy()
    op = out["operation"].argmax(1).numpy()
    edge_prob = (
        F.softmax(out["edge"], dim=1)[:, 1].numpy()
        if data.edge_index.shape[1] > 0
        else np.zeros(0, dtype=np.float32)
    )
    inst_ids, instances = predict_instances(
        num_nodes=data.num_nodes,
        sem_pred=sem,
        op_pred=op,
        edge_index=data.edge_index.numpy(),
        edge_pair_id=data.edge_pair_id.numpy(),
        edge_same_prob=edge_prob,
        threshold=threshold,
    )
    return InferResult(
        face_ids=[n.face_id for n in graph.nodes],
        semantic=sem,
        operation=op,
        instance_ids=inst_ids,
        instances=instances,
    )
