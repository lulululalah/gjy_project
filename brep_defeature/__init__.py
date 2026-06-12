"""brep_defeature — BRep 微小特征识别与去除。

分层：
- schema:   FaceGraph / 标签的数据契约（解耦几何后端与 ML 管线）。
- synth:    OCC-free 合成图生成器（开发/测试用，无需 OpenCASCADE）。
- graph:    图构建与 k-hop 局部窗口采样。
- models:   多任务 GNN（语义分割 / 实例边界 / 操作预测）。
- train/infer/instancing/metrics: 训练、推理、实例化、评估。
- occ:      pythonocc-core 几何后端（注入 / 提取 / 去除），在 Windows/conda 环境运行。

设计文档见 doc/算法设计-特征识别与去除.md。
"""

from .schema import (
    EDGE_FEATURES,
    NODE_FEATURES,
    Operation,
    Semantic,
    EdgeRecord,
    FaceGraph,
    FaceNode,
)

__all__ = [
    "FaceGraph",
    "FaceNode",
    "EdgeRecord",
    "Semantic",
    "Operation",
    "NODE_FEATURES",
    "EDGE_FEATURES",
]
