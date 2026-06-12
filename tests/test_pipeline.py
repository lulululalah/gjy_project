"""端到端测试：schema / synth / window / GNN / instancing / metrics / 训练。

全部不依赖 OpenCASCADE，可在任意装了 torch + torch_geometric 的机器上运行：
    /opt/anaconda3/bin/python -m pytest tests/ -q
"""

import random

import numpy as np
import torch

from brep_defeature.graph.build import to_pyg_data
from brep_defeature.graph.window import sample_windows
from brep_defeature.instancing import gt_instances, predict_instances
from brep_defeature.metrics import (
    instance_detection_scores,
    operation_accuracy,
    semantic_scores,
)
from brep_defeature.models.gnn import DefeatureGNN
from brep_defeature.schema import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    FaceGraph,
    NormStats,
    Operation,
    Semantic,
)
from brep_defeature.synth import synth_dataset, synth_model


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
def test_synth_model_has_all_label_kinds():
    g = synth_model("t", random.Random(1), grid=5, n_rivets=4, n_holes=4)
    sems = {n.semantic for n in g.nodes}
    assert int(Semantic.BACKGROUND) in sems
    assert int(Semantic.RIVET) in sems
    assert int(Semantic.HOLE) in sems
    # 每个特征面都有合法实例 id 与非 KEEP 操作
    for n in g.nodes:
        if n.semantic != int(Semantic.BACKGROUND):
            assert n.instance_id >= 0
            assert n.operation != int(Operation.KEEP)
        else:
            assert n.instance_id == -1


def test_node_feature_keys_complete():
    from brep_defeature.schema import NODE_CONT_FEATURES

    g = synth_model("t", random.Random(2))
    for n in g.nodes:
        for key in NODE_CONT_FEATURES + ["surface_type"]:
            assert key in n.feats, f"缺少节点特征 {key}"


def test_json_roundtrip(tmp_path):
    g = synth_model("rt", random.Random(3))
    p = tmp_path / "g.json"
    g.to_json(str(p))
    g2 = FaceGraph.from_json(str(p))
    assert g2.model_id == g.model_id
    assert len(g2.nodes) == len(g.nodes)
    assert len(g2.edges) == len(g.edges)
    assert g2.nodes[0].semantic == g.nodes[0].semantic


def test_edge_same_instance_consistency():
    """实例内边 same_instance=1 当且仅当两端在同一非背景实例。"""
    g = synth_model("e", random.Random(4))
    idx = g.id_to_index()
    for e in g.edges:
        a, b = g.nodes[idx[e.src]], g.nodes[idx[e.dst]]
        both_feat = a.instance_id >= 0 and b.instance_id >= 0 and a.instance_id == b.instance_id
        assert bool(e.same_instance) == bool(both_feat)


# --------------------------------------------------------------------------- #
# graph build + window
# --------------------------------------------------------------------------- #
def test_to_pyg_dims():
    g = synth_model("d", random.Random(5))
    stats = NormStats.fit([g])
    data = to_pyg_data(g, stats)
    assert data.x.shape[1] == NODE_FEATURE_DIM
    assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM
    # 无向边展开为双向
    assert data.edge_index.shape[1] == 2 * len(g.edges)
    assert data.edge_y.shape[0] == data.edge_index.shape[1]
    assert data.y_sem.shape[0] == data.num_nodes


def test_window_centered_and_local():
    g = synth_model("w", random.Random(6))
    data = to_pyg_data(g, NormStats.fit([g]))
    wins = sample_windows(data, k=2, background_keep=1.0, rng=random.Random(0))
    assert len(wins) > 0
    for w in wins:
        assert 0 <= w.center_index < w.num_nodes
        # 窗口规模应远小于整图（局部性）
        assert w.num_nodes <= data.num_nodes


# --------------------------------------------------------------------------- #
# model smoke
# --------------------------------------------------------------------------- #
def test_gnn_forward_shapes():
    g = synth_model("m", random.Random(7))
    data = to_pyg_data(g, NormStats.fit([g]))
    model = DefeatureGNN(hidden=32, num_layers=3)
    model.eval()
    out = model(data)
    assert out["semantic"].shape == (data.num_nodes, len(Semantic))
    assert out["operation"].shape == (data.num_nodes, len(Operation))
    assert out["edge"].shape == (data.edge_index.shape[1], 2)


def test_gnn_overfits_single_graph():
    """模型应能在单图上过拟合（验证多任务管线可学）。"""
    torch.manual_seed(0)
    g = synth_model("o", random.Random(8), grid=5, n_rivets=4, n_holes=4)
    data = to_pyg_data(g, NormStats.fit([g]))
    model = DefeatureGNN(hidden=64, num_layers=3, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(150):
        opt.zero_grad()
        out = model(data)
        loss = torch.nn.functional.cross_entropy(out["semantic"], data.y_sem)
        loss = loss + torch.nn.functional.cross_entropy(out["edge"], data.edge_y)
        loss.backward()
        opt.step()
    model.eval()
    pred = model(data)["semantic"].argmax(1)
    acc = float((pred == data.y_sem).float().mean())
    assert acc > 0.9, f"单图过拟合精度过低: {acc:.3f}"


# --------------------------------------------------------------------------- #
# instancing + metrics
# --------------------------------------------------------------------------- #
def test_instancing_with_ground_truth_edges():
    """用真值边亲和度做连通分量，应能精确还原真值实例。"""
    g = synth_model("i", random.Random(9))
    data = to_pyg_data(g, NormStats.fit([g]))
    sem = data.y_sem.numpy()
    op = data.y_op.numpy()
    edge_prob = data.edge_y.numpy().astype(float)  # 真值当作概率
    _, pred = predict_instances(
        num_nodes=data.num_nodes,
        sem_pred=sem,
        op_pred=op,
        edge_index=data.edge_index.numpy(),
        edge_pair_id=data.edge_pair_id.numpy(),
        edge_same_prob=edge_prob,
    )
    gt = gt_instances(data.instance_id.numpy())
    sc = instance_detection_scores([p.faces for p in pred], gt, iou_threshold=0.5)
    assert sc.f1 == 1.0, f"真值边应完美还原实例, got f1={sc.f1}"
    assert sc.matched_mean_iou > 0.99


def test_semantic_scores_perfect():
    y = np.array([0, 0, 1, 1, 2, 2])
    sc = semantic_scores(y, y.copy(), num_classes=3)
    assert sc.accuracy == 1.0
    assert sc.miou == 1.0
    assert operation_accuracy(y, y.copy()) == 1.0


# --------------------------------------------------------------------------- #
# 训练（小规模）
# --------------------------------------------------------------------------- #
def test_train_generalizes_on_synthetic():
    from brep_defeature.train import TrainConfig, evaluate, train, split_graphs

    graphs = synth_dataset(n_models=24, seed=0)
    tr, va = split_graphs(graphs, ratio=0.8, seed=0)
    cfg = TrainConfig(epochs=25, use_windows=True, window_k=2)
    model, stats, history = train(tr, va, cfg)
    val = evaluate(model, [to_pyg_data(g, stats) for g in va], torch.device("cpu"))
    # 验证集上应明显学到东西（远超随机 1/6）
    assert val["sem_miou"] > 0.4, f"验证 mIoU 偏低: {val}"
    assert val["inst_f1"] > 0.3, f"验证实例 F1 偏低: {val}"
