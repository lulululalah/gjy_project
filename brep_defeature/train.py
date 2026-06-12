"""多任务训练。

监督信号：语义(逐面) + 操作(逐面) + 同实例(逐边)。三者加权求和。
类别不平衡用逆频率权重；训练可选用 k-hop 窗口子图（局部窗口机制 + 均衡）。
评估始终在整图上做（与推理一致）。
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .graph.build import to_pyg_data
from .graph.window import sample_windows
from .instancing import gt_instances, predict_instances
from .metrics import instance_detection_scores, operation_accuracy, semantic_scores
from .models.gnn import DefeatureGNN
from .schema import FaceGraph, NormStats, Operation, Semantic


@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 5e-3
    batch_size: int = 8
    hidden: int = 64
    num_layers: int = 3
    use_windows: bool = True
    window_k: int = 2
    background_keep: float = 0.2
    w_sem: float = 1.0
    w_op: float = 1.0
    w_edge: float = 1.0
    seed: int = 0
    device: str = "cpu"


@dataclass
class Dataset:
    data_list: list
    stats: NormStats
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
def build_dataset(graphs: list[FaceGraph], stats: Optional[NormStats] = None) -> Dataset:
    if stats is None:
        stats = NormStats.fit(graphs)
    data_list = [to_pyg_data(g, stats) for g in graphs]
    return Dataset(data_list=data_list, stats=stats)


def _class_weights(data_list, attr: str, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for d in data_list:
        y = getattr(d, attr).numpy()
        for c in range(num_classes):
            counts[c] += int(np.sum(y == c))
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def _expand_windows(data_list, k: int, background_keep: float, seed: int) -> list:
    rng = random.Random(seed)
    out = []
    for d in data_list:
        out.extend(sample_windows(d, k=k, background_keep=background_keep, rng=rng))
    return [w for w in out if w.num_nodes > 0 and w.edge_index.shape[1] > 0]


# --------------------------------------------------------------------------- #
def train(
    train_graphs: list[FaceGraph],
    val_graphs: list[FaceGraph],
    cfg: TrainConfig = TrainConfig(),
):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_ds = build_dataset(train_graphs)
    val_ds = build_dataset(val_graphs, stats=train_ds.stats)

    # 类别权重在整图上统计
    w_sem = _class_weights(train_ds.data_list, "y_sem", len(Semantic))
    w_op = _class_weights(train_ds.data_list, "y_op", len(Operation))
    w_edge = _class_weights(train_ds.data_list, "edge_y", 2)

    train_units = (
        _expand_windows(train_ds.data_list, cfg.window_k, cfg.background_keep, cfg.seed)
        if cfg.use_windows
        else train_ds.data_list
    )

    device = torch.device(cfg.device)
    model = DefeatureGNN(hidden=cfg.hidden, num_layers=cfg.num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(train_units, batch_size=cfg.batch_size, shuffle=True)

    history = []
    best_state, best_metric = None, -1.0
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            loss = cfg.w_sem * F.cross_entropy(out["semantic"], batch.y_sem, weight=w_sem.to(device))
            loss = loss + cfg.w_op * F.cross_entropy(out["operation"], batch.y_op, weight=w_op.to(device))
            if batch.edge_index.shape[1] > 0:
                loss = loss + cfg.w_edge * F.cross_entropy(out["edge"], batch.edge_y, weight=w_edge.to(device))
            loss.backward()
            opt.step()
            total += float(loss.item())

        val = evaluate(model, val_ds.data_list, device)
        history.append({"epoch": epoch, "train_loss": total / max(len(loader), 1), **val})
        # 以语义 mIoU + 实例 F1 作为选优指标
        metric = val["sem_miou"] + val["inst_f1"]
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_ds.stats, history


@torch.no_grad()
def evaluate(model, data_list, device) -> dict:
    model.eval()
    sem_t, sem_p, op_t, op_p = [], [], [], []
    inst_f1s, inst_ious = [], []
    for d in data_list:
        d = d.to(device)
        out = model(d)
        sp = out["semantic"].argmax(1).cpu().numpy()
        op = out["operation"].argmax(1).cpu().numpy()
        edge_prob = F.softmax(out["edge"], dim=1)[:, 1].cpu().numpy() if d.edge_index.shape[1] else np.zeros(0)
        sem_t.append(d.y_sem.cpu().numpy())
        sem_p.append(sp)
        op_t.append(d.y_op.cpu().numpy())
        op_p.append(op)

        _, pred_inst = predict_instances(
            num_nodes=d.num_nodes,
            sem_pred=sp,
            op_pred=op,
            edge_index=d.edge_index.cpu().numpy(),
            edge_pair_id=d.edge_pair_id.cpu().numpy(),
            edge_same_prob=edge_prob,
        )
        gt = gt_instances(d.instance_id.cpu().numpy())
        sc = instance_detection_scores([p.faces for p in pred_inst], gt, iou_threshold=0.5)
        inst_f1s.append(sc.f1)
        inst_ious.append(sc.matched_mean_iou)

    sem_t = np.concatenate(sem_t)
    sem_p = np.concatenate(sem_p)
    op_t = np.concatenate(op_t)
    op_p = np.concatenate(op_p)
    sem = semantic_scores(sem_t, sem_p, len(Semantic))
    return {
        "sem_acc": sem.accuracy,
        "sem_macro_f1": sem.macro_f1,
        "sem_miou": sem.miou,
        "op_acc": operation_accuracy(op_t, op_p),
        "inst_f1": float(np.mean(inst_f1s)) if inst_f1s else 0.0,
        "inst_iou": float(np.mean(inst_ious)) if inst_ious else 0.0,
    }


def split_graphs(graphs: list[FaceGraph], ratio: float = 0.8, seed: int = 0):
    rng = random.Random(seed)
    idx = list(range(len(graphs)))
    rng.shuffle(idx)
    cut = int(len(graphs) * ratio)
    return [graphs[i] for i in idx[:cut]], [graphs[i] for i in idx[cut:]]


def main():
    from .synth import synth_dataset

    ap = argparse.ArgumentParser(description="在合成数据上训练多任务 defeature GNN")
    ap.add_argument("--models", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--no-windows", action="store_true")
    ap.add_argument("--out", type=str, default="defeature_gnn.pth")
    ap.add_argument("--stats", type=str, default="defeature_stats.npz")
    args = ap.parse_args()

    graphs = synth_dataset(n_models=args.models, seed=0)
    tr, va = split_graphs(graphs, ratio=0.8, seed=0)
    cfg = TrainConfig(epochs=args.epochs, use_windows=not args.no_windows)
    model, stats, history = train(tr, va, cfg)

    last = history[-1]
    print("== 最终验证集指标 ==")
    for k in ["sem_acc", "sem_macro_f1", "sem_miou", "op_acc", "inst_f1", "inst_iou"]:
        print(f"  {k:12s}: {last[k]:.3f}")
    torch.save(model.state_dict(), args.out)
    stats.save(args.stats)
    print(f"已保存模型 -> {args.out}，统计 -> {args.stats}")


if __name__ == "__main__":
    main()
