"""评估指标：语义 mIoU/PRF、操作准确率、实例级检测 F1。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SemanticScores:
    accuracy: float
    macro_f1: float
    miou: float
    per_class: dict[int, dict[str, float]]


def semantic_scores(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> SemanticScores:
    per_class: dict[int, dict[str, float]] = {}
    f1s, ious = [], []
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        support = int(np.sum(y_true == c))
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "iou": iou, "support": support}
        if support > 0:  # 只对出现过的类别计入均值
            f1s.append(f1)
            ious.append(iou)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    return SemanticScores(
        accuracy=acc,
        macro_f1=float(np.mean(f1s)) if f1s else 0.0,
        miou=float(np.mean(ious)) if ious else 0.0,
        per_class=per_class,
    )


def operation_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def _iou(a: set[int], b: set[int]) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class InstanceScores:
    precision: float
    recall: float
    f1: float
    matched_mean_iou: float
    num_pred: int
    num_gt: int


def instance_detection_scores(
    pred_instances: list[list[int]],
    gt_instances: list[list[int]],
    iou_threshold: float = 0.5,
) -> InstanceScores:
    """贪心 IoU 匹配预测实例与真值实例，给出检测 P/R/F1。"""
    pred_sets = [set(p) for p in pred_instances]
    gt_sets = [set(g) for g in gt_instances]

    pairs = []
    for pi, ps in enumerate(pred_sets):
        for gi, gs in enumerate(gt_sets):
            iou = _iou(ps, gs)
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)

    used_pred, used_gt = set(), set()
    matched_ious = []
    for iou, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matched_ious.append(iou)

    tp = len(matched_ious)
    prec = tp / len(pred_sets) if pred_sets else 0.0
    rec = tp / len(gt_sets) if gt_sets else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return InstanceScores(
        precision=prec,
        recall=rec,
        f1=f1,
        matched_mean_iou=float(np.mean(matched_ious)) if matched_ious else 0.0,
        num_pred=len(pred_sets),
        num_gt=len(gt_sets),
    )
