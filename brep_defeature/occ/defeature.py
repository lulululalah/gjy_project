"""几何去除与补全。

按 GNN 推理给出的「实例 + 操作」分组，删除特征面并补平周围曲面，使用
BRepAlgoAPI_Defeaturing（RemoveFeatures，OCCT 7.4+）打底。每个实例独立执行，
失败回滚不污染其他实例；每步做有效性与体积变化校验。

自验证：注入数据保存了「注入前的干净 shape」，去除结果与之的 Hausdorff 距离 /
体积差即去除算法的正确性度量（设计文档 §6.4）。

⚠️ 需要 pythonocc-core 运行环境（含 BRepAlgoAPI_Defeaturing）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Defeaturing
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopoDS import topods

from .extract import build_face_map


@dataclass
class DefeatureReport:
    requested: int = 0
    removed: int = 0
    failed: list[int] = field(default_factory=list)
    valid: bool = True
    volume_before: float = 0.0
    volume_after: float = 0.0

    @property
    def volume_change_ratio(self) -> float:
        if self.volume_before <= 1e-9:
            return 0.0
        return abs(self.volume_after - self.volume_before) / self.volume_before


def _volume(shape) -> float:
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return props.Mass()


def _is_valid(shape) -> bool:
    return BRepCheck_Analyzer(shape).IsValid()


def remove_instances(
    shape,
    instances: list[dict],
    max_volume_change: float = 0.05,
) -> tuple[object, DefeatureReport]:
    """逐实例删除特征面并补平。

    Args:
        shape: 待处理 TopoDS_Shape。
        instances: [{instance_id, operation, face_ids:[...]}, ...]（face_id 为当前 shape 面图下标）。
        max_volume_change: 单实例允许的最大体积变化比（超出则回滚该实例）。

    Returns:
        (result_shape, report)
    """
    report = DefeatureReport(requested=len(instances))
    report.volume_before = _volume(shape)
    current = shape

    for inst in instances:
        fmap = build_face_map(current)
        faces = []
        for fid in inst["face_ids"]:
            if 1 <= fid <= fmap.Extent():
                faces.append(topods.Face(fmap.FindKey(fid)))
        if not faces:
            report.failed.append(inst["instance_id"])
            continue

        vol_before = _volume(current)
        try:
            algo = BRepAlgoAPI_Defeaturing()
            algo.SetShape(current)
            for f in faces:
                algo.AddFaceToRemove(f)
            algo.Build()
            if not algo.IsDone():
                report.failed.append(inst["instance_id"])
                continue
            candidate = algo.Shape()
        except RuntimeError:
            report.failed.append(inst["instance_id"])
            continue

        vol_after = _volume(candidate)
        ok = _is_valid(candidate)
        change = abs(vol_after - vol_before) / vol_before if vol_before > 1e-9 else 0.0
        if not ok or change > max_volume_change:
            # 回滚：保持 current 不变
            report.failed.append(inst["instance_id"])
            continue

        current = candidate
        report.removed += 1

    report.volume_after = _volume(current)
    report.valid = _is_valid(current)
    return current, report


# --------------------------------------------------------------------------- #
# 自验证：与参考（注入前）模型对比
# --------------------------------------------------------------------------- #
@dataclass
class RecoveryError:
    max_distance: float       # 近似 Hausdorff（采样）
    volume_diff_ratio: float


def recovery_error(result_shape, reference_clean_shape) -> RecoveryError:
    """去除结果 vs 注入前干净模型的偏差。值越小说明去除越接近正确答案。"""
    dist = BRepExtrema_DistShapeShape(result_shape, reference_clean_shape)
    dist.Perform()
    max_d = dist.Value() if dist.IsDone() else float("inf")
    v_res = _volume(result_shape)
    v_ref = _volume(reference_clean_shape)
    ratio = abs(v_res - v_ref) / v_ref if v_ref > 1e-9 else 0.0
    return RecoveryError(max_distance=max_d, volume_diff_ratio=ratio)
