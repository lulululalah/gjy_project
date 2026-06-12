"""微小凸起特征的几何构造（要被 Fuse 融入原 BRep 拓扑）。

三类凸起，均沿宿主面外法向生成，并把底部**略微沉入**表面，
保证布尔并运算能在宿主面上压印出一条闭合边界 → 真正的拓扑融合
（宿主面被分割、共享边产生），而不是简单叠加。

- nail       小钉子：细圆柱柱身 + 球冠钉头
- hemisphere 半圆凸起：球心落在表面，外侧半球凸出
- frustum    梯台凸起：圆台（上小下大，截面呈梯形）

⚠️ 需要 pythonocc-core 运行环境。
"""

from __future__ import annotations

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.BRepPrimAPI import (
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeSphere,
)
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

FEATURE_KINDS = ("nail", "hemisphere", "frustum")


def _shifted(pnt: gp_Pnt, normal: gp_Dir, d: float) -> gp_Pnt:
    return gp_Pnt(pnt.X() + normal.X() * d, pnt.Y() + normal.Y() * d, pnt.Z() + normal.Z() * d)


def make_nail(pnt: gp_Pnt, normal: gp_Dir, radius: float, height: float):
    """细圆柱柱身 + 球冠钉头；柱身底部沉入表面。"""
    base = _shifted(pnt, normal, -0.3 * radius)
    shaft = BRepPrimAPI_MakeCylinder(gp_Ax2(base, normal), radius, height).Shape()
    head_center = _shifted(pnt, normal, height)
    head = BRepPrimAPI_MakeSphere(gp_Ax2(head_center, normal), radius * 1.4).Shape()
    return BRepAlgoAPI_Fuse(shaft, head).Shape()


def make_hemisphere(pnt: gp_Pnt, normal: gp_Dir, radius: float):
    """球心落在表面，Fuse 后外侧呈半球凸起。"""
    return BRepPrimAPI_MakeSphere(gp_Ax2(pnt, normal), radius).Shape()


def make_frustum(pnt: gp_Pnt, normal: gp_Dir, r_base: float, r_top: float, height: float):
    """圆台（梯台）：底大顶小；底部沉入表面。"""
    base = _shifted(pnt, normal, -0.3 * r_base)
    return BRepPrimAPI_MakeCone(gp_Ax2(base, normal), r_base, r_top, height).Shape()


def make_protrusion(kind: str, pnt: gp_Pnt, normal: gp_Dir, size: float):
    """按类型生成凸起实体。size 为特征基准尺寸（≈ 半径）。"""
    if kind == "nail":
        return make_nail(pnt, normal, radius=size * 0.45, height=size * 1.6)
    if kind == "hemisphere":
        return make_hemisphere(pnt, normal, radius=size)
    if kind == "frustum":
        return make_frustum(pnt, normal, r_base=size, r_top=size * 0.5, height=size * 1.2)
    raise ValueError(f"未知凸起类型: {kind}")
