"""在干净 BRep 上注入微小特征，并「注入即真值」地导出标签。

核心：用 BOP（布尔运算）历史把「注入工具体的面」映射到结果模型中的面，
从而精确知道哪些结果面属于哪个特征实例 —— 真值由构造过程给出，无需人工标注，
也无需 STEP 往返（避免面序错位风险，设计文档 T1）。

第一阶段聚焦：铆钉（Fuse 凸起）+ 孔（Cut 通孔/盲孔）。

⚠️ 需要 pythonocc-core 运行环境。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

from ..schema import Operation, Semantic
from .extract import build_face_map, shape_to_graph


@dataclass
class InjectionRecord:
    """一次注入的元信息（用于自验证去除算法：注入的逆操作即正确答案）。"""

    instance_id: int
    semantic: int
    operation: int
    params: dict = field(default_factory=dict)


def _faces_of(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(topods.Face(exp.Current()))
        exp.Next()
    return out


def _sample_point_on_face(face, rng: random.Random):
    """在面参数域中点附近取一点及其外法向。"""
    adaptor = BRepAdaptor_Surface(face)
    u = 0.5 * (adaptor.FirstUParameter() + adaptor.LastUParameter())
    v = 0.5 * (adaptor.FirstVParameter() + adaptor.LastVParameter())
    props = BRepLProp_SLProps(adaptor, u, v, 1, 1e-6)
    pnt = props.Value()
    n = props.Normal()
    if face.Orientation() == 1:  # REVERSED
        n.Reverse()
    return pnt, gp_Dir(n)


def _result_faces_for_tool(bop, tool_shape, result_fmap) -> set[int]:
    """用 BOP 历史，把工具体的面映射到结果模型中的 face_id 集合。"""
    ids: set[int] = set()
    for tf in _faces_of(tool_shape):
        for getter in (bop.Modified, bop.Generated):
            try:
                lst = getter(tf)
            except RuntimeError:
                continue
            from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape

            it = TopTools_ListIteratorOfListOfShape(lst)
            while it.More():
                fid = result_fmap.FindIndex(topods.Face(it.Value()))
                if fid > 0:
                    ids.add(fid)
                it.Next()
        # 工具面本身若原样保留在结果中
        fid = result_fmap.FindIndex(tf)
        if fid > 0:
            ids.add(fid)
    return ids


def _make_rivet_tool(pnt: gp_Pnt, normal: gp_Dir, radius: float, height: float):
    """圆柱台 + 顶部球冠，沿法向外凸。"""
    ax = gp_Ax2(pnt, normal)
    cyl = BRepPrimAPI_MakeCylinder(ax, radius, height).Shape()
    sph = BRepPrimAPI_MakeSphere(gp_Ax2(
        gp_Pnt(pnt.X() + normal.X() * height,
               pnt.Y() + normal.Y() * height,
               pnt.Z() + normal.Z() * height), normal), radius).Shape()
    return BRepAlgoAPI_Fuse(cyl, sph).Shape()


def _make_hole_tool(pnt: gp_Pnt, normal: gp_Dir, radius: float, depth: float):
    """沿法向反向（钻入）的圆柱刀具。"""
    inward = gp_Dir(-normal.X(), -normal.Y(), -normal.Z())
    # 起点略抬出表面，保证完全切穿表层
    start = gp_Pnt(pnt.X() + normal.X() * 0.1 * radius,
                   pnt.Y() + normal.Y() * 0.1 * radius,
                   pnt.Z() + normal.Z() * 0.1 * radius)
    ax = gp_Ax2(start, inward)
    return BRepPrimAPI_MakeCylinder(ax, radius, depth).Shape()


def inject_features(
    base_shape,
    rng: random.Random,
    n_rivets: int = 4,
    n_holes: int = 3,
    rivet_radius=(0.3, 0.8),
    hole_radius=(0.3, 0.8),
):
    """在 base_shape 上注入特征。

    Returns:
        (result_shape, labels, records)
        labels:  {result_face_id: (semantic, instance_id, operation)}
        records: list[InjectionRecord]（含逆操作参数，用于去除自验证）
    """
    shape = base_shape
    labels: dict[int, tuple] = {}
    records: list[InjectionRecord] = []
    inst_counter = 0

    # 候选宿主面：选面积较大的平面
    def planar_hosts():
        hosts = []
        for f in _faces_of(shape):
            ad = BRepAdaptor_Surface(f)
            from OCC.Core.GeomAbs import GeomAbs_Plane

            if ad.GetType() == GeomAbs_Plane:
                hosts.append(f)
        return hosts

    # --- 铆钉 ---
    for _ in range(n_rivets):
        hosts = planar_hosts()
        if not hosts:
            break
        host = rng.choice(hosts)
        try:
            pnt, normal = _sample_point_on_face(host, rng)
        except RuntimeError:
            continue
        r = rng.uniform(*rivet_radius)
        h = r * rng.uniform(1.0, 2.0)
        tool = _make_rivet_tool(pnt, normal, r, h)
        bop = BRepAlgoAPI_Fuse(shape, tool)
        bop.Build()
        if not bop.IsDone():
            continue
        result = bop.Shape()
        result_fmap = build_face_map(result)
        feat_ids = _result_faces_for_tool(bop, tool, result_fmap)
        # Fuse 后 face_id 体系整体变化：先把旧标签按几何同一性迁移，再给新特征面打标。
        labels = _remap_labels(labels, shape, result, result_fmap)
        for fid in feat_ids:
            labels[fid] = (int(Semantic.RIVET), inst_counter, int(Operation.REMOVE_PROTRUSION))
        records.append(InjectionRecord(
            instance_id=inst_counter, semantic=int(Semantic.RIVET),
            operation=int(Operation.REMOVE_PROTRUSION),
            params={"radius": r, "height": h,
                    "point": (pnt.X(), pnt.Y(), pnt.Z()),
                    "normal": (normal.X(), normal.Y(), normal.Z())},
        ))
        inst_counter += 1
        shape = result

    # --- 孔 ---
    for _ in range(n_holes):
        hosts = planar_hosts()
        if not hosts:
            break
        host = rng.choice(hosts)
        try:
            pnt, normal = _sample_point_on_face(host, rng)
        except RuntimeError:
            continue
        r = rng.uniform(*hole_radius)
        depth = r * rng.uniform(2.0, 5.0)
        tool = _make_hole_tool(pnt, normal, r, depth)
        bop = BRepAlgoAPI_Cut(shape, tool)
        bop.Build()
        if not bop.IsDone():
            continue
        result = bop.Shape()
        result_fmap = build_face_map(result)
        feat_ids = _result_faces_for_tool(bop, tool, result_fmap)
        labels = _remap_labels(labels, shape, result, result_fmap)
        for fid in feat_ids:
            labels[fid] = (int(Semantic.HOLE), inst_counter, int(Operation.FILL_HOLE))
        records.append(InjectionRecord(
            instance_id=inst_counter, semantic=int(Semantic.HOLE),
            operation=int(Operation.FILL_HOLE),
            params={"radius": r, "depth": depth,
                    "point": (pnt.X(), pnt.Y(), pnt.Z()),
                    "normal": (normal.X(), normal.Y(), normal.Z())},
        ))
        inst_counter += 1
        shape = result

    return shape, labels, records


def _remap_labels(old_labels, old_shape, new_shape, new_fmap) -> dict:
    """BOP 后 face_id 体系改变：用几何同一性把旧标签迁移到新 face_id。

    简化实现：依赖 BOP 通常保持未受影响面的 TopoDS 同一性，用 new_fmap.FindIndex
    在新面图里定位旧面。受影响的面（被分割/删除）丢弃旧标签，由调用方重新打标。
    """
    old_fmap = build_face_map(old_shape)
    remapped = {}
    for old_fid, lab in old_labels.items():
        if old_fid < 1 or old_fid > old_fmap.Extent():
            continue
        old_face = topods.Face(old_fmap.FindKey(old_fid))
        new_fid = new_fmap.FindIndex(old_face)
        if new_fid > 0:
            remapped[new_fid] = lab
    return remapped


def make_training_sample(base_shape, model_id: str, rng: random.Random, **kw):
    """注入 + 直接构图带标签，返回 (FaceGraph, result_shape, records)。"""
    result, labels, records = inject_features(base_shape, rng, **kw)
    graph = shape_to_graph(result, model_id=model_id, labels=labels)
    return graph, result, records
