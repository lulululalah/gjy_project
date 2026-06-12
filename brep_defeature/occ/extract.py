"""STEP / TopoDS_Shape -> FaceGraph。

面以确定性的 TopTools_IndexedMapOfShape 下标作为 face_id（同一进程内加载的同一
shape 稳定）。注入数据时不做 STEP 往返，直接对内存中的修改后 shape 提取 + 标注，
从而规避「STEP 重载后面序错位」的风险（设计文档 T1）。

节点/边特征键与 brep_defeature.schema 完全一致，使真实数据与合成数据共用同一管线。

⚠️ 需要 pythonocc-core 运行环境。
"""

from __future__ import annotations

import math
from typing import Optional

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GeomAbs import (
    GeomAbs_BSplineCurve,
    GeomAbs_BSplineSurface,
    GeomAbs_BezierCurve,
    GeomAbs_BezierSurface,
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Ellipse,
    GeomAbs_Line,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopoDS import TopoDS_Face, topods
from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_IndexedMapOfShape,
)

from ..schema import (
    CURVE_TYPE_INDEX,
    SURFACE_TYPE_INDEX,
    EdgeRecord,
    FaceGraph,
    FaceNode,
)

_SURF_MAP = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
    GeomAbs_BSplineSurface: "bspline",
    GeomAbs_BezierSurface: "bspline",
}
_CURVE_MAP = {
    GeomAbs_Line: "line",
    GeomAbs_Circle: "circle",
    GeomAbs_Ellipse: "ellipse",
    GeomAbs_BSplineCurve: "bspline",
    GeomAbs_BezierCurve: "bspline",
}


def read_step(path: str):
    """读取 STEP，返回 TopoDS_Shape。"""
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != 1:  # IFSelect_RetDone
        raise IOError(f"无法读取 STEP: {path} (status={status})")
    reader.TransferRoots()
    return reader.OneShape()


def build_face_map(shape) -> TopTools_IndexedMapOfShape:
    fmap = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, fmap)
    return fmap


# --------------------------------------------------------------------------- #
# 单面几何特征
# --------------------------------------------------------------------------- #
def _surface_area(face: TopoDS_Face) -> float:
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return props.Mass()


def _bbox_min_dim(face: TopoDS_Face) -> float:
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    box = Bnd_Box()
    brepbndlib.Add(face, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return min(xmax - xmin, ymax - ymin, zmax - zmin)


def _surface_info(face: TopoDS_Face):
    """返回 (surface_type_index, radius, nx, ny, nz, mean_curv, gauss_curv)。"""
    adaptor = BRepAdaptor_Surface(face)
    stype = adaptor.GetType()
    name = _SURF_MAP.get(stype, "other")
    radius = 0.0
    if stype == GeomAbs_Cylinder:
        radius = adaptor.Cylinder().Radius()
    elif stype == GeomAbs_Sphere:
        radius = adaptor.Sphere().Radius()
    elif stype == GeomAbs_Cone:
        radius = adaptor.Cone().RefRadius()
    elif stype == GeomAbs_Torus:
        radius = adaptor.Torus().MinorRadius()

    umin, umax, vmin, vmax = adaptor.FirstUParameter(), adaptor.LastUParameter(), \
        adaptor.FirstVParameter(), adaptor.LastVParameter()
    u = 0.5 * (umin + umax)
    v = 0.5 * (vmin + vmax)
    nx = ny = nz = 0.0
    mean_c = gauss_c = 0.0
    try:
        props = BRepLProp_SLProps(adaptor, u, v, 2, 1e-6)
        if props.IsNormalDefined():
            n = props.Normal()
            if face.Orientation() == 1:  # TopAbs_REVERSED
                n.Reverse()
            nx, ny, nz = n.X(), n.Y(), n.Z()
        if props.IsCurvatureDefined():
            mean_c = abs(props.MeanCurvature())
            gauss_c = abs(props.GaussianCurvature())
    except RuntimeError:
        pass
    return SURFACE_TYPE_INDEX[name], radius, nx, ny, nz, mean_c, gauss_c


def _wire_info(face: TopoDS_Face):
    """返回 (num_wires, inner_wire_count, min_inner_len, max_inner_len)。"""
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopAbs import TopAbs_WIRE

    outer = breptools.OuterWire(face)
    num_wires = 0
    inner_lengths = []
    exp = TopExp_Explorer(face, TopAbs_WIRE)
    while exp.More():
        wire = exp.Current()
        num_wires += 1
        if not wire.IsSame(outer):
            props = GProp_GProps()
            brepgprop.LinearProperties(wire, props)
            inner_lengths.append(props.Mass())
        exp.Next()
    inner = len(inner_lengths)
    return (
        num_wires,
        inner,
        min(inner_lengths) if inner_lengths else 0.0,
        max(inner_lengths) if inner_lengths else 0.0,
    )


def _num_edges(face: TopoDS_Face) -> int:
    cnt = 0
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        cnt += 1
        exp.Next()
    return cnt


# --------------------------------------------------------------------------- #
# 边类型（凸/凹/光滑）
# --------------------------------------------------------------------------- #
def _curve_type_index(edge) -> int:
    try:
        adaptor = BRepAdaptor_Curve(edge)
        name = _CURVE_MAP.get(adaptor.GetType(), "other")
    except RuntimeError:
        name = "other"
    return CURVE_TYPE_INDEX[name]


def _edge_length(edge) -> float:
    props = GProp_GProps()
    brepgprop.LinearProperties(edge, props)
    return props.Mass()


def _face_normal_at(face: TopoDS_Face, pnt) -> Optional[tuple]:
    """在投影到面的最近点处取法向（已按 orientation 校正）。"""
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier  # noqa: F401  (占位以提示依赖)
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf

    surf = BRep_Tool.Surface(face)
    proj = GeomAPI_ProjectPointOnSurf(pnt, surf)
    if proj.NbPoints() < 1:
        return None
    u, v = proj.Parameters(1, 0.0, 0.0)[1:] if False else proj.LowerDistanceParameters()
    adaptor = BRepAdaptor_Surface(face)
    props = BRepLProp_SLProps(adaptor, u, v, 1, 1e-6)
    if not props.IsNormalDefined():
        return None
    n = props.Normal()
    if face.Orientation() == 1:
        n.Reverse()
    return (n.X(), n.Y(), n.Z())


def _classify_edge(edge, f1: TopoDS_Face, f2: TopoDS_Face) -> tuple[float, float]:
    """返回 (edge_type ∈ {1,-1,0}, dihedral_angle 弧度)。

    取边中点，比较两面法向与连接方向：凸 -> +1，凹 -> -1，近共面 -> 0。
    """
    from OCC.Core.gp import gp_Pnt, gp_Vec

    adaptor = BRepAdaptor_Curve(edge)
    t = 0.5 * (adaptor.FirstParameter() + adaptor.LastParameter())
    mid = gp_Pnt()
    tan = gp_Vec()
    adaptor.D1(t, mid, tan)

    n1 = _face_normal_at(f1, mid)
    n2 = _face_normal_at(f2, mid)
    if n1 is None or n2 is None:
        return 0.0, 0.0

    v1 = gp_Vec(*n1)
    v2 = gp_Vec(*n2)
    dot = max(-1.0, min(1.0, v1.Dot(v2)))
    angle = math.acos(dot)
    if angle < math.radians(10):  # 近共面
        return 0.0, 0.0
    # 凸/凹符号：cross(n1,n2) 与边切向同向 -> 凹，反向 -> 凸（依实现/朝向需在 OCC 环境标定）
    cross = v1.Crossed(v2)
    sign = cross.Dot(tan)
    etype = 1.0 if sign < 0 else -1.0
    return etype, etype * angle


# --------------------------------------------------------------------------- #
# 顶层：shape -> FaceGraph
# --------------------------------------------------------------------------- #
def shape_to_graph(
    shape,
    model_id: str,
    labels: Optional[dict] = None,
    model_scale: Optional[float] = None,
) -> FaceGraph:
    """从 TopoDS_Shape 构建 FaceGraph。

    Args:
        shape: 实体/壳。
        model_id: 模型标识。
        labels: 可选 {face_id: (semantic, instance_id, operation)}，注入数据用。
        model_scale: 用于相对量的尺度（默认用包围盒对角线）。
    """
    from OCC.Core.gp import gp_Pnt  # noqa: F401

    fmap = build_face_map(shape)
    n_faces = fmap.Extent()

    if model_scale is None:
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib

        box = Bnd_Box()
        brepbndlib.Add(shape, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        model_scale = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) or 1.0

    # 第一遍：面级特征 + 总面积
    raw = {}
    total_area = 0.0
    for fid in range(1, n_faces + 1):
        face = topods.Face(fmap.FindKey(fid))
        area = _surface_area(face)
        total_area += area
        stype, radius, nx, ny, nz, mean_c, gauss_c = _surface_info(face)
        num_wires, inner, min_il, max_il = _wire_info(face)
        raw[fid] = {
            "_abs_area": area,
            "log_area": math.log1p(area),
            "compactness": 0.0,  # 见下
            "nx": nx, "ny": ny, "nz": nz,
            "mean_curvature": mean_c,
            "gauss_curvature": gauss_c,
            "radius": radius,
            "min_bbox_dim": _bbox_min_dim(face),
            "num_wires": num_wires,
            "inner_wire_count": inner,
            "min_inner_wire_len": min_il,
            "max_inner_wire_len": max_il,
            "num_edges": _num_edges(face),
            "surface_type": stype,
        }
        # 紧凑度 = 周长^2 / (4*pi*面积)
        perim = 0.0
        exp = TopExp_Explorer(face, TopAbs_EDGE)
        while exp.More():
            perim += _edge_length(exp.Current())
            exp.Next()
        raw[fid]["compactness"] = (perim * perim) / (4 * math.pi * area) if area > 1e-9 else 0.0

    total_area = total_area or 1.0

    # 边-面邻接
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    nodes = []
    for fid in range(1, n_faces + 1):
        f = raw[fid]
        f["rel_area"] = f["_abs_area"] / total_area
        f.pop("_abs_area")
        sem, inst, op = (0, -1, 0)
        if labels and fid in labels:
            sem, inst, op = labels[fid]
        nodes.append(FaceNode(face_id=fid, feats=f, semantic=sem, instance_id=inst, operation=op))

    # 邻域面积比
    area_of = {fid: raw[fid].get("rel_area", 0.0) * total_area for fid in raw}

    edges = []
    seen = set()
    for i in range(1, edge_face_map.Extent() + 1):
        edge = edge_face_map.FindKey(i)
        face_list = edge_face_map.FindFromIndex(i)
        faces = [face_list.First()] if face_list.Extent() == 1 else list(_iter_faces(face_list))
        if len(faces) < 2:
            continue
        # 同一对面可能有多条边；这里逐边建图（保留多重连接信息也可，简单起见去重面对）
        f1, f2 = topods.Face(faces[0]), topods.Face(faces[1])
        id1, id2 = fmap.FindIndex(f1), fmap.FindIndex(f2)
        if id1 == id2:
            continue
        key = (min(id1, id2), max(id1, id2))
        if key in seen:
            continue
        seen.add(key)
        etype, dih = _classify_edge(edge, f1, f2)
        length = _edge_length(edge)
        a1, a2 = area_of.get(id1, 1.0), area_of.get(id2, 1.0)
        area_ratio = min(a1, a2) / max(a1, a2) if max(a1, a2) > 1e-9 else 1.0
        same_inst = 0
        if labels and id1 in labels and id2 in labels:
            _, inst1, _ = labels[id1]
            _, inst2, _ = labels[id2]
            same_inst = int(inst1 >= 0 and inst1 == inst2)
        edges.append(
            EdgeRecord(
                src=id1, dst=id2,
                feats={
                    "edge_type": etype,
                    "dihedral_angle": dih,
                    "shared_len": length,
                    "rel_shared_len": length / model_scale,
                    "area_ratio": area_ratio,
                    "curve_type": _curve_type_index(edge),
                },
                same_instance=same_inst,
            )
        )

    # neighbor_area_ratio 后处理
    neigh: dict[int, list[int]] = {n.face_id: [] for n in nodes}
    for e in edges:
        neigh[e.src].append(e.dst)
        neigh[e.dst].append(e.src)
    for n in nodes:
        ns = [area_of.get(m, 0.0) for m in neigh[n.face_id]]
        mean_n = sum(ns) / len(ns) if ns else area_of.get(n.face_id, 1.0)
        own = area_of.get(n.face_id, 0.0)
        n.feats["neighbor_area_ratio"] = own / mean_n if mean_n > 1e-9 else 1.0

    return FaceGraph(model_id=model_id, nodes=nodes, edges=edges,
                     meta={"source": "occ", "num_faces": n_faces})


def _iter_faces(face_list):
    from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape

    it = TopTools_ListIteratorOfListOfShape(face_list)
    while it.More():
        yield it.Value()
        it.Next()


def step_to_graph(path: str, model_id: Optional[str] = None) -> FaceGraph:
    shape = read_step(path)
    return shape_to_graph(shape, model_id or path)
