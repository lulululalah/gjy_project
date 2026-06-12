"""注入锚点（rivet 放置位置）的 STP 读写与法向解析。

锚点文件是注入管线的**输入**：它只描述「在哪些位置放铆钉」，不含任何启发式
选面逻辑。当前由临时脚本 scripts/make_anchors.py 生成，**后期将由深度学习模型
预测后写出同一份 STP**，注入管线无需改动。

格式：STEP 文件，内含若干独立顶点（TopoDS_Vertex），每个顶点 = 一个铆钉中心。
法向不存进文件，注入时通过「投影到最近面」就地求得（见 resolve_anchor）。

⚠️ 需要 pythonocc-core 运行环境。
"""

from __future__ import annotations

from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound, topods
from OCC.Core.gp import gp_Pnt


def write_anchors_stp(points, path: str) -> None:
    """把锚点 (x,y,z) 列表写成 STP（独立顶点的 compound）。"""
    comp = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(comp)
    for (x, y, z) in points:
        v = BRepBuilderAPI_MakeVertex(gp_Pnt(float(x), float(y), float(z))).Vertex()
        builder.Add(comp, v)
    writer = STEPControl_Writer()
    writer.Transfer(comp, STEPControl_AsIs)
    if writer.Write(path) != 1:
        raise IOError(f"写入锚点 STP 失败: {path}")


def read_anchors_stp(path: str, dedup_tol: float = 1e-6):
    """从 STP 读回锚点 (x,y,z) 列表（按坐标去重）。"""
    reader = STEPControl_Reader()
    if reader.ReadFile(path) != 1:
        raise IOError(f"读取锚点 STP 失败: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    pts = []
    seen = set()
    exp = TopExp_Explorer(shape, TopAbs_VERTEX)
    while exp.More():
        p = BRep_Tool.Pnt(topods.Vertex(exp.Current()))
        key = (round(p.X() / dedup_tol), round(p.Y() / dedup_tol), round(p.Z() / dedup_tol))
        if key not in seen:
            seen.add(key)
            pts.append((p.X(), p.Y(), p.Z()))
        exp.Next()
    return pts


def resolve_anchor(shape, xyz):
    """把锚点投影到模型最近面，返回 (表面点 gp_Pnt, 外法向 gp_Dir)。

    锚点文件只存位置；铆钉的朝向由它所贴的面决定，故注入时就地求法向。
    """
    target = gp_Pnt(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    best = None  # (dist, face, u, v)
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        surf = BRep_Tool.Surface(face)
        proj = GeomAPI_ProjectPointOnSurf(target, surf)
        if proj.NbPoints() >= 1:
            d = proj.LowerDistance()
            if best is None or d < best[0]:
                u, v = proj.LowerDistanceParameters()
                best = (d, face, u, v)
        exp.Next()
    if best is None:
        return None, None
    _, face, u, v = best
    props = BRepLProp_SLProps(BRepAdaptor_Surface(face), u, v, 1, 1e-6)
    if not props.IsNormalDefined():
        return None, None
    pnt = props.Value()
    n = props.Normal()
    if face.Orientation() == 1:  # REVERSED
        n.Reverse()
    return pnt, n
