#! python 3
"""在 Rhino 8 里交互式地沿选中的边放置铆钉，并布尔融入实体。

运行方式（Rhino 8）：
    命令行输入 _ScriptEditor  ->  打开本文件 -> 运行（Python 3）
    或：_RunPythonScript 选择本文件

交互流程：
    1. 选择若干**边**（face 的 edge，可多选，可来自不同实体）。
    2. 选铆钉形状：dome（半球小圆钉）/ frustum（梯台）。
    3. 设半径、每条边上的数量、是否布尔融合。
    4. 沿每条边等距取点，按所贴面的法向生成铆钉，按所属实体布尔 Union。

之后可 _Export 选中实体为 STEP(.stp)，喂给本项目的识别/去除管线
（occ/extract.py 用“球面=铆钉”即可标出 dome 铆钉）。
"""

import scriptcontext as sc
import Rhino
import rhinoscriptsyntax as rs
from Rhino.Geometry import (
    Brep,
    Circle,
    LoftType,
    Plane,
    Point3d,
    Sphere,
    Vector3d,
)

TOL = sc.doc.ModelAbsoluteTolerance


# --------------------------------------------------------------------------- #
def select_edges():
    """让用户选边（brep 子对象），返回 ObjRef 列表。"""
    go = Rhino.Input.Custom.GetObject()
    go.SetCommandPrompt("选择放置铆钉的边（可多选，回车结束）")
    go.GeometryFilter = Rhino.DocObjects.ObjectType.Curve
    go.SubObjectSelect = True
    go.EnablePreSelect(True, True)
    res = go.GetMultiple(1, 0)
    if res != Rhino.Input.GetResult.Object:
        return []
    return [go.Object(i) for i in range(go.ObjectCount)]


def suggest_radius(refs):
    """用所选实体包围盒对角线 *0.0012 作为默认半径建议。"""
    bbox = None
    for r in refs:
        brep = r.Brep()
        if not brep:
            continue
        bb = brep.GetBoundingBox(True)
        bbox = bb if bbox is None else Rhino.Geometry.BoundingBox.Union(bbox, bb)
    if bbox is None:
        return 0.03
    return max(bbox.Diagonal.Length * 0.0012, 1e-4)


def normal_at(objref, pt):
    """边上一点处、所贴面的外法向。"""
    edge = objref.Edge()
    brep = objref.Brep()
    if edge and brep:
        adj = edge.AdjacentFaces()
        if adj and len(adj) > 0:
            face = brep.Faces[adj[0]]
            ok, u, v = face.ClosestPoint(pt)
            if ok:
                n = face.NormalAt(u, v)
                if face.OrientationIsReversed:
                    n.Reverse()
                return n
    return Vector3d.ZAxis


def make_dome(center, radius):
    """半球小圆钉：球心落在表面，布尔后外侧呈半球凸起。"""
    return Brep.CreateFromSphere(Sphere(center, radius))


def make_frustum(center, normal, r_base, r_top, height, sink):
    """梯台：底大顶小的圆台；底面略沉入表面以保证布尔压印。"""
    n = Vector3d(normal)
    n.Unitize()
    base_c = center - n * sink
    top_c = center + n * height
    cb = Circle(Plane(base_c, n), r_base).ToNurbsCurve()
    ct = Circle(Plane(top_c, n), r_top).ToNurbsCurve()
    loft = Brep.CreateFromLoft([cb, ct], Point3d.Unset, Point3d.Unset, LoftType.Straight, False)
    if not loft or len(loft) == 0:
        return None
    capped = loft[0].CapPlanarHoles(TOL)
    return capped if capped else loft[0]


def edge_points(objref, count):
    """沿边等距取 count 个点（含两端）。"""
    crv = objref.Curve()
    if crv is None:
        return []
    params = crv.DivideByCount(max(count - 1, 1), True)
    if not params:
        return [crv.PointAtStart, crv.PointAtEnd]
    return [crv.PointAt(t) for t in params]


# --------------------------------------------------------------------------- #
def main():
    refs = select_edges()
    if not refs:
        print("未选择任何边。")
        return

    shape = rs.GetString("铆钉形状", "dome", ["dome", "frustum"])
    if not shape:
        return
    shape = shape.lower()

    default_r = suggest_radius(refs)
    radius = rs.GetReal("铆钉半径（底面/球）", round(default_r, 4), 1e-5)
    if radius is None:
        return
    count = rs.GetInteger("每条边上的铆钉数", 10, 1)
    if count is None:
        return
    res = rs.GetBoolean("是否布尔融合到实体", [("融合", "否", "是")], [True])
    do_union = True if not res else res[0]

    # 按所属实体分组收集铆钉
    rivets_by_parent = {}      # parent ObjectId -> [rivet Brep]
    parent_ref = {}            # parent ObjectId -> ObjRef（取其 Brep）
    n_made = 0
    for ref in refs:
        pid = ref.ObjectId
        parent_ref[pid] = ref
        for pt in edge_points(ref, count):
            n = normal_at(ref, pt)
            if shape == "frustum":
                riv = make_frustum(pt, n, r_base=radius, r_top=radius * 0.5,
                                   height=radius * 1.2, sink=radius * 0.3)
            else:
                riv = make_dome(pt, radius)
            if riv:
                rivets_by_parent.setdefault(pid, []).append(riv)
                n_made += 1

    if n_made == 0:
        print("未生成铆钉。")
        return

    if not do_union:
        for rl in rivets_by_parent.values():
            for riv in rl:
                sc.doc.Objects.AddBrep(riv)
        sc.doc.Views.Redraw()
        print("已添加 %d 个铆钉（未融合）。可手动 _BooleanUnion。" % n_made)
        return

    n_union = 0
    for pid, rivets in rivets_by_parent.items():
        parent_brep = parent_ref[pid].Brep()
        if parent_brep is None:
            for riv in rivets:
                sc.doc.Objects.AddBrep(riv)
            continue
        union = Brep.CreateBooleanUnion([parent_brep] + rivets, TOL)
        if union and len(union) > 0:
            sc.doc.Objects.Replace(pid, union[0])
            for extra in union[1:]:
                sc.doc.Objects.AddBrep(extra)
            n_union += len(rivets)
        else:
            print("实体 %s 布尔失败，改为直接添加铆钉。" % pid)
            for riv in rivets:
                sc.doc.Objects.AddBrep(riv)

    sc.doc.Views.Redraw()
    print("完成：生成 %d 个铆钉（形状=%s，半径=%.4f），融合 %d 个。" %
          (n_made, shape, radius, n_union))


if __name__ == "__main__":
    main()
