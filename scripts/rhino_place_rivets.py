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

import json
import os

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
RIVET_LAYER = "Rivets"


def ensure_rivet_layer():
    if not rs.IsLayer(RIVET_LAYER):
        rs.AddLayer(RIVET_LAYER, (230, 40, 30))
    return RIVET_LAYER


def default_json_path():
    p = sc.doc.Path
    if p:
        return os.path.splitext(p)[0] + "_rivets.json"
    return os.path.join(os.path.expanduser("~"), "Desktop", "rhino_rivets.json")


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


def adjacent_face(objref):
    """边所贴的面（取第一个相邻面）。"""
    edge = objref.Edge()
    brep = objref.Brep()
    if edge and brep:
        adj = edge.AdjacentFaces()
        if adj and len(adj) > 0:
            return brep.Faces[adj[0]]
    return None


def _normal(face, u, v):
    n = face.NormalAt(u, v)
    if face.OrientationIsReversed:
        n.Reverse()
    return n


def place_on_face(face, pt, tangent, inset):
    """把边上的点沿“面内、垂直于边”的方向内缩 inset，落回面上。

    返回 (中心点 Point3d, 外法向 Vector3d)。inset<=0 时就用边上的点。
    """
    if face is None:
        return pt, Vector3d.ZAxis
    ok, u, v = face.ClosestPoint(pt)
    if not ok:
        return pt, Vector3d.ZAxis
    n = _normal(face, u, v)
    if inset and inset > 0 and tangent is not None:
        t = Vector3d(tangent)
        if t.Unitize():
            b = Vector3d.CrossProduct(n, t)        # 面内、垂直于边的方向
            if b.Unitize():
                for s in (1.0, -1.0):              # 朝面内的一侧
                    cand = Point3d(pt) + b * (s * inset)
                    ok2, u2, v2 = face.ClosestPoint(cand)
                    if ok2 and face.IsPointOnFace(u2, v2) == Rhino.Geometry.PointFaceRelation.Interior:
                        return face.PointAt(u2, v2), _normal(face, u2, v2)
    return face.PointAt(u, v), n


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
    """沿边等距取 count 个点，返回 [(点, 切向)]（含两端）。"""
    crv = objref.Curve()
    if crv is None:
        return []
    params = crv.DivideByCount(max(count - 1, 1), True)
    if params is None or len(params) == 0:
        return [(crv.PointAtStart, crv.TangentAtStart),
                (crv.PointAtEnd, crv.TangentAtEnd)]
    return [(crv.PointAt(t), crv.TangentAt(t)) for t in params]


def connected_faces(brep, seed_faces):
    """从若干 face 出发，沿共享边做 BFS，返回同一连通实体（lump）的全部 face 下标集合。"""
    seen = set()
    stack = list(seed_faces)
    while stack:
        fi = stack.pop()
        if fi in seen or fi < 0 or fi >= brep.Faces.Count:
            continue
        seen.add(fi)
        for af in brep.Faces[fi].AdjacentFaces():
            if af not in seen:
                stack.append(af)
    return seen


def all_components(brep):
    """把 brep 的所有 face 按连通性分成若干实体(lump)，返回 [set(face index)]。"""
    seen = set()
    comps = []
    for i in range(brep.Faces.Count):
        if i in seen:
            continue
        comp = connected_faces(brep, [i])
        seen |= comp
        comps.append(comp)
    return comps


def bbox_dist_to_point(bb, pt):
    """点到包围盒的距离（盒内为 0）。"""
    dx = max(bb.Min.X - pt.X, 0.0, pt.X - bb.Max.X)
    dy = max(bb.Min.Y - pt.Y, 0.0, pt.Y - bb.Max.Y)
    dz = max(bb.Min.Z - pt.Z, 0.0, pt.Z - bb.Max.Z)
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _sub_brep(brep, face_indices):
    """从 brep 抽取给定 face 子集为新 brep（保留其它实体用）。"""
    if not face_indices:
        return None
    lst = sorted(face_indices)
    try:
        import System
        sub = brep.DuplicateSubBrep(System.Array[int](lst))
        if sub:
            return sub
    except Exception as ex:                     # noqa: BLE001
        print("  DuplicateSubBrep(Array) 失败:", ex)
    try:
        return brep.DuplicateSubBrep(lst)
    except Exception as ex:                      # noqa: BLE001
        print("  DuplicateSubBrep(list) 失败:", ex)
    return None


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
    inset = rs.GetReal("铆钉离边界的内缩距离（0=正好在边上）", round(radius * 2.0, 4), 0.0)
    if inset is None:
        return
    res = rs.GetBoolean("是否布尔融合到实体", [("融合", "否", "是")], [True])
    do_union = True if not res else res[0]

    # 按所属对象分组：铆钉、母 brep、被选边所贴的 face 下标（用于定位 lump）
    rivets_by_parent = {}      # pid -> [rivet Brep]
    parent_brep = {}           # pid -> Brep
    seeds_by_parent = {}       # pid -> set(face index)
    records = []               # 放钉记录（中心/法向/形状/半径），即时写出真值
    n_made = 0
    for ref in refs:
        pid = ref.ObjectId
        if pid not in parent_brep:
            parent_brep[pid] = ref.Brep()
        edge = ref.Edge()
        if edge is not None:
            for fi in edge.AdjacentFaces():
                seeds_by_parent.setdefault(pid, set()).add(fi)
        face = adjacent_face(ref)
        for pt, tan in edge_points(ref, count):
            center, n = place_on_face(face, pt, tan, inset)
            if shape == "frustum":
                riv = make_frustum(center, n, r_base=radius, r_top=radius * 0.5,
                                   height=radius * 1.2, sink=radius * 0.3)
            else:
                riv = make_dome(center, radius)
            if riv:
                rivets_by_parent.setdefault(pid, []).append(riv)
                records.append({
                    "id": len(records),
                    "shape": shape,
                    "radius": radius,
                    "center": [center.X, center.Y, center.Z],
                    "normal": [n.X, n.Y, n.Z],
                    "parent_object": str(pid),
                })
                n_made += 1

    if n_made == 0:
        print("未生成铆钉。")
        return

    # 即时写出铆钉记录（真值），不依赖事后导出/检测
    json_path = default_json_path()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"source": "rhino", "shape": shape, "radius": radius,
                   "count": len(records), "rivets": records}, f,
                  ensure_ascii=False, indent=2)
    print("铆钉记录(真值) -> %s（%d 个）" % (json_path, len(records)))

    if not do_union:
        layer = ensure_rivet_layer()
        for rl in rivets_by_parent.values():
            for riv in rl:
                oid = sc.doc.Objects.AddBrep(riv)
                if oid:
                    rs.ObjectLayer(oid, layer)
        sc.doc.Views.Redraw()
        print("已添加 %d 个铆钉到图层 '%s'（未融合）。可手动 _BooleanUnion。" % (n_made, layer))
        return

    n_union = 0
    for pid, rivets in rivets_by_parent.items():
        brep = parent_brep[pid]
        if brep is None:
            for riv in rivets:
                sc.doc.Objects.AddBrep(riv)
            continue

        comps = all_components(brep)             # 母 brep 的所有独立实体(lump)
        print("诊断：对象 %s 面数=%d，独立实体(lump)=%d，铆钉=%d"
              % (pid, brep.Faces.Count, len(comps), len(rivets)))

        # 单实体：直接整体布尔
        if len(comps) <= 1:
            union = Brep.CreateBooleanUnion([brep] + rivets, TOL)
            if union is not None and len(union) > 0:
                sc.doc.Objects.Replace(pid, union[0])
                for i in range(1, len(union)):
                    sc.doc.Objects.AddBrep(union[i])
                n_union += len(rivets)
            else:
                print("  布尔失败，直接添加铆钉（原件保留）。")
                for riv in rivets:
                    sc.doc.Objects.AddBrep(riv)
            continue

        # 多实体：先把所有 lump 都成功拆出来，否则绝不替换母对象（防丢件）
        lumps = [_sub_brep(brep, c) for c in comps]
        n_ok = sum(1 for lp in lumps if lp is not None)
        print("  lump 拆分成功 %d/%d" % (n_ok, len(lumps)))
        if n_ok < len(lumps):
            print("  有 lump 拆分失败 -> 不替换母对象，仅叠加铆钉（不丢件，但未融合）。")
            for riv in rivets:
                sc.doc.Objects.AddBrep(riv)
            continue

        # 铆钉只并入最近的 lump，其余 lump 原样保留
        bbs = [lp.GetBoundingBox(True) for lp in lumps]
        assign = {i: [] for i in range(len(lumps))}
        for riv in rivets:
            ctr = riv.GetBoundingBox(True).Center
            best, bd = None, 1e18
            for i, b in enumerate(bbs):
                d = bbox_dist_to_point(b, ctr)
                if d < bd:
                    bd, best = d, i
            if best is not None:
                assign[best].append(riv)

        out = []
        for i, lp in enumerate(lumps):
            if assign[i]:
                u = Brep.CreateBooleanUnion([lp] + assign[i], TOL)
                if u is not None and len(u) > 0:
                    out.extend(u[k] for k in range(len(u)))
                    n_union += len(assign[i])
                else:
                    out.append(lp)
                    out.extend(assign[i])
            else:
                out.append(lp)                   # 无钉的 lump 原样保留（机身/机头等）

        # 全部 lump 都已收进 out，才替换母对象，保证不丢件
        sc.doc.Objects.Replace(pid, out[0])
        for k in range(1, len(out)):
            sc.doc.Objects.AddBrep(out[k])
        print("  完成：母对象拆分重组为 %d 个对象。" % len(out))

    sc.doc.Views.Redraw()
    print("完成：生成 %d 个铆钉（形状=%s，半径=%.4f），融合 %d 个。" %
          (n_made, shape, radius, n_union))


if __name__ == "__main__":
    main()
