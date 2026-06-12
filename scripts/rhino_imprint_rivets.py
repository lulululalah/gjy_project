#! python 3
"""在 Rhino 8 里：选边后，直接在所贴面的【拓扑/参数域】上压印铆钉锚点（不做布尔）。

思路（不通过布尔运算）：
    - 沿选中的边、在所贴面上取若干锚点（从边界向面内缩一点）。
    - 在每个锚点处生成一个小圆，PullToBrepFace 把圆“贴”到该面的曲面上。
    - 用这些圆 Brep.Split 切分该面，再 JoinBreps 合回一个 brep。
      => 每个铆钉变成该面上的一个**子面(disk)+内环(wire)**，直接进入 BRep 拓扑。
    - 全程无布尔、无悬空小球；锚点一定落在面上（位置可靠）。

之后：每个铆钉就是一个小圆盘子面，可被识别/去除（去除=删子面并补平）。
放钉记录仍写 JSON（中心/法向/半径），可用于 label_rivets --centers 打标。

运行：Rhino 命令 _-RunPythonScript "….../scripts/rhino_imprint_rivets.py"
（用带减号的 _-RunPythonScript，确保从磁盘读最新版）
"""

import json
import os

import scriptcontext as sc
import Rhino
import rhinoscriptsyntax as rs
from Rhino.Geometry import Circle, Curve, Plane, Point3d, Vector3d

SCRIPT_VERSION = "imprint-v1"
TOL = sc.doc.ModelAbsoluteTolerance


def select_edges():
    go = Rhino.Input.Custom.GetObject()
    go.SetCommandPrompt("选择要压印铆钉的边（可多选，回车结束）")
    go.GeometryFilter = Rhino.DocObjects.ObjectType.Curve
    go.SubObjectSelect = True
    go.EnablePreSelect(True, True)
    if go.GetMultiple(1, 0) != Rhino.Input.GetResult.Object:
        return []
    return [go.Object(i) for i in range(go.ObjectCount)]


def adjacent_face(objref):
    edge = objref.Edge()
    brep = objref.Brep()
    if edge and brep:
        adj = edge.AdjacentFaces()
        if adj and len(adj) > 0:
            return brep.Faces[adj[0]]
    return None


def edge_points(objref, count):
    crv = objref.Curve()
    if crv is None:
        return []
    params = crv.DivideByCount(max(count - 1, 1), True)
    if params is None or len(params) == 0:
        return [(crv.PointAtStart, crv.TangentAtStart)]
    return [(crv.PointAt(t), crv.TangentAt(t)) for t in params]


def place_on_face(face, pt, tangent, inset):
    """把边上的点沿面内、垂直于边方向内缩 inset，落回面上。返回 (中心, 法向)。"""
    ok, u, v = face.ClosestPoint(pt)
    if not ok:
        return pt, Vector3d.ZAxis
    n = face.NormalAt(u, v)
    if face.OrientationIsReversed:
        n.Reverse()
    if inset and inset > 0 and tangent is not None:
        t = Vector3d(tangent)
        if t.Unitize():
            b = Vector3d.CrossProduct(n, t)
            if b.Unitize():
                for s in (1.0, -1.0):
                    cand = Point3d(pt) + b * (s * inset)
                    ok2, u2, v2 = face.ClosestPoint(cand)
                    if ok2 and face.IsPointOnFace(u2, v2) == Rhino.Geometry.PointFaceRelation.Interior:
                        return face.PointAt(u2, v2), face.NormalAt(u2, v2)
    return face.PointAt(u, v), n


def main():
    print("=== rhino_imprint_rivets %s ===" % SCRIPT_VERSION)
    refs = select_edges()
    if not refs:
        print("未选择任何边。")
        return
    print("选中边数 = %d" % len(refs))

    radius = rs.GetReal("铆钉(压印圆)半径", 0.02, 1e-6)
    if radius is None:
        return
    count = rs.GetInteger("每条边上的铆钉数", 10, 1)
    if count is None:
        return
    inset = rs.GetReal("离边界内缩距离", round(radius * 2.0, 4), 0.0)
    if inset is None:
        return

    # 按所属对象收集压印圆（已贴到面上）+ 记录
    cutters_by_pid = {}
    brep_by_pid = {}
    records = []
    for ref in refs:
        pid = ref.ObjectId
        if pid not in brep_by_pid:
            brep_by_pid[pid] = ref.Brep()
        face = adjacent_face(ref)
        if face is None:
            continue
        for pt, tan in edge_points(ref, count):
            center, n = place_on_face(face, pt, tan, inset)
            circle = Circle(Plane(center, n), radius).ToNurbsCurve()
            pulled = Curve.PullToBrepFace(face, circle, TOL)   # 把圆贴到曲面上
            if pulled:
                for c in pulled:
                    cutters_by_pid.setdefault(pid, []).append(c)
                records.append({
                    "id": len(records), "shape": "disk", "radius": radius,
                    "center": [center.X, center.Y, center.Z],
                    "normal": [n.X, n.Y, n.Z], "parent_object": str(pid),
                })

    if not records:
        print("未生成任何压印圆（PullToBrepFace 失败？检查半径/面是否选对）。")
        return

    # 写记录 JSON（与之前一致，可用于 label_rivets --centers）
    doc_path = sc.doc.Path
    json_path = (os.path.splitext(doc_path)[0] + "_rivets.json") if doc_path \
        else os.path.join(os.path.expanduser("~"), "Desktop", "rhino_rivets.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"source": "rhino-imprint", "radius": radius,
                   "count": len(records), "rivets": records}, f, ensure_ascii=False, indent=2)
    print("铆钉记录(真值) -> %s（%d 个）" % (json_path, len(records)))

    # 逐对象：Split 压印 + JoinBreps 合回一个 brep（拓扑里多出子面/内环，无布尔）
    n_imprinted = 0
    for pid, cutters in cutters_by_pid.items():
        brep = brep_by_pid[pid]
        pieces = brep.Split(cutters, TOL)
        if pieces is None or len(pieces) == 0:
            print("  对象 %s Split 无结果（圆可能未落在面内）-> 跳过，原件保留。" % pid)
            continue
        joined = Rhino.Geometry.Brep.JoinBreps(pieces, TOL)
        result = joined[0] if (joined and len(joined) > 0) else pieces[0]
        nfaces_before = brep.Faces.Count
        sc.doc.Objects.Replace(pid, result)
        n_imprinted += len(cutters)
        print("  对象 %s：面数 %d -> %d，压印圆 %d" % (pid, nfaces_before, result.Faces.Count, len(cutters)))

    sc.doc.Views.Redraw()
    print("完成：压印 %d 个铆钉锚点（拓扑子面/内环，无布尔）。" % n_imprinted)


if __name__ == "__main__":
    main()
