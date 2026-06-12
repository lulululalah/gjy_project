"""OCC 几何后端集成测试。

需要 pythonocc-core；缺失则整体跳过（不影响纯 ML 测试）。
运行：/opt/anaconda3/envs/occ/bin/python -m pytest tests/test_occ.py -q
"""

import random

import pytest

pytest.importorskip("OCC.Core.BRepPrimAPI", reason="需要 pythonocc-core 环境")

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox  # noqa: E402

from brep_defeature.occ.extract import shape_to_graph  # noqa: E402
from brep_defeature.occ.inject import inject_features, make_training_sample  # noqa: E402
from brep_defeature.occ.defeature import remove_instances, recovery_error  # noqa: E402
from brep_defeature.schema import (  # noqa: E402
    EDGE_FEATURE_DIM,
    NODE_CONT_FEATURES,
    NODE_FEATURE_DIM,
    Semantic,
)


def _box(dx=20.0, dy=20.0, dz=5.0):
    return BRepPrimAPI_MakeBox(dx, dy, dz).Shape()


# --------------------------------------------------------------------------- #
# 提取
# --------------------------------------------------------------------------- #
def test_box_extraction_basic():
    g = shape_to_graph(_box(), "box")
    assert g.num_nodes == 6, f"立方体应有 6 个面, got {g.num_nodes}"
    # 所有面应为平面
    from brep_defeature.schema import SURFACE_TYPE_INDEX

    for n in g.nodes:
        assert n.feats["surface_type"] == SURFACE_TYPE_INDEX["plane"]
    # 每个节点含全部连续特征键
    for n in g.nodes:
        for key in NODE_CONT_FEATURES:
            assert key in n.feats, f"缺特征 {key}"
    # 立方体相邻面共 12 条边（去重面对后每对面 1 条）
    assert len(g.edges) >= 6


def test_extraction_to_pyg_dims():
    pytest.importorskip("torch_geometric", reason="PyG 不在 OCC 环境中")
    from brep_defeature.graph.build import to_pyg_data
    from brep_defeature.schema import NormStats

    g = shape_to_graph(_box(), "box")
    data = to_pyg_data(g, NormStats.fit([g]))
    assert data.x.shape[1] == NODE_FEATURE_DIM
    assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM


# --------------------------------------------------------------------------- #
# 注入 + 真值
# --------------------------------------------------------------------------- #
def test_inject_rivet_labels():
    shape, labels, records = inject_features(
        _box(), random.Random(0), n_rivets=1, n_holes=0,
        rivet_radius=(1.0, 1.5),
    )
    assert any(r.semantic == int(Semantic.RIVET) for r in records), "应记录一个铆钉注入"
    rivet_faces = [fid for fid, (sem, _, _) in labels.items() if sem == int(Semantic.RIVET)]
    assert len(rivet_faces) >= 1, "至少有一个面被标为铆钉"


def test_inject_hole_labels():
    shape, labels, records = inject_features(
        _box(dz=8.0), random.Random(1), n_rivets=0, n_holes=1,
        hole_radius=(1.0, 1.5),
    )
    assert any(r.semantic == int(Semantic.HOLE) for r in records), "应记录一个孔注入"
    hole_faces = [fid for fid, (sem, _, _) in labels.items() if sem == int(Semantic.HOLE)]
    assert len(hole_faces) >= 1, "至少有一个面被标为孔"


def test_make_training_sample_graph_labeled():
    g, shape, records = make_training_sample(
        _box(), "sample", random.Random(2), n_rivets=2, n_holes=1,
        rivet_radius=(1.0, 1.5), hole_radius=(1.0, 1.5),
    )
    sems = {n.semantic for n in g.nodes}
    assert int(Semantic.BACKGROUND) in sems
    assert int(Semantic.RIVET) in sems or int(Semantic.HOLE) in sems
    # 特征面有合法实例 id
    for n in g.nodes:
        if n.semantic != int(Semantic.BACKGROUND):
            assert n.instance_id >= 0


# --------------------------------------------------------------------------- #
# 去除 + 恢复误差
# --------------------------------------------------------------------------- #
def test_anchor_stp_roundtrip_and_normal(tmp_path):
    from brep_defeature.occ.anchors import read_anchors_stp, resolve_anchor, write_anchors_stp

    pts = [(1.0, 2.0, 3.0), (4.5, 5.5, 6.5), (10.0, 0.0, 0.0)]
    p = tmp_path / "anchors.stp"
    write_anchors_stp(pts, str(p))
    back = read_anchors_stp(str(p))
    assert len(back) == len(pts)
    for w, r in zip(sorted(pts), sorted(back)):
        assert all(abs(a - b) < 1e-3 for a, b in zip(w, r))

    # 锚点投影到立方体顶面，法向应为 +Z
    pnt, n = resolve_anchor(_box(20.0, 20.0, 10.0), (10.0, 10.0, 10.2))
    assert pnt is not None
    assert abs(pnt.Z() - 10.0) < 1e-3
    assert n.Z() > 0.9


def test_protrusions_fuse_into_topology():
    """三类凸起都应能 Fuse 进立方体、增大体积、且结果为有效实体（拓扑融合）。"""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.gp import gp_Dir, gp_Pnt

    from brep_defeature.occ.defeature import _volume
    from brep_defeature.occ.extract import build_face_map
    from brep_defeature.occ.features import FEATURE_KINDS, make_protrusion

    base = _box(20.0, 20.0, 10.0)
    top_pnt = gp_Pnt(10.0, 10.0, 10.0)
    up = gp_Dir(0, 0, 1)
    for kind in FEATURE_KINDS:
        solid = make_protrusion(kind, top_pnt, up, size=1.0)
        fuse = BRepAlgoAPI_Fuse(base, solid)
        fuse.Build()
        assert fuse.IsDone(), f"{kind} Fuse 未完成"
        result = fuse.Shape()
        assert BRepCheck_Analyzer(result).IsValid(), f"{kind} 结果非有效实体"
        assert _volume(result) > _volume(base), f"{kind} 未增大体积"
        # 顶面被压印分割 -> 面数增加（拓扑确实改变）
        assert build_face_map(result).Size() > build_face_map(base).Size()


def test_defeature_rivet_restores_shape():
    base = _box()
    from brep_defeature.occ.defeature import _volume

    base_vol = _volume(base)

    shaped, labels, records = inject_features(
        base, random.Random(3), n_rivets=1, n_holes=0, rivet_radius=(1.0, 1.5),
    )
    # 注入后体积应增加（凸起）
    assert _volume(shaped) > base_vol

    # 组装实例（按真值）交给去除
    inst_faces = {}
    for fid, (sem, inst, op) in labels.items():
        inst_faces.setdefault(inst, {"instance_id": inst, "operation": op, "face_ids": []})
        inst_faces[inst]["face_ids"].append(fid)
    result, report = remove_instances(shaped, list(inst_faces.values()))

    assert report.valid, "去除结果应为有效实体"
    # 去除后体积应接近原始干净体积
    err = recovery_error(result, base)
    assert err.volume_diff_ratio < 0.05, f"恢复体积差过大: {err.volume_diff_ratio:.3f}"
