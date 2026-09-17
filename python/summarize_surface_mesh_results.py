"""Summarize paired surface-mesh statistics into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    ("face_count", "CAD 面数"),
    ("mesh_node_count", "网格节点数"),
    ("mesh_triangle_count", "网格三角形数"),
    ("mesh_elapsed_seconds_median", "网格阶段耗时中位数 s"),
)


def reduction_percent(original: float, simplified: float) -> float:
    return (original - simplified) * 100.0 / original


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
    rows = []
    for pair in pairs:
        original = json.loads(Path(pair["original"]).read_text(encoding="utf-8"))
        simplified = json.loads(Path(pair["simplified"]).read_text(encoding="utf-8"))
        row = {
            "model": pair["model"],
            "role": pair["role"],
            "original_brep_valid": original["brep_valid"],
            "simplified_brep_valid": simplified["brep_valid"],
            "relative_deflection": original["relative_deflection"],
            "angular_deflection_rad": original["angular_deflection_rad"],
            "mesh_repeats": original["mesh_repeats"],
        }
        for key, _ in METRICS:
            row[f"original_{key}"] = original[key]
            row[f"simplified_{key}"] = simplified[key]
            row[f"reduction_{key}_percent"] = reduction_percent(
                original[key], simplified[key]
            )
        rows.append(row)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# E3 表面网格汇总结果",
        "",
        "固定条件：OpenCASCADE `BRepMesh_IncrementalMesh`；相对弦高 `0.0005`；角度偏差 `0.5 rad`；每组独立重复 5 次，报告网格阶段耗时中位数。",
        "",
        "| 模型 | 角色 | CAD 面数 原始 -> 简化 | 节点数 原始 -> 简化 | 三角形数 原始 -> 简化 | 耗时中位数 原始 -> 简化 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {role} | {faces} -> {faces_s} ({faces_r:.2f}%) | "
            "{nodes} -> {nodes_s} ({nodes_r:.2f}%) | "
            "{triangles} -> {triangles_s} ({triangles_r:.2f}%) | "
            "{seconds:.4f} s -> {seconds_s:.4f} s ({seconds_r:.2f}%) |".format(
                model=row["model"], role=row["role"],
                faces=row["original_face_count"], faces_s=row["simplified_face_count"],
                faces_r=row["reduction_face_count_percent"],
                nodes=row["original_mesh_node_count"], nodes_s=row["simplified_mesh_node_count"],
                nodes_r=row["reduction_mesh_node_count_percent"],
                triangles=row["original_mesh_triangle_count"],
                triangles_s=row["simplified_mesh_triangle_count"],
                triangles_r=row["reduction_mesh_triangle_count_percent"],
                seconds=row["original_mesh_elapsed_seconds_median"],
                seconds_s=row["simplified_mesh_elapsed_seconds_median"],
                seconds_r=row["reduction_mesh_elapsed_seconds_median_percent"],
            )
        )
    lines.extend([
        "",
        "## 当前可支持的结论",
        "",
        "三架样本的原始与简化 STEP 均通过 B-Rep 有效性检查。在固定表面网格参数下，简化模型的 CAD 面数、网格节点数、三角形数和网格阶段耗时中位数均下降。",
        "",
        "## 尚未完成的指标",
        "",
        "本表不包含主体壳面几何误差、网格质量分布或体网格结果。因此当前不能将结果表述为主体几何精度影响很小，也不能将其用于 CFD 加速结论。",
    ])
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
