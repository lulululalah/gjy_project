"""Render E2 before/after feature comparisons without marker overlays."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "data" / "plane_model" / "after_two"
SIMPLIFIED = ROOT / "data" / "plane_model" / "ending"
WORK = ROOT / "work" / "test_removal_pipeline"
OUT = ROOT / "results" / "e2_removal"
BACKGROUND = "#C8D3DA"
RIVET = "#E63946"
WINDOW = "#F4A261"
SIMPLIFIED_COLOR = "#6CA6A4"

RIVET_CASES = (
    (
        "109-ww1-standart-e-1-aircraft CATIA STP_standart-e1_wing_rivets_decals.step",
        WORK / "109-ww1-standart-e-1-aircraft_CATIA_STP_standart-e1" / "initial.pred.csv",
        (90, 0, 0),
    ),
    (
        "Gulfstream G280 v17_wing_rivets.step",
        WORK / "Gulfstream_G280_v17" / "initial.pred.csv",
        None,
        {"zoom": 2.1, "transparent": True},
    ),
)
WINDOW_CASES = (
    (
        "Gulfstream G280 v17_wing_rivets.step",
        WORK / "Gulfstream_G280_v17" / "decal.surface.pred.csv",
        (90, 0, 0),
        {"zoom": 1.35, "transparent": True, "focus_padding": 0.35},
    ),
    ("Air Plane Idea A.step", WORK / "Air_Plane_Idea_A" / "decal.pred.csv", (0, 90, 0), {"zoom": 1.0, "transparent": True}),
    ("27- DC-10 SolidWorks stp igs_DC 10_wing_rivets.step", WORK / "27-_DC-10_SolidWorks_stp_igs_DC_10" / "initial.pred.csv", (0, 0, 90), {"zoom": 1.0, "transparent": True}),
    (
        "109-ww1-standart-e-1-aircraft CATIA STP_standart-e1_wing_rivets_decals.step",
        WORK / "109-ww1-standart-e-1-aircraft_CATIA_STP_standart-e1" / "star.surface.pred.csv",
        (0, 90, 0),
        {
            "zoom": 1.0,
            "transparent": True,
            "focus_padding": 1.2,
        },
    ),
)


def load_step(path: Path):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != 1:
        raise RuntimeError(f"Cannot read STEP: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"STEP has no transferable shape: {path}")
    return shape


def read_labels(path: Path, feature: str) -> dict[int, str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        return {
            int(row["face_id"]): feature if row["pred_name"] == feature else "background"
            for row in rows
        }


def mesh_polygons(shape, labels: dict[int, str] | None, feature: str):
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    bounds = (
        box.GetXMin(), box.GetYMin(), box.GetZMin(),
        box.GetXMax(), box.GetYMax(), box.GetZMax(),
    )
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    diagonal = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5
    mesher = BRepMesh_IncrementalMesh(shape, diagonal * 0.0008, False, 0.5, True)
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError("Surface meshing did not complete.")
    accent = RIVET if feature == "rivet" else WINDOW
    polygons, colors, feature_vertices = [], [], []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    while explorer.More():
        face_id += 1
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(TopoDS.Face(explorer.Current()), location)
        if triangulation is not None:
            transform = location.Transformation()
            color = SIMPLIFIED_COLOR if labels is None else (
                accent if labels.get(face_id) == feature else BACKGROUND
            )
            for index in range(1, triangulation.NbTriangles() + 1):
                indices = triangulation.Triangle(index).Get()
                polygons.append([
                    tuple(triangulation.Node(node).Transformed(transform).Coord())
                    for node in indices
                ])
                colors.append(color)
                if labels is not None and labels.get(face_id) == feature:
                    feature_vertices.extend(polygons[-1])
        explorer.Next()
    feature_bounds = None
    if feature_vertices:
        coordinates = list(zip(*feature_vertices))
        feature_bounds = tuple(min(axis) for axis in coordinates) + tuple(
            max(axis) for axis in coordinates
        )
    return polygons, colors, bounds, feature_bounds


def draw(
    axis, polygons, colors, bounds, panel: str, camera, focus_bounds=None,
    focus_padding=0.15, transparent=False, zoom=1.0,
) -> None:
    xmin, ymin, zmin, xmax, ymax, zmax = focus_bounds or bounds
    if focus_bounds:
        span = max(xmax - xmin, ymax - ymin, zmax - zmin)
        padding = focus_padding * span
        xmin, ymin, zmin = xmin - padding, ymin - padding, zmin - padding
        xmax, ymax, zmax = xmax + padding, ymax + padding, zmax + padding
        kept = [
            (polygon, color) for polygon, color in zip(polygons, colors)
            if xmin <= sum(vertex[0] for vertex in polygon) / 3 <= xmax
            and ymin <= sum(vertex[1] for vertex in polygon) / 3 <= ymax
            and zmin <= sum(vertex[2] for vertex in polygon) / 3 <= zmax
        ]
        polygons, colors = zip(*kept) if kept else ([], [])
    base_alpha = 0.28 if focus_bounds or transparent else 1.0
    axis.add_collection3d(
        Poly3DCollection(polygons, facecolors=colors, edgecolors="none", alpha=base_alpha)
    )
    accent_polygons = [
        polygon for polygon, color in zip(polygons, colors)
        if color in {RIVET, WINDOW}
    ]
    if accent_polygons:
        accent = RIVET if RIVET in colors else WINDOW
        axis.add_collection3d(
            Poly3DCollection(
                accent_polygons, facecolors=accent, edgecolors=accent,
                linewidths=1.2, alpha=1.0,
            )
        )
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_zlim(zmin, zmax)
    axis.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin), zoom=zoom)
    axis.view_init(elev=camera[0], azim=camera[1], roll=camera[2] if len(camera) > 2 else 0)
    axis.set_axis_off()
    axis.set_title(f"({panel})", loc="left", fontsize=10, pad=2)


def render_pairs(cases, feature: str, output: Path, camera=None, local_rivet_view=False, local_feature_view=False) -> None:
    fig = plt.figure(figsize=(9.8, 2.85 * len(cases)), constrained_layout=True)
    for row, case in enumerate(cases):
        filename, prediction = case[:2]
        case_camera = camera if camera is not None else case[2]
        labels = read_labels(prediction, feature)
        original_polygons, original_colors, original_bounds, feature_bounds = mesh_polygons(
            load_step(ORIGINAL / filename), labels, feature
        )
        raw_settings = case[3] if len(case) > 3 else {}
        settings = dict(raw_settings)
        simplified_filename = settings.pop("simplified_filename", filename)
        full_view = settings.pop("full_view", False)
        simplified_polygons, simplified_colors, simplified_bounds, _ = mesh_polygons(
            load_step(SIMPLIFIED / simplified_filename), None, feature
        )
        left = fig.add_subplot(len(cases), 2, 2 * row + 1, projection="3d")
        right = fig.add_subplot(len(cases), 2, 2 * row + 2, projection="3d")
        focus = None if full_view else (
            feature_bounds if local_feature_view else (case[2] if local_rivet_view else None)
        )
        draw(
            left, original_polygons, original_colors, original_bounds,
            chr(97 + 2 * row), case_camera, focus, **settings,
        )
        draw(
            right, simplified_polygons, simplified_colors, simplified_bounds,
            chr(98 + 2 * row), case_camera, focus, **settings,
        )
    accent = RIVET if feature == "rivet" else WINDOW
    label = "Rivet face" if feature == "rivet" else "Surface-feature face"
    fig.legend(
        handles=[
            Patch(facecolor=BACKGROUND, label="Retained surface"),
            Patch(facecolor=accent, label=label),
            Patch(facecolor=SIMPLIFIED_COLOR, label="Simplified model"),
        ],
        loc="lower center", ncol=3, frameon=False, fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")


def main() -> None:
    render_pairs(
        RIVET_CASES, "rivet", OUT / "e2_rivet_before_after.png",
        camera=(0, -90), local_feature_view=True,
    )
    render_pairs(
        WINDOW_CASES, "surface_feature", OUT / "e2_window_before_after.png",
        local_feature_view=True,
    )


if __name__ == "__main__":
    main()
