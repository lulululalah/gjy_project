import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import visualize_cad_targets as viewer


class ViewerInputTests(unittest.TestCase):
    def test_unlabelled_comparison_endpoint_uses_display_background_only(self):
        with patch.object(viewer, "load_labels") as labels:
            self.assertEqual(viewer.comparison_display_labels({1: None, 3: None}, None),
                             {1: "background", 3: "background"})
        labels.assert_not_called()

    def test_extra_mesh_detail_frames_the_complete_source_group(self):
        case = {"projection": (0, 0, 1), "up": (0, 1, 0), "extra_details": [
            {"name": "tail_decal", "semantic": "decal", "original_face_ids": [1, 2]}]}
        with patch.object(viewer, "bounds_of_faces", return_value=(0, 0, 0, 12, 6, 1)) as bounds:
            regions = viewer.select_extra_mesh_details({1: "first", 2: "second"},
                                                       {1: "decal", 2: "decal"}, case, (2400, 800))
        bounds.assert_called_once_with({1: "first", 2: "second"})
        self.assertEqual(regions[0]["name"], "tail_decal")
        self.assertEqual(regions[0]["center"], (6, 3, 0.5))
        self.assertAlmostEqual(regions[0]["scale"], 6.48)
        self.assertEqual(regions[0]["original_face_ids"], [1, 2])

    def test_extra_mesh_detail_rejects_missing_or_wrong_semantic_faces(self):
        case = {"extra_details": [{"name": "tail_decal", "semantic": "decal",
                                    "original_face_ids": [1, 2]}]}
        for labels in ({1: "decal"}, {1: "decal", 2: "window"}):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "Invalid source faces"):
                viewer.select_extra_mesh_details({1: None, 2: None}, labels, case, (2400, 800))

    def test_figure11_body_style_is_opt_in_and_opaque(self):
        for style in (None, viewer.FIGURE11_STYLE):
            with self.subTest(style=style):
                display = MagicMock()
                ais = display.DisplayShape.return_value[0]
                viewer.show_faces(display, {1: None}, {1: "background"}, style=style)
                kwargs = display.DisplayShape.call_args.kwargs
                self.assertEqual(kwargs["transparency"], 0.0)
                if style:
                    self.assertAlmostEqual(kwargs["color"].Red(), style["body_rgb"][0])
                    ais.SetMaterial.assert_called_once()
                else:
                    self.assertIs(kwargs["color"], viewer.BODY)
                    ais.SetMaterial.assert_not_called()

    def test_mesh_detail_uses_the_observed_side_and_shared_world_position(self):
        boxes = {
            1: (1, 0, 0, 2, 1, 1),
            2: (1, 0, 4, 2, 1, 5),
            3: (-2, 0, 4, -1, 1, 5),
        }
        with patch.object(viewer, "bounds_of_faces", side_effect=lambda faces: boxes[next(iter(faces))]):
            region = viewer.select_mesh_detail({1: None, 2: None, 3: None},
                {1: "window", 2: "window", 3: "window"}, (-2, -1, -10, 2, 1, 10),
                {"name": "fixture", "projection": (1, 0, 0), "up": (0, 1, 0),
                 "length_axis": 2, "detail_fraction": 0.2, "detail_semantic": "window"},
                (2400, 800))
        self.assertEqual(region["original_face_id"], 2)
        self.assertEqual(region["center"], (1.5, 0.5, 4.5))
        self.assertAlmostEqual(region["scale"], 4 / 3)

    def test_mesh_detail_without_requested_feature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No window detail"):
            viewer.select_mesh_detail({1: None}, {1: "background"}, (0, 0, 0, 1, 1, 1),
                {"name": "fixture", "projection": (1, 0, 0), "up": (0, 1, 0),
                 "length_axis": 2, "detail_fraction": 0.2, "detail_semantic": "window"},
                (2400, 800))

    def test_figure11_export_routes_without_opening_interactive_gui(self):
        with patch.object(sys, "argv", ["viewer", "--export-figure11", "output",
                                        "--figure11-model", "airplane_body"]), \
             patch.object(viewer, "export_figure11") as export, \
             patch.object(viewer, "init_display") as init_display:
            viewer.main()
        export.assert_called_once_with(Path("output"), "airplane_body")
        init_display.assert_not_called()

    def test_projected_camera_frames_aircraft_length_on_z_axis(self):
        camera = viewer.paired_camera((-2, -1, -5, 2, 1, 5),
                                      (1, 0, 0), (0, 1, 0), (2400, 1000))
        self.assertEqual(camera["center"], (0, 0, 0))
        self.assertAlmostEqual(camera["scale"], 10 * 1000 / 2400 * 1.08)

    def test_oblique_camera_scale_covers_projected_bounds(self):
        camera = viewer.paired_camera((-2, -1, -5, 2, 1, 5),
                                      (1, 0, 1), (0, 1, 0), (1000, 1000))
        self.assertAlmostEqual(camera["scale"], 14 / (2 ** 0.5) * 1.08)

    def test_degenerate_camera_vectors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonzero"):
            viewer.paired_camera((0, 0, 0, 1, 1, 1), (0, 0, 0), (0, 1, 0), (100, 100))

    def test_paper_comparison_export_does_not_open_interactive_gui(self):
        with patch.object(sys, "argv", ["viewer", "--export-paper-comparisons", "output",
                                        "--comparison-model", "airplane_body"]), \
             patch.object(viewer, "export_paper_comparisons") as export, \
             patch.object(viewer, "init_display") as init_display:
            viewer.main()
        export.assert_called_once_with(Path("output"), "airplane_body")
        init_display.assert_not_called()

    def test_comparison_selection_without_export_is_rejected(self):
        with patch.object(sys, "argv", ["viewer", "--comparison-model", "dc10"]):
            with self.assertRaises(SystemExit):
                viewer.main()

    def read_fixture(self, faces):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(json.dumps({"faces": faces}), encoding="utf-8")
            return viewer.load_labels(path)

    def test_valid_labels(self):
        self.assertEqual(
            self.read_fixture([{"face_id": 1, "semantic": "window"}]),
            {1: "window"},
        )

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.read_fixture([
                {"face_id": 1, "semantic": "window"},
                {"face_id": 1, "semantic": "decal"},
            ])

    def test_unknown_semantics_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.read_fixture([{"face_id": 1, "semantic": "unknown"}])

    def test_equal_counts_with_different_ids_are_rejected_before_gui(self):
        with patch.object(sys, "argv", ["viewer"]), \
             patch.object(viewer, "load_faces", return_value={1: None, 2: None}), \
             patch.object(viewer, "load_labels", return_value={2: "window", 3: "decal"}), \
             patch.object(viewer, "init_display") as init_display:
            with self.assertRaisesRegex(RuntimeError, "Face mismatch"):
                viewer.main()
            init_display.assert_not_called()

    def test_top_startup_view_configures_camera(self):
        display = MagicMock()
        start_display = MagicMock()
        add_menu = MagicMock()
        add_function = MagicMock()
        with patch.object(sys, "argv", ["viewer", "--view", "top"]), \
             patch.object(viewer, "load_faces", return_value={}), \
             patch.object(viewer, "load_labels", return_value={}), \
             patch.object(viewer, "init_display", return_value=(
                 display, start_display, add_menu, add_function,
             )):
            viewer.main()
        display.View.SetProj.assert_called_once_with(0.0, 1.0, 0.0)
        display.View.SetUp.assert_called_once_with(0.0, 0.0, -1.0)
        display.FitAll.assert_called_once()
        start_display.assert_called_once()
        self.assertIn(
            "Top_view", [call.args[1].__name__ for call in add_function.call_args_list]
        )

    def test_detail_center_changes_camera_only(self):
        display = MagicMock()
        with patch.object(sys, "argv", [
                 "viewer", "--view", "top", "--center", "34.1", "0.8", "4.0",
                 "--detail-scale", "2.0",
             ]), \
             patch.object(viewer, "load_faces", return_value={}), \
             patch.object(viewer, "load_labels", return_value={}), \
             patch.object(viewer, "init_display", return_value=(
                 display, MagicMock(), MagicMock(), MagicMock(),
             )):
            viewer.main()
        display.View.SetAt.assert_called_once_with(34.1, 0.8, 4.0)
        display.View.Camera().SetScale.assert_called_once_with(2.0)

    def test_fixed_camera_uses_explicit_center_and_scale_without_fitting(self):
        display = MagicMock()
        viewer.apply_camera(display, "top", (10.0, 2.0, 3.0), 40.0)
        display.View.SetProj.assert_called_once_with(0.0, 1.0, 0.0)
        display.View.SetUp.assert_called_once_with(0.0, 0.0, -1.0)
        display.View.SetAt.assert_called_once_with(10.0, 2.0, 3.0)
        display.View.Camera().SetScale.assert_called_once_with(40.0)
        display.FitAll.assert_not_called()
        view_calls = [call[0] for call in display.View.mock_calls]
        self.assertLess(view_calls.index("SetAt"), view_calls.index("SetProj"))

    def test_export_mode_does_not_open_interactive_window(self):
        with patch.object(sys, "argv", ["viewer", "--export-figure6", "output"]), \
             patch.object(viewer, "export_figure6") as export, \
             patch.object(viewer, "init_display") as init_display:
            viewer.main()
        export.assert_called_once_with(Path("output"))
        init_display.assert_not_called()

    def test_detail_regions_select_three_distinct_positions_not_three_faces_of_one_rivet(self):
        boxes = [
            (1, (0.0, 1.0, 0.0, 0.05, 1.05, 0.05)),
            (2, (0.0, 0.98, 0.0, 0.05, 1.0, 0.05)),
            (3, (8.0, 0.0, 0.0, 8.05, 0.05, 0.05)),
            (4, (4.0, 0.0, 8.0, 4.05, 0.05, 8.05)),
        ]
        regions = viewer.select_detail_regions(boxes)
        self.assertEqual({region["original_face_id"] for region in regions}, {1, 3, 4})
        self.assertEqual([region["number"] for region in regions], [1, 2, 3])
        self.assertEqual(len({region["scale"] for region in regions}), 1)

    def test_too_few_distinct_rivet_positions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct rivet locations"):
            viewer.select_detail_regions([(1, (0, 0, 0, 0.05, 0.05, 0.05))])

    def test_horizontal_export_does_not_open_interactive_window(self):
        with patch.object(sys, "argv", ["viewer", "--export-figure6-horizontal", "output"]), \
             patch.object(viewer, "export_figure6") as export, \
             patch.object(viewer, "init_display") as init_display:
            viewer.main()
        export.assert_called_once_with(Path("output"), horizontal=True)
        init_display.assert_not_called()

    def test_surface_regions_exclude_cockpit_and_opposite_side(self):
        regions = viewer.select_surface_detail_regions([
            (1, "window", (-30, 0, 5, -29, 2, 6)),
            (2, "window", (-20, 0, 5, -19, 2, 6)),
            (3, "window", (-10, 0, 5, -9, 2, 6)),
            (4, "window", (-65, 0, 5, -55, 6, 6)),
            (5, "window", (-20, 0, -6, -19, 2, -5)),
            (6, "decal", (60, 20, 0.1, 65, 25, 0.2)),
            (7, "decal", (65, 20, 0.1, 70, 25, 0.2)),
            (8, "decal", (60, 20, -0.2, 65, 25, -0.1)),
        ])
        self.assertEqual(regions[0]["original_face_ids"], [2])
        self.assertEqual(regions[1]["original_face_ids"], [6, 7])
        for actual, expected in zip(regions[1]["center"], (65, 22.5, 0.15)):
            self.assertAlmostEqual(actual, expected)

    def test_surface_regions_require_both_feature_types(self):
        with self.assertRaisesRegex(ValueError, "passenger window and decal"):
            viewer.select_surface_detail_regions([])

    def test_figure7_export_does_not_open_interactive_window(self):
        with patch.object(sys, "argv", ["viewer", "--export-figure7", "output"]), \
             patch.object(viewer, "export_figure7") as export, \
             patch.object(viewer, "init_display") as init_display:
            viewer.main()
        export.assert_called_once_with(Path("output"))
        init_display.assert_not_called()

    def test_simultaneous_figure_exports_are_rejected(self):
        with patch.object(sys, "argv", ["viewer", "--export-figure6", "one", "--export-figure7", "two"]), \
             patch.object(viewer, "export_figure6") as export6, \
             patch.object(viewer, "export_figure7") as export7, \
             self.assertRaises(SystemExit):
            viewer.main()
        export6.assert_not_called()
        export7.assert_not_called()

    def test_figure8_export_does_not_open_interactive_window(self):
        with patch.object(sys, "argv", ["viewer", "--export-figure8", "output"]), \
             patch.object(viewer, "export_figure8") as export, \
             patch.object(viewer, "init_display") as init_display:
            viewer.main()
        export.assert_called_once_with(Path("output"))
        init_display.assert_not_called()

    def test_figure8_uses_three_visually_reviewed_positions(self):
        with patch.object(viewer, "export_paired_rivet_views") as export:
            viewer.export_figure8(Path("output"))
        self.assertEqual(export.call_args.args[2], "fig8_xian_y20")
        self.assertEqual(export.call_args.kwargs["detail_face_ids"], {48, 59, 83})
        self.assertEqual(export.call_args.kwargs["detail_scale_factor"], 0.25)

    def test_figure7_keeps_its_existing_candidate_filter(self):
        with patch.object(viewer, "export_paired_rivet_views") as export:
            viewer.export_figure7(Path("output"))
        self.assertEqual(export.call_args.args[2], "fig7_airbus")
        candidate_filter = export.call_args.args[3]
        self.assertTrue(candidate_filter((0, -0.4, 0, 0.01, -0.3, 0.01)))
        self.assertFalse(candidate_filter((0, -1.8, 0, 0.01, -1.7, 0.01)))
        self.assertEqual(export.call_args.kwargs, {})

    def test_figure8_cannot_be_combined_with_other_exports(self):
        for other in ("--export-figure6", "--export-figure7"):
            with self.subTest(other=other), \
                 patch.object(sys, "argv", ["viewer", other, "one", "--export-figure8", "two"]), \
                 patch.object(viewer, "export_figure8") as export, \
                 self.assertRaises(SystemExit):
                viewer.main()
            export.assert_not_called()


if __name__ == "__main__":
    unittest.main()
