import unittest
from pathlib import Path
from unittest.mock import patch

from run_test_removal_pipeline import log_contains, verified_route


class RemovalPipelineRoutingTests(unittest.TestCase):
    def test_airbus_uses_embedded_route_without_changing_other_routes(self):
        self.assertEqual(verified_route("Airbus"), "embedded")
        self.assertEqual(verified_route("Airplane_body"), "embedded")
        self.assertEqual(verified_route("Air_Plane_Idea_A"), "invalid-host")
        self.assertEqual(verified_route("27-_DC-10"), "split")
        self.assertEqual(verified_route("87-747-400"), "generic")

    def test_airbus_geometry_log_requires_every_regression_marker(self):
        log_path = Path("embedded.log")
        with patch.object(Path, "read_text", return_value="alpha\nbeta\ngamma\n"):
            self.assertTrue(log_contains(log_path, ("alpha", "gamma")))
            self.assertFalse(log_contains(log_path, ("alpha", "missing")))


if __name__ == "__main__":
    unittest.main()
