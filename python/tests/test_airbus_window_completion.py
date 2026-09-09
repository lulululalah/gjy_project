import sys
import unittest
from pathlib import Path


PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from surface_visibility_guard import find_compact_airbus_window_components


class AirbusWindowCompletionTests(unittest.TestCase):
    def test_completes_compact_sixteen_face_background_component(self):
        face_metrics = {
            face_id: {
                "area": 0.02,
                "edge_count": 7,
                "center": (face_id * 0.01, 2.0, 3.7),
            }
            for face_id in range(1, 17)
        }
        neighbors = {
            face_id: {
                1 + (face_id % 16),
                16 if face_id == 1 else face_id - 1,
            }
            for face_id in range(1, 17)
        }
        predictions = {face_id: 0 for face_id in face_metrics}

        groups = find_compact_airbus_window_components(
            face_metrics, neighbors, predictions
        )

        self.assertEqual(groups, [list(range(1, 17))])

    def test_rejects_noncompact_or_partly_predicted_components(self):
        face_metrics = {
            face_id: {
                "area": 0.02,
                "edge_count": 7,
                "center": (face_id * 0.1, 2.0, 3.7),
            }
            for face_id in range(1, 17)
        }
        neighbors = {
            face_id: {
                1 + (face_id % 16),
                16 if face_id == 1 else face_id - 1,
            }
            for face_id in range(1, 17)
        }
        predictions = {face_id: 0 for face_id in face_metrics}
        self.assertEqual(
            find_compact_airbus_window_components(
                face_metrics, neighbors, predictions
            ),
            [],
        )

        face_metrics[16]["center"] = (0.16, 2.0, 3.7)
        predictions[16] = 2
        self.assertEqual(
            find_compact_airbus_window_components(
                face_metrics, neighbors, predictions
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
