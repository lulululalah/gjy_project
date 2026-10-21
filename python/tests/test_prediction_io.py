import csv
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from visualization.predictions import read_predictions_csv, write_predictions_csv


class NonClosingStringIO(io.StringIO):
    def close(self):
        pass




class PredictionCsvTests(unittest.TestCase):
    def test_real_face_ids_and_probabilities_are_written(self):
        output_buffer = NonClosingStringIO()
        with (
            patch.object(Path, "mkdir"),
            patch.object(Path, "open", return_value=output_buffer),
        ):
            write_predictions_csv(
                Path("predictions.csv"),
                face_ids=[7, 19],
                pred_labels=[1, 2],
                probabilities=np.array([[0.1, 0.8, 0.1], [0.2, 0.3, 0.5]]),
                label_names=["background", "rivet", "surface_feature"],
            )
        output_buffer.seek(0)
        rows = list(csv.DictReader(output_buffer))

        self.assertEqual([row["face_id"] for row in rows], ["7", "19"])
        self.assertEqual(rows[0]["pred_name"], "rivet")
        self.assertEqual(float(rows[1]["pred_confidence"]), 0.5)
        self.assertEqual(float(rows[1]["prob_surface_feature"]), 0.5)

    def test_duplicate_face_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            write_predictions_csv(
                Path("predictions.csv"),
                face_ids=[3, 3],
                pred_labels=[0, 1],
            )

    def test_prediction_csv_reload_preserves_face_ids(self):
        input_buffer = io.StringIO("face_id,pred_label\n19,2\n7,1\n")
        with patch.object(Path, "open", return_value=input_buffer):
            face_ids, pred_labels = read_predictions_csv(Path("predictions.csv"))
        self.assertEqual(face_ids, [19, 7])
        self.assertEqual(pred_labels, [2, 1])

    def test_prediction_csv_reload_rejects_duplicate_face_ids(self):
        input_buffer = io.StringIO("face_id,pred_label\n7,1\n7,2\n")
        with patch.object(Path, "open", return_value=input_buffer):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_predictions_csv(Path("predictions.csv"))


if __name__ == "__main__":
    unittest.main()
