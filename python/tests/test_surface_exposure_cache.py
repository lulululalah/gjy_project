import sys
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from recognition.postprocess import compute_surface_exposure_scores


class ExposureCacheTests(unittest.TestCase):
    def test_reuse_only_missing_faces_and_invalidate_changed_geometry(self):
        work = ROOT / "work"
        work.mkdir(exist_ok=True)
        directory = work / ("exposure_cache_test_" + uuid.uuid4().hex)
        directory.mkdir()
        try:
            self.assertTrue(directory.resolve().is_relative_to(work.resolve()))
            source = directory / "model.step"
            cache = directory / "cache"
            source.write_bytes(b"geometry-one")
            with patch("recognition.postprocess._compute_surface_exposure_scores",
                       side_effect=lambda _, ids: {int(i): 0.5 for i in ids}) as compute:
                self.assertEqual(compute_surface_exposure_scores(source, [1], cache), {1: 0.5})
                self.assertEqual(compute_surface_exposure_scores(source, [1, 2], cache), {1: 0.5, 2: 0.5})
                self.assertEqual(compute.call_args.args[1], [2])
                compute_surface_exposure_scores(source, [2, 1], cache)
                self.assertEqual(compute.call_count, 2)
                source.write_bytes(b"geometry-two")
                compute_surface_exposure_scores(source, [1], cache)
                self.assertEqual(compute.call_count, 3)
        finally:
            if not directory.resolve().is_relative_to(work.resolve()):
                raise ValueError("Test cleanup target outside work directory")
            shutil.rmtree(directory)

    def test_no_cache_preserves_the_original_scoring_path(self):
        with patch("recognition.postprocess._compute_surface_exposure_scores",
                   return_value={3: 1.0}) as compute:
            self.assertEqual(compute_surface_exposure_scores(Path("unused.step"), [3]), {3: 1.0})
            compute.assert_called_once_with(Path("unused.step"), [3])


if __name__ == "__main__":
    unittest.main()
