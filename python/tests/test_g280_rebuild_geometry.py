"""Local integration regression for G280 window-host trimming."""

import os
import subprocess
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

DETECTOR = ROOT / 'build/Release/Detector.exe'
STAGE = ROOT / 'work/test_removal_pipeline/Gulfstream_G280_v17'
INPUT = STAGE / 'Gulfstream_G280_v17_removed.step'
PREDICTIONS = STAGE / 'surface.pred.csv'


@unittest.skipUnless(find_spec('OCC') is not None and all(p.exists() for p in (DETECTOR, INPUT, PREDICTIONS)),
                     'Requires local OCC runtime, Detector and archived G280 stage inputs')
class G280WindowHostGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = ROOT / 'work/figure11_g280_repair_20261015/regression'
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / 'repaired.step'
        environment = os.environ.copy()
        windows = Path(os.environ['SYSTEMROOT'])
        environment['PATH'] = os.pathsep.join((str(DETECTOR.parent), str(windows / 'System32'), str(windows)))
        result = subprocess.run([str(DETECTOR), '--rebuild-embedded-window-hosts', str(INPUT),
                                 str(PREDICTIONS), str(output), 'gulfstream-g280'],
                                cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)
        # The native build uses its bundled OCC DLLs; pythonOCC is a separate
        # runtime used only to validate the resulting STEP.
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
        from OCC.Core.gp import gp_Dir, gp_Lin, gp_Pnt
        from mesh_step_surface import load_step
        cls.checker = BRepCheck_Analyzer
        cls.intersector_type = IntCurvesFace_ShapeIntersector
        cls.ray = staticmethod(lambda x, y: gp_Lin(gp_Pnt(x, y, 200), gp_Dir(0, 0, -1)))
        cls.shape = load_step(output)

    def test_rebuilt_shape_remains_brep_valid(self):
        self.assertTrue(self.checker(self.shape).IsValid())

    def test_window_band_has_no_coincident_front_skins(self):
        intersector = self.intersector_type()
        intersector.Load(self.shape, 1e-8)
        checked = 0
        for x in (-25.297740449, -18.682641558, -10.413767943, -2.144894329, 2.816429839):
            for y in (3.871889511, 7.459990026, 9.852057035):
                intersector.Perform(self.ray(x, y), 0, 400)
                depths = sorted(intersector.WParameter(i) for i in range(1, intersector.NbPnt() + 1))
                self.assertGreaterEqual(len(depths), 2, (x, y))
                self.assertGreater(depths[1] - depths[0], 1e-6, (x, y, depths[:2]))
                checked += 1
        self.assertEqual(checked, 15)


if __name__ == '__main__':
    unittest.main()
