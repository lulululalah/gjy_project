import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cad.windows import supported_window_location

class WindowLocationTests(unittest.TestCase):
    def test_rebuilt_patch_can_be_supported_by_smaller_original_pane(self):
        self.assertTrue(supported_window_location(20,(1,2,3),[(8,(1,3,3))]))

    def test_remote_face_or_large_carrier_cannot_supply_support(self):
        self.assertFalse(supported_window_location(20,(1,2,3),[(8,(20,3,3))]))
        self.assertFalse(supported_window_location(20,(1,2,3),[(1000,(1,2,3))]))
        self.assertFalse(supported_window_location(20,(1,2,3),[]))

if __name__=='__main__':
    unittest.main()
