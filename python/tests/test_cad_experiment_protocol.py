import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cad.candidates import stage_stem, verified_route, reconstruction_profile, intermediate_step_name


class RoutingTests(unittest.TestCase):
    def test_geometry_families(self):
        for name,route in [('Airbus','embedded'),('Airplane_body','embedded'),
            ('AULIRA_2','embedded'),('Gulfstream_G280_v17','embedded'),
            ('Air_Plane_Idea_A','invalid-host'),('27-_DC-10','split'),('87-747-400','generic')]:
            with self.subTest(name=name):
                self.assertEqual(verified_route(name),route)

    def test_real_source_names_keep_stable_case_identifiers(self):
        self.assertEqual(stage_stem('27- DC-10 SolidWorks stp igs_DC 10_wing_rivets.step'),
            '27-_DC-10_SolidWorks_stp_igs_DC_10')
        self.assertEqual(stage_stem('Gulfstream G280 v17_wing_rivets.step'),'Gulfstream_G280_v17')

    def test_all_embedded_cases_have_explicit_profiles_and_named_intermediates(self):
        for stem,profile in [('Gulfstream_G280_v17','gulfstream-g280'),('Airbus','airbus'),
            ('Airplane_body','airplane-body'),('AULIRA_2','aulira')]:
            self.assertEqual(reconstruction_profile(stem),profile)
            self.assertTrue(intermediate_step_name(stem).startswith(stem+'_'))
        with self.assertRaises(ValueError):
            intermediate_step_name('../other')


if __name__=='__main__':
    unittest.main()
