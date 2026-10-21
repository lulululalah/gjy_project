import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cad.candidates import transfer_surface_candidates, complete_geometric_candidates


class TransferTests(unittest.TestCase):
    def geometry(self,ids):
        return pd.DataFrame(dict(id=ids,area=[2.,3.],perimeter=[6.,8.],surfaceType=[6,6],
            centerX=[1.,10.],centerY=[0.,0.],centerZ=[0.,0.]))

    def predictions(self,ids,labels):
        return pd.DataFrame(dict(face_id=ids,pred_label=labels,pred_name=['background']*2,
            prob_surface_feature=[.1,.2]))

    def test_face_numbers_can_change_and_truth_cannot_affect_correspondence(self):
        original=self.geometry([1,2]);current=self.geometry([9,1])
        first=self.predictions([1,2],[2,0]);second=self.predictions([9,1],[0,0])
        result,report=transfer_surface_candidates(original,first,current,second)
        self.assertEqual(result.pred_label.tolist(),[2,0])
        self.assertEqual(result.cad_source_face_ids.tolist(),['1',''])
        self.assertEqual(report['added_candidates'],1)
        altered,_=transfer_surface_candidates(original.assign(label=0),first.assign(true_label=0),
            current.assign(label=2),second.assign(true_label=2))
        self.assertEqual(altered.pred_label.tolist(),result.pred_label.tolist())
        self.assertEqual(second.pred_label.tolist(),[0,0])

    def test_changed_geometry_is_not_matched_even_with_same_face_number(self):
        original=self.geometry([1,2]);current=self.geometry([1,2]);current.loc[0,'area']=20.
        result,_=transfer_surface_candidates(original,self.predictions([1,2],[2,0]),current,self.predictions([1,2],[0,2]))
        self.assertEqual(result.pred_label.tolist(),[0,2])

    def test_current_predictions_must_align_with_exported_geometry(self):
        with self.assertRaisesRegex(ValueError,'not aligned'):
            transfer_surface_candidates(self.geometry([1,2]),self.predictions([1,2],[2,0]),
                self.geometry([9,1]),self.predictions([1,9],[0,0]))

    def test_large_perforated_host_is_preserved_despite_both_prediction_passes(self):
        original=self.geometry([1,2]);current=self.geometry([9,1])
        current['innerWireCount']=[109,0];current['relativeArea']=[.17,.0001]
        result,report=transfer_surface_candidates(original,self.predictions([1,2],[2,2]),
            current,self.predictions([9,1],[2,2]))
        self.assertEqual(result.pred_label.tolist(),[0,2])
        self.assertEqual(report['protected_host_faces'],[9])

    def test_repeated_panes_need_a_group_and_neural_attachment_evidence(self):
        features=pd.DataFrame(dict(id=range(1,10),area=[1.]*9,neighborAreaMax=[100.]*9,
            relativeArea=[.0001]*9,surfaceType=[6]*9,numEdges=[2]*9,smoothEdgeCount=[1]*9,
            innerWireCount=[0]*9,numWires=[1]*9,neighbors=['20']*9,normalNeighborDotMin=[.8]*9))
        predictions=pd.DataFrame(dict(face_id=range(1,10),pred_label=[0]*9,pred_name=['background']*9,
            prob_surface_feature=[.2]*8+[.01]))
        completed,report=complete_geometric_candidates(features,predictions)
        self.assertEqual(completed.pred_label.tolist(),[2]*8+[0])
        self.assertEqual(report['repeated_panes'],8)
        missing=predictions.copy();missing.loc[0,'prob_surface_feature']=.01
        incomplete,_=complete_geometric_candidates(features,missing)
        self.assertEqual(incomplete.pred_label.tolist(),[0]*9)
        changed,_=complete_geometric_candidates(features.assign(label=2),predictions.assign(true_label=2))
        self.assertEqual(changed.pred_label.tolist(),completed.pred_label.tolist())

    def test_decorative_patch_requires_one_small_smooth_host_attachment(self):
        features=pd.DataFrame(dict(id=[1,2],area=[1.,1.],neighborAreaMax=[100.,100.],
            relativeArea=[.0001]*2,surfaceType=[6]*2,numEdges=[10]*2,smoothEdgeCount=[1]*2,
            innerWireCount=[0]*2,numWires=[1]*2,neighbors=['20','20 21'],normalNeighborDotMin=[.999]*2))
        predictions=self.predictions([1,2],[0,0]);predictions['prob_surface_feature']=.2
        completed,_=complete_geometric_candidates(features,predictions)
        self.assertEqual(completed.pred_label.tolist(),[2,0])

if __name__=='__main__':
    unittest.main()
