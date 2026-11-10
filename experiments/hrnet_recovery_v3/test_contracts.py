import unittest
from contracts import study_tip_supervision, exact_offset_target, encode_offsets

class Contracts(unittest.TestCase):
    def row(self,status='provisional_engineering_review_v3'):
        return dict(catheter_instance_id=1,tip_state='Assessable distal tip',review_status=status,x=1.2,y=2.3)
    def test_unknown_is_not_negative(self):
        self.assertFalse(study_tip_supervision([self.row('weak_heuristic_unreviewed')],[1])['enabled'])
        self.assertFalse(study_tip_supervision([self.row()],[1,2])['enabled'])
    def test_valid_and_negative(self):
        self.assertTrue(study_tip_supervision([self.row()],[1])['enabled'])
        self.assertTrue(study_tip_supervision([],[],True)['enabled'])
        with self.assertRaises(ValueError):study_tip_supervision([self.row()],[1],True)
    def test_exact_offsets_and_collision(self):
        for x,y in [(0.1,0.2),(5.49,9.51),(191.2,191.4)]:
            gx,gy,dx,dy=exact_offset_target(x,y,192,192)
            self.assertAlmostEqual(gx+dx,x);self.assertAlmostEqual(gy+dy,y)
        off,mask=encode_offsets([(5.1,5.2),(5.3,5.4)],192,192)
        self.assertEqual(mask.sum(),0)
if __name__=='__main__':unittest.main()
