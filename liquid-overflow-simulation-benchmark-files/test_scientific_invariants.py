"""Scientific correctness checks, independent of expected benchmark winners."""
import unittest
import numpy as np
from overflow_study_v2 import (C,generate,first_crossing,features,split_cycles,
                                training_frames,sweep,select_threshold)

class ScientificInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.ds=generate(8821,120)

    def test_events_are_first_rim_crossings(self):
        ds=self.ds
        np.testing.assert_array_equal(ds.event,first_crossing(ds.height,ds.freeboard))
        for h,rim,e in zip(ds.height,ds.freeboard,ds.event):
            if e>=0:
                self.assertTrue(np.all(h[:e]<rim));self.assertGreaterEqual(h[e],rim)
            else:self.assertTrue(np.all(h<rim))

    def test_sensor_failure_does_not_change_latent_ground_truth(self):
        normal=generate(823,30);failed=generate(823,30,'tof_freeze_90s')
        np.testing.assert_array_equal(normal.event,failed.event)
        np.testing.assert_array_equal(normal.height,failed.height)
        self.assertFalse(np.array_equal(normal.risk,failed.risk))

    def test_cycle_splits_are_disjoint_and_exhaustive(self):
        parts=split_cycles(self.ds.event,22)
        np.testing.assert_array_equal(np.sort(np.concatenate(parts)),np.arange(len(self.ds.event)))
        self.assertEqual(len(set(np.concatenate(parts))),len(self.ds.event))

    def test_temporal_features_are_causal(self):
        a=self.ds.risk.copy();b=a.copy();b[:,151:]=0
        np.testing.assert_array_equal(features(a)[:,:151],features(b)[:,:151])

    def test_persistence_warning_uses_third_sample(self):
        scores=np.zeros((2,360));scores[0,5:8]=90;scores[1,5:7]=90
        result=sweep(scores,np.array([20,-1]),np.array([80]))
        np.testing.assert_array_equal(result['warning_times'][:,0],[7,-1])
        self.assertEqual(result['mean_lead_s'][0],13)

    def test_premature_warnings_are_not_timely_detections(self):
        scores=np.full((2,360),90.)
        result=sweep(scores,np.array([100,-1]),np.array([80]))
        self.assertEqual(result['detection_rate'][0],0)
        self.assertEqual(result['early_warning_rate'][0],1)
        self.assertEqual(result['false_alarm_rate'][0],1)
        self.assertEqual(result['precision'][0],0)

    def test_training_excludes_event_and_post_event_frames(self):
        ds=self.ds;x,y=training_frames(ds)
        times=np.arange(C.duration_s)[::C.train_frame_stride][None,:]
        expected=((ds.event[:,None]<0)|(times<ds.event[:,None])).sum()
        self.assertEqual(len(y),expected)
        gap=ds.event[:,None]-times
        self.assertEqual(int(y.sum()),int(((ds.event[:,None]>=0)&(gap>0)&(gap<=60)).sum()))

    def test_disabled_warning_threshold_is_valid_fallback(self):
        r=sweep(np.ones((2,360))*100,np.array([100,-1]))
        j,feasible=select_threshold(r)
        self.assertFalse(feasible);self.assertGreater(r['threshold'][j],100)

    def test_threshold_search_has_no_integer_score_scale_advantage(self):
        rng=np.random.default_rng(16);scores=rng.uniform(0,100,(10,360))
        events=np.array([200]*5+[-1]*5)
        a=sweep(scores,events);b=sweep(scores*1.7+11.,events)
        ia,oka=select_threshold(a);ib,okb=select_threshold(b)
        self.assertEqual(oka,okb)
        for metric in ('detection_rate','false_alarm_rate','early_warning_rate'):
            self.assertEqual(a[metric][ia],b[metric][ib])

    def test_generator_is_deterministic(self):
        a=generate(88,8);b=generate(88,8)
        for key in a.__dataclass_fields__:np.testing.assert_array_equal(getattr(a,key),getattr(b,key))

if __name__=='__main__':unittest.main(verbosity=2)
