"""Checks of the extension's scientific interpretation, not its expected winners."""
import unittest
import numpy as np
import overflow_study_v2 as base
from sensitivity_study import sensitivity_cycle, generate_sensitivity


class SensitivityInvariants(unittest.TestCase):
    def test_nominal_extension_exactly_matches_base_generator(self):
        for seed in (12, 197, 881):
            for a, b in zip(base.make_cycle(seed), sensitivity_cycle(seed)):
                np.testing.assert_array_equal(a,b)

    def test_every_alternative_uses_first_rim_crossing(self):
        for condition in ('heating_power_0.8','heating_power_1.2','growth_power_1.5','growth_power_2.5'):
            ds=generate_sensitivity(75000,25,condition)
            np.testing.assert_array_equal(ds.event,base.first_crossing(ds.height,ds.freeboard))

    def test_geometry_error_changes_only_height_feature(self):
        a=generate_sensitivity(75000,25,'fresh_nominal')
        b=generate_sensitivity(75000,25,'freeboard_error_20pct')
        for key in ('raw','height','freeboard','event','cycle_seed'):
            np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        np.testing.assert_array_equal(a.risk[:,:,[0,2,3]],b.risk[:,:,[0,2,3]])
        self.assertFalse(np.array_equal(a.risk[:,:,1],b.risk[:,:,1]))

    def test_growth_changes_do_not_modify_thermal_or_humidity_measurements(self):
        a=generate_sensitivity(75000,25,'fresh_nominal')
        b=generate_sensitivity(75000,25,'growth_power_1.5')
        np.testing.assert_array_equal(a.raw[:,:,2:],b.raw[:,:,2:])
        np.testing.assert_array_equal(a.cycle_seed,b.cycle_seed)
        self.assertFalse(np.array_equal(a.height,b.height))

    def test_alternative_heating_equation_at_one_time(self):
        seed=99
        risk,raw,height,rim,event=sensitivity_cycle(seed,heating_power=1.2)
        rng=np.random.default_rng(seed)
        tau=rng.uniform(65.,100.)
        # Underlying thermal curve remains monotone and begins at 25 C;
        # observed noise means the raw proxy is not itself exactly monotone.
        t=np.arange(base.C.duration_s)
        temperature=25+74*(1-np.exp(-(t/tau)**1.2))
        self.assertEqual(temperature[0],25)
        self.assertTrue(np.all(np.diff(temperature)>=0))
        self.assertLess(np.max(np.abs(raw[:,2]-temperature)),2.)


if __name__=='__main__':unittest.main(verbosity=2)
