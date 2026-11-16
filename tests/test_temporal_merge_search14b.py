"""Numerical invariants and independent NumPy references for temporal proposals."""
import unittest
import numpy as np
import torch

from research.temporal_merge_search14b.methods import calibrate, fit_portfolio, transform


class TemporalProposals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = np.array([[.1, .2, .1, .3, .2, .1], [.3, .4, .2, .5, .4, .3],
                          [.6, .8, .7, .7, .8, .6], [.5, .6, .5, .8, .7, .4]])
        cls.c = calibrate(cls.v, ['a', 'a', 'b', 'b', 'c', 'c'], draws=12)
        cls.values = [torch.tensor([1., 2., 3., 4.]), torch.tensor([1.1, 2.1, 2.9, 4.]),
                      torch.tensor([1.2, 2.3, 3.2, 4.]), torch.tensor([1.1, 2.4, 3., 4.])]
        cls.weights = [.05, .1, .6, .25]

    def test_portfolio_beats_its_prior_objective(self):
        p, prior = np.array([[.9, .2], [.3, .7]]), np.array([.5, .5])
        w = fit_portfolio(p, prior)
        loss = lambda z: -np.log(p@z).mean()+.02*np.sum(z*np.log(z/prior))
        self.assertAlmostEqual(w.sum(), 1)
        self.assertTrue((w > 0).all())
        self.assertLessEqual(loss(w), loss(prior)+1e-10)

    def test_calibration_covariance_is_psd_and_factor_matches(self):
        covariance = np.array(self.c['portfolio_covariance'])
        factor = np.array(self.c['covariance_factor'])
        np.testing.assert_allclose(factor.T@factor, covariance, atol=1e-12)
        np.testing.assert_allclose(covariance.sum(0), 0, atol=1e-12)
        np.testing.assert_allclose(np.array(self.c['residual_projection'])@np.ones(4), 0, atol=1e-12)

    def test_independent_numpy_reference_for_all_seven_arms(self):
        values = np.stack([v.numpy().astype(float) for v in self.values])
        a, c = values[-1], self.c
        x = values-a
        cap = 2*np.abs(np.array(self.weights)@x)
        rollback, beneficial, mass = np.zeros_like(a), np.zeros_like(a), 0.
        for j, delta in enumerate(np.diff(values, axis=0)):
            conf = c['negative_interval_confidence'][j]
            if conf > 0 and mass > 0:
                b = beneficial/mass
                overlap = 2*np.minimum(abs(delta),abs(b))/np.maximum(abs(delta)+abs(b),1e-30)
                rollback += conf*overlap*(delta*b<0)*delta
            if c['quality_differences'][j] > 0:
                beneficial += c['quality_differences'][j]*delta
                mass += c['quality_differences'][j]
        qis = np.clip(-rollback, -cap, cap)
        slope = np.array(c['quality_residual'])@x/c['quality_residual_norm2']
        residual2 = ((np.array(c['residual_projection'])@x)**2).sum(0)
        reliability = np.clip(slope**2*c['quality_residual_norm2']/np.maximum(residual2,1e-30),0,1)
        drr = reliability*np.clip(slope*c['quality_target_gain'],-cap,cap)
        update = np.array(c['portfolio_weights'])@x
        variance = np.einsum('ik,ij,jk->k', x, np.array(c['portfolio_covariance']), x)
        actual = transform(self.values, c, self.weights)
        np.testing.assert_allclose(actual['bcm_linear'], a+update, atol=3e-7)
        for s, suffix in [(.5,'05'),(1.,'10')]:
            expected = {'qis':a+s*qis, 'drr':a+s*drr,
                        'bcm':a+update**3/np.maximum(update**2+s*variance,1e-30)}
            for name, value in expected.items():
                np.testing.assert_allclose(actual[name+'_'+suffix], value, atol=3e-7)

    def test_flat_coordinates_and_common_translation_are_preserved(self):
        output = transform(self.values, self.c, self.weights)
        moved = transform([x+8 for x in self.values], self.c, self.weights)
        for name, value in output.items():
            self.assertEqual(float(value[-1]), 4.)
            torch.testing.assert_close(moved[name], value+8, atol=2e-6, rtol=0)

    def test_quality_without_nonlinear_time_signal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'linear time'):
            calibrate(np.array([[.1,.1],[.2,.2],[.3,.3]]), ['a','b'], draws=4)


if __name__ == '__main__':
    unittest.main()
