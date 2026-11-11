import unittest
import numpy as np

from scipy.stats import norm, uniform
from scipy.special import i0e

from pocomc.sampler import Sampler
from pocomc.prior import Prior

class SamplerTestCase(unittest.TestCase):
    @staticmethod
    def log_likelihood_single(x):
        return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x ** 2)

    @staticmethod
    def log_likelihood_vectorized(x):
        # Gaussian log likelihood with mu = 0, sigma = 1
        return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x ** 2, axis=1)

    def test_run(self):
        
        n_dim = 2
        prior = Prior(n_dim*[norm(0, 1)])

        sampler = Sampler(
            prior=prior,
            likelihood=self.log_likelihood_single,
            train_config={'epochs': 1},
            random_state=0,
        )
        sampler.run()

    def test_run2(self):
        
        n_dim = 2
        prior = Prior(n_dim*[norm(0, 1)])

        sampler = Sampler(
            prior=prior,
            likelihood=self.log_likelihood_vectorized,
            vectorize=True,
            train_config={'epochs': 1},
            random_state=0,
        )
        sampler.run()

    def test_periodic(self):
        # The first parameter is a phase in [0, 2 pi] whose posterior mode lies on the bounds
        # (0 = 2 pi), with a von Mises likelihood exp(kappa * (cos(phase) - 1)). The second
        # parameter has a Gaussian likelihood. The evidence is known analytically.
        kappa, s, sigma = 50.0, 0.5, 3.0

        def log_likelihood(x):
            return kappa * (np.cos(x[:, 0]) - 1.0) - 0.5 * (x[:, 1] - 1.0) ** 2 / s ** 2

        logz_true = np.log(i0e(kappa)) + 0.5 * np.log(s ** 2 / (sigma ** 2 + s ** 2)) - 0.5 / (sigma ** 2 + s ** 2)

        prior = Prior([uniform(0.0, 2.0 * np.pi), norm(0.0, sigma)])

        sampler = Sampler(
            prior=prior,
            likelihood=log_likelihood,
            vectorize=True,
            periodic=[0],
            precondition=False,
            random_state=0,
        )
        sampler.run(n_total=2048, progress=False)

        # The cut of the period was moved away from the mode, to the opposite side of the period
        self.assertTrue(abs(sampler.scaler.periodic_shift[0] - 0.5) < 0.1)

        # Evidence
        logz, _ = sampler.evidence()
        self.assertTrue(abs(logz - logz_true) < 0.5)

        # Posterior of the phase: circular mean 0 and circular standard deviation ~ 1 / sqrt(kappa)
        samples, weights, _, _ = sampler.posterior()
        self.assertTrue(np.all((samples[:, 0] >= 0.0) & (samples[:, 0] <= 2.0 * np.pi)))
        resultant = np.sum(weights * np.exp(1j * samples[:, 0]))
        self.assertTrue(abs(np.angle(resultant)) < 0.05)
        circular_std = np.sqrt(-2.0 * np.log(np.abs(resultant)))
        self.assertTrue(abs(circular_std - 1.0 / np.sqrt(kappa)) < 0.03)

    def test_reflective_removed(self):
        # The reflective argument no longer exists
        prior = Prior(2 * [uniform(0.0, 1.0)])
        with self.assertRaises(TypeError):
            Sampler(prior=prior, likelihood=self.log_likelihood_single, reflective=[0])


if __name__ == '__main__':
    unittest.main()
