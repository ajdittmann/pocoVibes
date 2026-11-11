import unittest
import numpy as np

from pocomc.scaler import Reparameterize


class ReparameterizeTestCase(unittest.TestCase):
    @staticmethod
    def make_unconstrained_data():
        # Make a dataset to use in tests
        np.random.seed(0)
        n_data = 100
        n_dim = 10
        x = np.random.randn(n_data, n_dim) * 3 + 1

        lower_bound = np.nan
        upper_bound = np.nan
        return x, lower_bound, upper_bound

    @staticmethod
    def make_lower_bounded_data():
        # Make a dataset to use in tests
        np.random.seed(0)
        n_data = 100
        n_dim = 10
        x = np.random.exponential(scale=1, size=(n_data, n_dim))

        lower_bound = 0
        upper_bound = np.nan
        return x, lower_bound, upper_bound

    @staticmethod
    def make_upper_bounded_data():
        # Make a dataset to use in tests
        np.random.seed(0)
        n_data = 100
        n_dim = 10
        x = -np.random.exponential(scale=1, size=(n_data, n_dim))

        lower_bound = np.nan
        upper_bound = 0
        return x, lower_bound, upper_bound

    @staticmethod
    def make_lower_and_upper_bounded_data():
        # Make a dataset to use in tests
        np.random.seed(0)
        n_data = 100
        n_dim = 10
        x = np.random.uniform(low=0, high=1, size=(n_data, n_dim))

        lower_bound = 0
        upper_bound = 1
        return x, lower_bound, upper_bound

    def test_unconstrained(self):
        # Test that methods work without errors on unconstrained input data
        np.random.seed(0)

        x, lb, ub = self.make_unconstrained_data()
        r = Reparameterize(n_dim=x.shape[1], bounds=(lb, ub))
        r.fit(x)

        u = r.forward(x)
        x_r, _ = r.inverse(u)

        self.assertEqual(x.shape, u.shape)
        self.assertEqual(x.dtype, u.dtype)

        self.assertEqual(x_r.shape, u.shape)
        self.assertEqual(x_r.dtype, u.dtype)

        self.assertTrue(np.allclose(x, x_r))

    def test_lower_bounded(self):
        # Test that methods work without errors on unconstrained input data
        np.random.seed(0)

        x, lb, ub = self.make_lower_bounded_data()
        r = Reparameterize(n_dim=x.shape[1], bounds=(lb, ub))
        r.fit(x)

        u = r.forward(x)
        x_r, _ = r.inverse(u)

        self.assertEqual(x.shape, u.shape)
        self.assertEqual(x.dtype, u.dtype)

        self.assertEqual(x_r.shape, u.shape)
        self.assertEqual(x_r.dtype, u.dtype)

        self.assertTrue(np.allclose(x, x_r))

    def test_upper_bounded(self):
        # Test that methods work without errors on unconstrained input data
        np.random.seed(0)

        x, lb, ub = self.make_upper_bounded_data()
        r = Reparameterize(n_dim=x.shape[1], bounds=(lb, ub))
        r.fit(x)

        u = r.forward(x)
        x_r, _ = r.inverse(u)

        self.assertEqual(x.shape, u.shape)
        self.assertEqual(x.dtype, u.dtype)

        self.assertEqual(x_r.shape, u.shape)
        self.assertEqual(x_r.dtype, u.dtype)

        self.assertTrue(np.allclose(x, x_r))

    def test_lower_and_upper_bounded(self):
        # Test that methods work without errors on unconstrained input data
        np.random.seed(0)

        x, lb, ub = self.make_lower_and_upper_bounded_data()
        r = Reparameterize(n_dim=x.shape[1], bounds=(lb, ub))
        r.fit(x)

        u = r.forward(x)
        x_r, _ = r.inverse(u)

        self.assertEqual(x.shape, u.shape)
        self.assertEqual(x.dtype, u.dtype)

        self.assertEqual(x_r.shape, u.shape)
        self.assertEqual(x_r.dtype, u.dtype)

        self.assertTrue(np.allclose(x, x_r))

    def test_out_of_bounds(self):
        # Test that providing out-of-bound inputs raises an error
        np.random.seed(0)
        x, lb, ub = self.make_lower_and_upper_bounded_data()
        x[0] = lb - 1  # Artificially make example 0 go outside the bounds
        x[1] = ub + 1  # Artificially make example 1 go outside the bounds

        r = Reparameterize(n_dim=x.shape[1], bounds=(lb, ub))
        self.assertRaises(ValueError, r.fit, x)

    @staticmethod
    def make_periodic_data():
        # Parameter 0 is a phase in [0, 2 pi] whose samples lie across the bounds (around 0 = 2 pi);
        # parameter 1 has both bounds and parameter 2 has none
        np.random.seed(0)
        n_data = 2000
        bounds = np.array([[0.0, 2.0 * np.pi], [-1.0, 3.0], [-np.inf, np.inf]])
        x_prior = np.column_stack([np.random.uniform(0.0, 2.0 * np.pi, n_data),
                                   np.random.uniform(-1.0, 3.0, n_data),
                                   np.random.randn(n_data)])
        x = x_prior.copy()
        x[:, 0] = np.mod(0.3 * np.random.randn(n_data), 2.0 * np.pi)
        return x_prior, x, bounds

    def test_periodic_zero_shift(self):
        # Without a shift, a periodic parameter is transformed exactly like a bounded one
        x_prior, x, bounds = self.make_periodic_data()
        for transform in ("probit", "logit"):
            r = Reparameterize(n_dim=3, bounds=bounds, transform=transform)
            r.fit(x_prior)
            p = Reparameterize(n_dim=3, bounds=bounds, periodic=[0], transform=transform)
            p.fit(x_prior)

            u_r = r.forward(x)
            u_p = p.forward(x)
            self.assertTrue(np.array_equal(u_r, u_p))
            self.assertTrue(np.array_equal(r.inverse(u_r)[0], p.inverse(u_p)[0]))
            self.assertTrue(np.array_equal(r.inverse(u_r)[1], p.inverse(u_p)[1]))

    def test_periodic_shift(self):
        # Samples across the bounds of a periodic parameter are centred by the shift
        x_prior, x, bounds = self.make_periodic_data()
        for transform in ("probit", "logit"):
            r = Reparameterize(n_dim=3, bounds=bounds, periodic=[0], transform=transform)
            r.fit(x_prior)

            u_before = r.forward(x)
            self.assertTrue(r.update_periodic_shift(x))
            self.assertTrue(abs(r.periodic_shift[0] - 0.5) < 0.05)
            self.assertTrue(np.all(r.periodic_shift[1:] == 0.0))
            u = r.forward(x)

            # The two halves of the samples (at the two bounds) are joined in the middle
            self.assertTrue(np.std(u_before[:, 0]) > 1.0)
            self.assertTrue(np.std(u[:, 0]) < 0.3)
            self.assertTrue(abs(np.mean(u[:, 0])) < 0.1)

            # The other parameters are not affected
            self.assertTrue(np.array_equal(u_before[:, 1:], u[:, 1:]))

            # The transformation is invertible (the two bounds are the same point)
            x_r, log_det_J = r.inverse(u)
            d = np.abs(x_r - x)
            d[:, 0] = np.minimum(d[:, 0], 2.0 * np.pi - d[:, 0])
            self.assertTrue(np.all(d < 1e-10))
            self.assertTrue(np.all((x_r[:, 0] >= 0.0) & (x_r[:, 0] <= 2.0 * np.pi)))

            # The Jacobian is that of the transformation without the shift
            s = Reparameterize(n_dim=3, bounds=bounds, transform=transform)
            s.fit(x_prior)
            self.assertTrue(np.allclose(log_det_J, s.inverse(u)[1]))

            # Centred samples do not change the shift again
            shift = r.periodic_shift.copy()
            self.assertFalse(r.update_periodic_shift(x))
            self.assertTrue(np.array_equal(shift, r.periodic_shift))

    def test_periodic_shift_weights(self):
        # Only the samples with non-zero weight determine the shift
        x_prior, x, bounds = self.make_periodic_data()
        r = Reparameterize(n_dim=3, bounds=bounds, periodic=[0])
        r.fit(x_prior)

        x_all = np.concatenate([x_prior, x])
        weights = np.concatenate([np.zeros(len(x_prior)), np.ones(len(x))])
        self.assertTrue(r.update_periodic_shift(x_all, weights))
        self.assertTrue(abs(r.periodic_shift[0] - 0.5) < 0.05)

    def test_periodic_uniform_samples(self):
        # Samples spread over the whole period do not move the cut
        x_prior, _, bounds = self.make_periodic_data()
        r = Reparameterize(n_dim=3, bounds=bounds, periodic=[0])
        r.fit(x_prior)
        self.assertFalse(r.update_periodic_shift(x_prior))
        self.assertTrue(np.all(r.periodic_shift == 0.0))

    def test_periodic_needs_bounds(self):
        # A periodic parameter needs a finite lower and upper bound (and a valid index)
        _, _, bounds = self.make_periodic_data()
        self.assertRaises(ValueError, Reparameterize, 3, bounds, [2])
        self.assertRaises(ValueError, Reparameterize, 3, bounds, [3])


if __name__ == '__main__':
    unittest.main()
