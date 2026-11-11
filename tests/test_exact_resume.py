"""
Exact-resume tests for pocoMC checkpointing.

Each test runs the sampler once without interruption (the reference), then runs it
again with checkpoints enabled and a simulated crash (an exception raised inside the
likelihood after a given number of likelihood calls), resumes from the latest state
file in a freshly created sampler, and checks that every output is bit-for-bit
identical to the reference: posterior samples, weights, log-likelihoods, log-priors,
log-evidence and its error, the full results dictionary, the number of likelihood
calls and the number of iterations.

Run with ``python -m pytest tests/test_exact_resume.py`` or directly with
``python tests/test_exact_resume.py``.
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import norm, uniform

from pocomc import Sampler, Prior


class Crash(Exception):
    """Simulated interruption of a run."""


class CountingLikelihood:
    """Rosenbrock-like log-likelihood that can raise ``Crash`` after ``crash_at`` calls."""

    def __init__(self, crash_at=None, blobs=False, zero_region=False):
        self.calls = 0
        self.crash_at = crash_at
        self.blobs = blobs
        self.zero_region = zero_region

    def __call__(self, x):
        self.calls += 1
        if self.crash_at is not None and self.calls > self.crash_at:
            raise Crash()
        logl = -0.5 * (10.0 * (x[1] - x[0] ** 2) ** 2 + (x[0] - 1.0) ** 2) - 0.5 * np.sum(x[2:] ** 2)
        if self.zero_region and x[0] < -1.0:
            logl = -np.inf  # zero likelihood on part of the prior
        if self.blobs:
            return logl, np.sum(x)
        return logl


class PeriodicLikelihood(CountingLikelihood):
    """Log-likelihood whose mode lies on the bounds of the second parameter (-5 = 5), which is periodic."""

    def __call__(self, x):
        self.calls += 1
        if self.crash_at is not None and self.calls > self.crash_at:
            raise Crash()
        return 20.0 * (np.cos(2.0 * np.pi * (x[1] + 5.0) / 10.0) - 1.0) - 0.5 * (x[0] - 1.0) ** 2 - 0.5 * x[2] ** 2


def make_sampler(likelihood, output_dir, precondition, sample, blobs, periodic=None):
    prior = Prior([uniform(-5, 10), uniform(-5, 10), norm(0, 2)])
    return Sampler(prior, likelihood,
                   n_effective=64, n_active=32,
                   precondition=precondition, sample=sample,
                   train_config=dict(epochs=20),
                   blobs_dtype=float if blobs else None,
                   periodic=periodic,
                   output_dir=output_dir, output_label="pmc",
                   random_state=0)


def outputs(sampler):
    """Collect every output that must be identical between runs."""
    if sampler.have_blobs:
        samples, weights, logl, logp, blobs = sampler.posterior(return_blobs=True)
    else:
        samples, weights, logl, logp = sampler.posterior()
        blobs = None
    logz, logz_err = sampler.evidence()
    return dict(samples=samples, weights=weights, logl=logl, logp=logp, blobs=blobs,
                logz=logz, logz_err=logz_err, results=sampler.results,
                calls=sampler.calls, t=sampler.t)


def assert_identical(testcase, a, b, path="outputs"):
    """Bit-for-bit comparison of nested outputs."""
    if isinstance(a, dict):
        testcase.assertEqual(set(a.keys()), set(b.keys()), path)
        for key in a:
            assert_identical(testcase, a[key], b[key], f"{path}[{key!r}]")
        return
    if a is None or b is None:
        testcase.assertIs(a, b, path)
        return
    a = np.asarray(a)
    b = np.asarray(b)
    testcase.assertEqual(a.shape, b.shape, path)
    testcase.assertEqual(a.dtype, b.dtype, path)
    if a.dtype.hasobject:
        for ai, bi in zip(a.ravel(), b.ravel()):
            testcase.assertTrue((ai is None and bi is None) or np.array_equal(ai, bi, equal_nan=True), path)
    else:
        testcase.assertTrue(a.tobytes() == b.tobytes(), f"{path} differs")


RUN_KWARGS = dict(n_total=256, n_evidence=256, progress=False)


class ExactResumeTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _reference(self, precondition, sample, blobs):
        sampler = make_sampler(CountingLikelihood(blobs=blobs), self.tmp / "ref", precondition, sample, blobs)
        sampler.run(**RUN_KWARGS)
        return outputs(sampler)

    def _check(self, precondition, sample, blobs, crash_points, label):
        reference = self._reference(precondition, sample, blobs)
        total_calls = reference["calls"]

        # Saving checkpoints at every opportunity must not change the results
        out_dir = self.tmp / "saved"
        sampler = make_sampler(CountingLikelihood(blobs=blobs), out_dir, precondition, sample, blobs)
        sampler.run(save_interval=0, **RUN_KWARGS)
        assert_identical(self, reference, outputs(sampler), f"{label} with checkpoints")

        for frac in crash_points:
            crash_at = int(frac * total_calls)
            with self.subTest(label=label, crash_at=crash_at):
                out_dir = self.tmp / f"crash_{crash_at}"
                sampler = make_sampler(CountingLikelihood(crash_at=crash_at, blobs=blobs), out_dir,
                                       precondition, sample, blobs)
                with self.assertRaises(Crash):
                    sampler.run(save_interval=0, **RUN_KWARGS)
                latest = out_dir / "pmc_latest.state"
                self.assertTrue(latest.exists())
                resumed = make_sampler(CountingLikelihood(blobs=blobs), out_dir, precondition, sample, blobs)
                resumed.run(resume_state_path=latest, save_interval=0, **RUN_KWARGS)
                assert_identical(self, reference, outputs(resumed), f"{label} resumed at call {crash_at}")

    def test_exact_resume(self):
        crash_points = [0.01, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
        for precondition in (True, False):
            for sample in ("tpcn", "rwm"):
                for blobs in (False, True):
                    label = f"precondition={precondition}/{sample}/blobs={blobs}"
                    try:
                        self._check(precondition, sample, blobs, crash_points, label)
                    except RuntimeError as e:
                        if "stub" in str(e):  # sandbox without a real torch: flow cannot be trained
                            continue
                        raise

    def test_resume_from_iteration_snapshots(self):
        # Resuming from any numbered snapshot (save_every=1) gives the reference results
        for precondition in (True, False):
            label = f"precondition={precondition}"
            try:
                reference = self._reference(precondition, "tpcn", False)
            except RuntimeError as e:
                if "stub" in str(e):
                    continue
                raise
            out_dir = self.tmp / f"snap_{precondition}"
            sampler = make_sampler(CountingLikelihood(), out_dir, precondition, "tpcn", False)
            sampler.run(save_every=1, **RUN_KWARGS)
            snapshots = sorted(out_dir.glob("pmc_[0-9]*.state"), key=lambda p: int(p.stem.split("_")[1]))
            self.assertGreater(len(snapshots), 2)
            for snapshot in [snapshots[-1], snapshots[len(snapshots) // 2], snapshots[0]]:
                with self.subTest(label=label, snapshot=snapshot.name):
                    resumed = make_sampler(CountingLikelihood(), out_dir, precondition, "tpcn", False)
                    resumed.run(resume_state_path=snapshot, **RUN_KWARGS)
                    assert_identical(self, reference, outputs(resumed), f"{label} from {snapshot.name}")

    def test_exact_resume_zero_likelihood_region(self):
        # Prior samples with zero likelihood change the warm-up log-evidences, which are pooled
        # (and the history rewritten) at the end of warm-up; resuming must still be exact
        reference_sampler = make_sampler(CountingLikelihood(zero_region=True), self.tmp / "ref", False, "tpcn", False)
        reference_sampler.run(**RUN_KWARGS)
        reference = outputs(reference_sampler)
        self.assertTrue(np.any(np.asarray(reference["results"]["logz"][:4]) != 0.0))
        for frac in [0.01, 0.05, 0.1, 0.3, 0.6, 0.9]:
            crash_at = int(frac * reference["calls"])
            with self.subTest(crash_at=crash_at):
                out_dir = self.tmp / f"zero_{crash_at}"
                sampler = make_sampler(CountingLikelihood(crash_at=crash_at, zero_region=True), out_dir, False, "tpcn", False)
                with self.assertRaises(Crash):
                    sampler.run(save_interval=0, **RUN_KWARGS)
                resumed = make_sampler(CountingLikelihood(zero_region=True), out_dir, False, "tpcn", False)
                resumed.run(resume_state_path=out_dir / "pmc_latest.state", save_interval=0, **RUN_KWARGS)
                assert_identical(self, reference, outputs(resumed), f"zero region, resumed at call {crash_at}")

    def test_exact_resume_periodic(self):
        # The cut of a periodic parameter is moved during the run (the mode of the likelihood lies on
        # the bounds of the parameter); the position of the cut is part of the state, so resuming
        # must still be exact
        for precondition in (True, False):
            label = f"periodic, precondition={precondition}"
            try:
                reference_sampler = make_sampler(PeriodicLikelihood(), self.tmp / f"ref_{precondition}",
                                                 precondition, "tpcn", False, periodic=[1])
                reference_sampler.run(**RUN_KWARGS)
            except RuntimeError as e:
                if "stub" in str(e):  # sandbox without a real torch: flow cannot be trained
                    continue
                raise
            reference = outputs(reference_sampler)
            self.assertNotEqual(reference_sampler.scaler.periodic_shift[1], 0.0)
            for frac in [0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]:
                crash_at = int(frac * reference["calls"])
                with self.subTest(label=label, crash_at=crash_at):
                    out_dir = self.tmp / f"periodic_{precondition}_{crash_at}"
                    sampler = make_sampler(PeriodicLikelihood(crash_at=crash_at), out_dir, precondition, "tpcn", False, periodic=[1])
                    with self.assertRaises(Crash):
                        sampler.run(save_interval=0, **RUN_KWARGS)
                    resumed = make_sampler(PeriodicLikelihood(), out_dir, precondition, "tpcn", False, periodic=[1])
                    resumed.run(resume_state_path=out_dir / "pmc_latest.state", save_interval=0, **RUN_KWARGS)
                    assert_identical(self, reference, outputs(resumed), f"{label}, resumed at call {crash_at}")
                    self.assertEqual(resumed.scaler.periodic_shift.tobytes(), reference_sampler.scaler.periodic_shift.tobytes())

    def test_resume_from_final_state(self):
        # Resuming from the final state does not repeat any work
        reference = self._reference(False, "tpcn", False)
        out_dir = self.tmp / "final"
        sampler = make_sampler(CountingLikelihood(), out_dir, False, "tpcn", False)
        sampler.run(save_interval=1e9, **RUN_KWARGS)
        likelihood = CountingLikelihood()
        resumed = make_sampler(likelihood, out_dir, False, "tpcn", False)
        resumed.run(resume_state_path=out_dir / "pmc_final.state", **RUN_KWARGS)
        self.assertEqual(likelihood.calls, 0)
        assert_identical(self, reference, outputs(resumed), "from final")


if __name__ == "__main__":
    unittest.main(verbosity=2)
