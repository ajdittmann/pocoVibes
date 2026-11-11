"""
Checks for fixes 6-10 (they need a real torch/zuko installation).

Run from anywhere with::

    python tests/check_fixes_6_to_10.py

Each check prints PASS or FAIL with the numbers it is based on; the exit code is the
number of failed checks.

6. Periodic parameters: in a preconditioned run in which the posterior mode of a periodic
   parameter lies on its bounds, the point at which the period is cut open is moved to the
   opposite side of the period, and the evidence and the posterior agree with their
   analytic values. (The earlier check of the boundary step in the MCMC kernels was
   removed together with that step.)
7. Flow.fit(validation_split=v) trains on (1 - v) of the samples and validates on v.
8. Flow.fit keeps the best model (lowest monitored loss) when all epochs are used
   without early stopping.
9. The importance-sampling evidence counts samples with non-finite prior (or NaN
   likelihood) as zero weight and divides by the number of samples drawn.
10. Preconditioned runs refit theta_geometry (through the current flow) at every SMC
    iteration, also when the flow is not retrained, and no longer fit u_geometry.
"""
import io
import sys
import contextlib
from pathlib import Path

# Check the pocomc package of this repository (not an installed copy)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from scipy.stats import uniform, norm
from scipy.special import i0e

import pocomc
import pocomc.flow as flow_module
from pocomc import Sampler, Prior
from pocomc.flow import Flow
from pocomc.mcmc import preconditioned_rwm, preconditioned_pcn
from pocomc.scaler import Reparameterize
from pocomc.geometry import Geometry
from pocomc.tools import ProgressBar, logsumexp

RESULTS = []


def report(name, passed, details):
    RESULTS.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    for line in details:
        print(f"         {line}")


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


# ---------------------------------------------------------------------------
# 6. Periodic parameters
# ---------------------------------------------------------------------------

def check_6():
    # The first parameter is a phase in [0, 2 pi] with a von Mises likelihood whose mode lies on
    # the bounds (0 = 2 pi); the second parameter has a Gaussian likelihood. The evidence is
    # known analytically.
    kappa, s, sigma = 50.0, 0.5, 3.0
    logz_true = np.log(i0e(kappa)) + 0.5 * np.log(s ** 2 / (sigma ** 2 + s ** 2)) - 0.5 / (sigma ** 2 + s ** 2)
    prior = Prior([uniform(0.0, 2.0 * np.pi), norm(0.0, sigma)])
    sampler = Sampler(prior, lambda z: kappa * (np.cos(z[0]) - 1.0) - 0.5 * (z[1] - 1.0) ** 2 / s ** 2,
                      periodic=[0], random_state=6)
    with quiet():
        sampler.run(n_total=2048, n_evidence=2048, progress=False)
    samples, weights, _, _ = sampler.posterior()
    logz, logz_err = sampler.evidence()
    shift = float(sampler.scaler.periodic_shift[0])
    resultant = np.sum(weights * np.exp(1j * samples[:, 0]))
    circular_mean = float(np.angle(resultant))
    circular_std = float(np.sqrt(-2.0 * np.log(np.abs(resultant))))
    passed = bool(abs(shift - 0.5) < 0.1 and abs(logz - logz_true) < 0.3 and abs(circular_mean) < 0.05
                  and abs(circular_std - 1.0 / np.sqrt(kappa)) < 0.03)
    report("6. periodic parameter with the posterior mode on its bounds (preconditioned run)", passed, [
        f"shift of the cut: {shift:.3f} of the period (expected about 0.5)",
        f"logZ = {logz:.3f} +- {logz_err:.3f} (analytic {logz_true:.3f})",
        f"circular mean of the phase: {circular_mean:+.4f} (expected 0), "
        f"circular std: {circular_std:.4f} (expected about {1.0 / np.sqrt(kappa):.4f})",
    ])


# ---------------------------------------------------------------------------
# 7. validation_split
# ---------------------------------------------------------------------------

def check_7():
    torch.manual_seed(7)
    n = 1000
    x = torch.randn(n, 2)
    calls = []
    original = flow_module._batch_indices

    def recording(n_items, batch_size, shuffle):
        calls.append(n_items)
        return original(n_items, batch_size, shuffle)

    details = []
    passed = True
    flow_module._batch_indices = recording
    try:
        for split in (0.2, 0.5, 0.8):
            calls.clear()
            Flow(2, "maf3").fit(x, validation_split=split, epochs=1, batch_size=100, patience=5)
            n_train, n_valid = calls[0], calls[1]
            expected_valid = n - int((1.0 - split) * n)
            ok = n_valid == expected_valid and n_train == n - expected_valid
            passed &= ok
            details.append(f"validation_split={split}: trained on {n_train}, validated on {n_valid} "
                           f"(expected {n - expected_valid} / {expected_valid})")
    finally:
        flow_module._batch_indices = original
    report("7. validation_split is the fraction held out for validation", passed, details)


# ---------------------------------------------------------------------------
# 8. Best model kept when epochs run out
# ---------------------------------------------------------------------------

def check_8():
    details = []
    for seed in range(20):
        torch.manual_seed(seed)
        x = torch.randn(400, 2) * torch.tensor([1.0, 0.2]) + torch.tensor([3.0, -1.0])
        flow = Flow(2, "maf3")
        # A large learning rate makes the loss noisy, so the last epoch is usually not the best;
        # the patience is large enough that early stopping never triggers
        history = flow.fit(x, validation_split=0.5, epochs=25, batch_size=200, patience=1000,
                           learning_rate=5e-2, shuffle=False)
        val = np.asarray(history["val_loss"])
        if len(val) != 25:
            continue  # stopped early: not the case being tested
        best, last = int(np.argmin(val)), len(val) - 1
        if best == last:
            continue  # best is the last epoch: nothing to distinguish, try another seed
        x_valid = x[int(0.5 * len(x)):]
        with torch.no_grad():
            final_loss = (-flow.flow().log_prob(x_valid).sum()).item() / len(x_valid)
        passed = abs(final_loss - val[best]) <= 1e-4 * max(1.0, abs(val[best]))
        details.append(f"seed {seed}: best epoch {best} (val loss {val[best]:.4f}), last epoch {last} "
                       f"(val loss {val[last]:.4f}); loss of the returned model = {final_loss:.4f}")
        details.append("the returned model must have the best epoch's loss, not the last epoch's")
        report("8. Flow.fit keeps the best model when all epochs are used", passed, details)
        return
    report("8. Flow.fit keeps the best model when all epochs are used", False,
           ["inconclusive: no seed produced a run whose best epoch was not the last one"])


# ---------------------------------------------------------------------------
# 9. Importance-sampling evidence
# ---------------------------------------------------------------------------

def check_9():
    np.random.seed(9)
    torch.manual_seed(9)

    def loglike(z):
        # NaN likelihood (zero weight) where z[1] > 0.9
        return np.nan if z[1] > 0.9 else 0.0

    prior = Prior([uniform(0.0, 1.0), uniform(0.0, 1.0)])
    sampler = Sampler(prior, loglike, n_effective=512, n_active=256, random_state=9)
    sampler.prior_samples = sampler.sample_prior(sampler.n_prior)
    sampler.scaler.fit(sampler.prior_samples)
    sampler.pbar = ProgressBar(False)
    # Prior density 2 on z[0] < 0.5 and zero (non-finite log-prior) elsewhere: normalized.
    # With the NaN region, the true evidence is 0.9 -> logZ = log(0.9).
    sampler.log_prior = lambda z: np.where(z[:, 0] < 0.5, np.log(2.0), -np.inf)
    n = 200_000
    calls_before = sampler.calls
    np.random.seed(9)
    torch.manual_seed(9)
    logz, dlogz = sampler._compute_evidence(n)
    n_evaluated = sampler.calls - calls_before

    # Reproduce what the previous code returned: it divided by the number of samples with a
    # finite prior and propagated NaN likelihoods
    np.random.seed(9)
    torch.manual_seed(9)
    with torch.no_grad():
        theta_q, logq = sampler.flow.sample(n)
    theta_q, logq = theta_q.numpy(), logq.numpy()
    x_q, logdetj = sampler.scaler.inverse(theta_q)
    logp = sampler.log_prior(x_q)
    keep = np.isfinite(logp)
    logl_old = np.array([loglike(z) for z in x_q[keep]])
    logw_old = logl_old + logp[keep] + logdetj[keep] - logq[keep]
    logz_old = logsumexp(logw_old) - np.log(len(logw_old))

    truth = np.log(0.9)
    passed = abs(logz - truth) < max(5 * dlogz, 0.02) and n_evaluated == int(np.sum(keep))
    report("9. importance-sampling evidence counts invalid samples as zero weight", passed, [
        f"true logZ = {truth:.4f}; fixed estimate = {logz:.4f} +- {dlogz:.4f}; previous code would give {logz_old}",
        f"likelihood calls counted: {n_evaluated} (samples with finite prior: {int(np.sum(keep))} of {n})",
    ])


# ---------------------------------------------------------------------------
# 10. theta_geometry refit at every SMC iteration
# ---------------------------------------------------------------------------

def check_10():
    prior = Prior([norm(0.0, 3.0), norm(0.0, 3.0), norm(0.0, 3.0)])
    sampler = Sampler(prior, lambda z: -0.5 * np.sum((z - 1.0) ** 2 / 0.1),
                      n_effective=256, n_active=128, train_frequency=1000, random_state=10)
    counts = dict(theta=0, u=0, flow=0)

    def wrap(obj, name, key):
        original = getattr(obj, name)

        def wrapped(*args, **kwargs):
            counts[key] += 1
            return original(*args, **kwargs)
        setattr(obj, name, wrapped)

    wrap(sampler.theta_geometry, "fit", "theta")
    wrap(sampler.u_geometry, "fit", "u")
    wrap(sampler.flow, "fit", "flow")
    with quiet():
        sampler.run(n_total=512, n_evidence=512, progress=False)
    n_smc = sampler.t - sampler.n_prior // sampler.n_active
    passed = counts["theta"] == n_smc and counts["u"] == 0 and counts["flow"] < n_smc
    report("10. preconditioned runs refit theta_geometry at every SMC iteration", passed, [
        f"SMC iterations: {n_smc}; flow trainings: {counts['flow']} (so {n_smc - counts['flow']} iterations without retraining)",
        f"theta_geometry fits: {counts['theta']} (expected {n_smc}); u_geometry fits: {counts['u']} (expected 0)",
    ])


if __name__ == "__main__":
    print(f"pocomc imported from {Path(pocomc.__file__).parent}\n")
    for check in (check_6, check_7, check_8, check_9, check_10):
        try:
            check()
        except Exception as e:  # report and continue with the other checks
            report(f"{check.__name__} crashed", False, [f"{type(e).__name__}: {e}"])
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(len(RESULTS) - sum(RESULTS))
