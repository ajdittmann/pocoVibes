"""
Benchmark of the cost of saving a checkpoint.

A synthetic sampler state of realistic size is built (past iterations, current particles,
the sampler's real normalizing flow and an MCMC loop state, i.e. a checkpoint taken in the
middle of the mutation step), and the time of repeated saves is measured. Between saves one
new iteration is appended, as happens during a run. For comparison, the time of a
whole-state dill pickle (the previous state file format) of the same content is shown.

Usage::

    python benchmarks/checkpoint_benchmark.py

The overhead of checkpointing after every MCMC step is the save time divided by the
duration of one MCMC step (one likelihood evaluation per active particle, e.g. 2 s with
``n_active`` parallel workers and a 2 s likelihood).
"""
import os
import sys
import time
import shutil
import tempfile
from pathlib import Path

import dill
import numpy as np
from scipy.stats import uniform

# Benchmark the pocomc package of this repository (not an installed copy)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pocomc
from pocomc import Sampler, Prior


def build(n_dim, n_active, n_iter, output_dir):
    prior = Prior([uniform(-5, 10)] * n_dim)
    s = Sampler(prior, lambda x: 0.0, n_active=n_active, n_effective=2 * n_active,
                output_dir=output_dir, random_state=0)
    rng = np.random.default_rng(0)
    s.prior_samples = rng.standard_normal((s.n_prior, n_dim))

    def particles():
        return dict(u=rng.standard_normal((n_active, n_dim)), x=rng.standard_normal((n_active, n_dim)),
                    logdetj=rng.standard_normal(n_active), logl=rng.standard_normal(n_active),
                    logp=rng.standard_normal(n_active), logw=rng.standard_normal(n_active), blobs=None,
                    iter=1, logz=-1.0, calls=1000, steps=12, efficiency=0.9, ess=512.0, accept=0.2, beta=0.5)

    for _ in range(n_iter):
        s.particles.update(particles())
    s.current_particles = dict(particles(), weights=rng.random(n_iter * n_active))
    cp = s.current_particles
    mcmc = dict(kernel="preconditioned_pcn", u=cp["u"], x=cp["x"], theta=cp["u"], logdetj=cp["logdetj"],
                logdetj_flow=cp["logdetj"], logl=cp["logl"], logp=cp["logp"], blobs=None, sigma=np.float64(0.5),
                mu=np.zeros(n_dim), i=5, cnt=2, logp2_val=np.float64(-3.0), n_calls=np.int64(1280))
    return s, mcmc, particles


def old_style_save(s, path):
    """Whole-state dill pickle, as written by the previous save_state."""
    s.particles._logw_denominator()  # the cache was part of the pickled state
    state = dict(particles=s.particles, current=s.current_particles, prior=s.prior_samples,
                 flow={k: v.detach().cpu().numpy() for k, v in s.flow.flow.state_dict().items()})
    tmp = str(path) + ".temp"
    with open(tmp, "wb") as f:
        dill.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)


def main():
    print(f"pocomc imported from {Path(pocomc.__file__).parent}")

    print(f"{'n_dim':>5s} {'n_active':>8s} {'iters':>5s} | {'save (median)':>13s} {'max':>8s} "
          f"{'state MB':>8s} | {'old dill save':>13s} | {'overhead @2s/step':>17s}")
    for n_dim in (10, 50):
        for n_active in (256, 1024):
            for n_iter in (50, 200):
                out = Path(tempfile.mkdtemp())
                s, mcmc, particles = build(n_dim, n_active, n_iter, out)
                path = out / "pmc_latest.state"
                s._save_state(path, mcmc_state=mcmc, verbose=False)  # writes the full history once
                times = []
                for _ in range(9):
                    s.particles.update(particles())
                    t = time.perf_counter()
                    s._save_state(path, mcmc_state=mcmc, verbose=False)
                    times.append(time.perf_counter() - t)
                size = path.stat().st_size / 1e6
                t = time.perf_counter()
                old_style_save(s, out / "old.state")
                old = time.perf_counter() - t
                med = float(np.median(times))
                print(f"{n_dim:5d} {n_active:8d} {n_iter:5d} | {med * 1e3:11.1f}ms "
                      f"{max(times) * 1e3:6.1f}ms {size:8.2f} | {old * 1e3:11.1f}ms | {100 * med / 2.0:16.2f}%")
                shutil.rmtree(out)


if __name__ == "__main__":
    main()
