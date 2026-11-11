# pocoVibes: new features compared to pocoMC

- **Frequent, cheap checkpoints.** `run(save_interval=seconds)` saves the sampler state to
  `{output_label}_latest.state` whenever at least `save_interval` seconds have passed, checked after
  every MCMC step, warm-up batch and iteration. A crash loses at most the work since the last
  checkpoint (about one MCMC step, warm-up batch or flow training) instead of a whole iteration. `save_every` (every N iterations) still works.
- **Checkpoints inside the MCMC mutation step.** A run interrupted in the middle of an iteration
  resumes from the last completed MCMC step.
- **Exact resume.** A resumed run gives bit-for-bit the same results as an uninterrupted one
  (random number generator states, flow weights and all sampler state are restored).
- **Constant-cost state files.** State files are small NumPy (`.npz`) archives written atomically;
  past iterations go to an append-only history directory (`{output_label}_history`). The cost of a
  save no longer grows with the length of the run (about 20–30 ms per save, measured on macOS).
- **Correct evidence for likelihoods that are zero (or NaN) on part of the prior.**
- **NaN likelihoods are handled** (treated as zero likelihood) instead of crashing the run.
- **More accurate bounded-parameter transforms** (probit and logit) near the bounds.
- **Periodic parameters that work.** With `periodic=[...]`, the point at which the period of a
  parameter is cut open is moved away from the particles during the run, so a posterior mode that
  lies across the bounds (e.g. a phase near 0, which is the same as 2 pi) is sampled in one piece
  instead of as two separate modes. In pocoMC the option had no effect. The `reflective` option,
  which had no effect either, is removed.
- **Faster sampling overheads**: vectorized MCMC proposals, flow training, weight trimming and
  evidence bootstrap; incremental importance-weight computation; smaller Student-t geometry fits;
  cheaper gathering of past particles.
- **New tests and tools**: `tests/test_exact_resume.py` (crash-and-resume tests),
  `tests/check_fixes_6_to_10.py` (checks of the flow and evidence fixes) and
  `benchmarks/checkpoint_benchmark.py` (cost of saving a checkpoint).
