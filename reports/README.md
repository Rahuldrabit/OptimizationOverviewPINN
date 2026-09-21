# Reports — Old vs. Real

This folder consolidates every report the project has generated, in one place,
so you can compare them directly. Three subfolders:

## `real_run/` — start here

The one real, PyTorch-trained run. All 10 algorithms trained actual PINNs on
ODE, Heat, Burgers, and Wave (2 seeds, reduced step budget to fit the
session — see caveats below). Read `REAL_RESULTS_SUMMARY.md` first; it's the
honest analysis, including a Friedman significance test on the ranking. Also
useful is a copy of the auto-generated `HPO_INVESTIGATION_REPORT_REAL.md` —
its sections 1 and 5 contain hardcoded template narrative text unrelated to
the actual data. The ranking table, per-benchmark hyperparameter tables, and
the four plots in `plots/` are computed from real results.

**Scale caveat**: this run used 2 seeds, 60 training steps per candidate, and
halved population/generation counts to fit within one working session. It's
a real but small-scale pilot, not a publication-scale result — see the
"To turn this into a top-journal-ready result" section in
`REAL_RESULTS_SUMMARY.md` for what to scale up before submitting anywhere.

## `old_broken_run/` — kept for reference/comparison only

The original report your project had generated before this session. Every
algorithm shows identical hyperparameters and error on Heat, Burgers, and
Wave, because `train_pinn_placeholder()` was silently returning a fixed
`val_rel_l2 = 0.05` for those three benchmarks regardless of what was being
tuned — no PDE was ever actually trained. ODE's numbers came from a synthetic
NumPy formula standing in for real training because PyTorch wasn't
installed. **Do not cite anything in this folder** — it's kept only so you
can see exactly what was wrong and confirm the fix. Full diagnosis is in
`real_run/REAL_RESULTS_SUMMARY.md`.

## `speed_benchmark_ode_only/` — not yet re-verified

The convergence-speed report (ODE only, "which optimizer converges fastest").
This was generated before PyTorch was installed, so it likely also ran on
the synthetic NumPy fallback rather than real training — I have not
re-verified or regenerated this one in this session. Treat it the same as
`old_broken_run/` until it's re-run with the real trainer (I can do this on
request; it's a smaller job than the full grid since it's ODE-only).

---

Source of truth for what changed and why: `real_run/REAL_RESULTS_SUMMARY.md`.
