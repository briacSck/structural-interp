# C-channel conformity audit — annotated (run of 2026-07-06, n = 13)

Raw table: `c_conformity_raw.md`. Headline: **conformity score
(max of per-channel dispersion audits, with the insensitivity gate)
rank-orders counterfactual failure at Spearman 0.87** across the 13
microfounded agents — vs 0.53 for the x-channel alone and 0.18 for the
c-channel alone. The "one specification test per mechanism channel" story is
closed.

## What each piece contributes
- **c-dispersion audit** catches exactly what x-dispersion missed: the
  salience-asymmetric agents (asym_low cv_c = 0.94) and the coarse-in-c
  categorical agents (cat6x2 / cat8x2: cv_c = 1.34 / 1.30).
- **The insensitivity gate is conceptually load-bearing, not a hack:** a
  channel the agent is uniformly insensitive to is *conforming* — some
  parameter value (theta2 = 0) fits it exactly, so transport survives
  (Finding 1 of zoo_v2). Without the gate, near-zero mean sensitivity makes
  the CV pure noise and misranks the regime-blind sparse-max agent
  (Spearman drops to 0.69). The gate encodes the theory: dispersion only
  signals non-conformity where the agent actually responds.

## Residual misranking (honest)
sparse_m0.0 keeps an elevated cv_x (0.815) despite the lowest CF (0.0048) —
its x-sensitivity profile is genuinely uneven (an artifact of fitting the
column-averaged policy), which the audit reads as mild lookup-ness. This is
the main gap between 0.87 and 1.0; worth one design iteration (e.g.,
comparing dispersion to a smooth-fit baseline rather than across-agent
normalization) before the note's final figure.

## Status: the paper's empirical arc is complete
1. Behaviorally similar agents, different mechanisms (Exhibit 1).
2. Structural estimation: robust to in-class inattention (as-if parameters,
   Fosgerau equivalence), confidently wrong for out-of-class cognition
   (zoo_v2).
3. Probes useless (R^2 = 1 everywhere), probe directions causally idle
   (Exhibit 4 = amnesic-probing lesson).
4. **Causal conformity audits — dispersion of local causal sensitivity per
   channel, gated by insensitivity — predict counterfactual failure at
   Spearman 0.87 on 13 microfounded bounded-rationality agents.**
