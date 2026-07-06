# Subsidy counterfactual with re-learning — annotated vs pre-registration

Raw: `subsidy_cf_raw.md`. Pre-registration: `paper/prereg_subsidy_cf.md`
(committed 11fdd1a BEFORE the run). Design: 50% RC subsidy (10 → 5),
stable-cognitive-primitive adaptation, prediction from pre-policy NFXP at
R̂C/2.

## H1 — CONFIRMED: in-class agents pass a true Lucas-style test

The fixed-attention in-class agents (clone, sparse-max, RI) survive a policy
intervention WITH re-learning: grid RMSE 0.004–0.013, replacement-rate errors
≤ 0.002 (e.g. sparse_m0.0: predicted 0.0844 vs actual 0.0842). The as-if
parameters estimated pre-policy transport through the intervention because
the cognitive primitive (attention weight) is held invariant — exactly the
condition the pre-registration stated.

## H2 — CONFIRMED: categorical agents fail, and the PRE-policy audit saw it

Failures are large and ordered by coarseness (grid RMSE 0.252 → 0.020 from
cat4x1 to cat16x5). The regulator-legible number: for cat4x1 the model
predicts a post-subsidy replacement rate of 0.134 vs an actual 0.094 — the
predicted policy response is roughly TWICE the actual one. A cost-benefit
analysis of the subsidy calibrated on this structural model would be wrong
by that factor.

**Headline: Spearman(pre-policy conformity audit, subsidy-CF RMSE) = 0.80**
(replacement-rate error: 0.63) — an audit computed on in-distribution
activations BEFORE the policy change rank-predicts which agents' policy
predictions fail AFTER it.

## H3 — scope boundary (deferred to paper #2, as pre-registered)
Under endogenous attention the invariance of the cognitive primitive fails
by construction and the pre-policy audit cannot see it. Stated in the note
as the method's boundary: conformity audits certify invariance under
mechanism-preserving interventions.

## The note's complete empirical arc (final)
Two counterfactual types, one diagnostic:
- regime-shift transport (no adaptation): conformity audit Spearman 0.87;
- subsidy with re-learning (Lucas-style): conformity audit Spearman 0.80.
Both discriminate in-class bounded rationality (benign, absorbed into as-if
parameters) from out-of-class cognition (silently fatal), where behavioral
fit (2–3% gaps) and linear probes (R² = 1.0) see nothing.
