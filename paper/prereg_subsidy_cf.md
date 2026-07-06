# Pre-registration — replacement-subsidy counterfactual with agent re-learning

**Date: 2026-07-07, written BEFORE running the experiment** (user-approved
design, session of 2026-07-06/07). Code: `src/subsidy_counterfactual.py`.

## Intervention
A **50% proportional replacement-cost subsidy**: true RC falls from 10.0 to
5.0. (Proportional rather than absolute so the econometrician's counterfactual
R̂C/2 stays positive even when R̂C is badly biased.)

## Adaptation rule (Q2 = stable cognitive primitive)
Each agent re-derives its policy in the post-subsidy environment while
keeping its cognitive constraint fixed: the clone re-clones the new optimal
policy; categorical agents re-average the new optimal CCPs over the SAME
partitions (with post-environment ergodic weights); sparse-max keeps its
attention m; RI keeps its λ; asym keeps its clamp. The cognitive map is
treated as the deep primitive. (Endogenous attention — the Lucas-for-attention
case — is deliberately excluded here and reserved for paper #2.)

## Econometrician's prediction
NFXP estimates (θ̂₁, θ̂₂, R̂C) from the PRE-policy panels (zoo_v2_rows.csv,
crc32 seeds) are held fixed; the predicted post-policy behavior is the model
solved at (θ̂₁, θ̂₂, R̂C/2).

## Metrics
1. Ergodic-weighted RMSE between predicted and adapted policies (post
   environment), plus unweighted full-grid RMSE.
2. **Regulator's scalar:** error on the post-subsidy replacement rate —
   |rate(model at θ̂, RC/2) − rate(adapted agent)| — i.e., how wrong the
   predicted policy elasticity is.
3. Rank correlation between the PRE-policy conformity audit
   (results/c_conformity_raw.md scores) and subsidy-CF failure.

## Pre-registered hypotheses
- **H1 (in-class robustness):** fixed-attention in-class agents (clone,
  sparse_m*, ri_lam*) pass the subsidy counterfactual: their policy RMSE and
  replacement-rate errors are small, comparable to the clone's.
- **H2 (out-of-class failure, audit-predicted):** categorical agents fail
  (larger errors, ordered by coarseness), and the PRE-policy conformity audit
  rank-predicts this failure (positive Spearman, expected similar to the
  0.87 found for the regime-shift CF).
- **H3 (deferred to paper #2):** with ENDOGENOUS attention, in-class agents
  would fail too and the pre-policy audit could not see it — conformity
  audits certify invariance only under mechanism-preserving interventions.
  Not tested here; stated as the method's scope boundary.

## Analysis commitments
- All 13 agents, same training seeds as zoo_v2 (torch seed 0, crc32 panel
  seeds). No agent excluded post hoc; any anomaly is reported, not dropped.
- Headline = Spearman(conformity, subsidy-CF RMSE) + the H1 vs H2 contrast.
