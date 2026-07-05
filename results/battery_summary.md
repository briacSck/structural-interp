# Audit battery — do two channel audits jointly order CF failure?

| agent | c-use audit | x-dispersion (CV) | risk_c | risk_x | battery (max) | CF RMSE |
|---|---|---|---|---|---|---|
| clone | 0.2088 | 0.626 | 0.000 | 0.000 | 0.000 | 0.0106 |
| fine2d | 0.1597 | 0.664 | 0.235 | 0.149 | 0.235 | 0.0245 |
| coarse2d | 0.2074 | 0.815 | 0.006 | 0.751 | 0.751 | 0.1078 |
| blindbucket | 0.0457 | 0.878 | 0.781 | 1.000 | 1.000 | 0.1776 |
| asym | 0.0817 | 0.715 | 0.609 | 0.352 | 0.609 | 0.0268 |

Rank correlations with CF RMSE: battery 1.00; c-channel alone 0.70;
x-channel alone 1.00.

## Actual readings (run of 2026-07-06)

**Headline: the battery rank-orders counterfactual failure perfectly
(Spearman = 1.00 over the 5-agent zoo).** The c-channel alone does not
(0.70): it wrongly ranks coarse2d as safe (high c-use, second-worst failure).
Adding the x-dispersion channel fixes exactly that case — the multi-channel
logic works as designed.

**Honest caveats, for the note:**
1. n = 5 agents; a perfect Spearman over 5 points is encouraging, not
   conclusive. The note needs a larger zoo (10-20 agents, varied rules and
   seeds) before claiming a monotone audit->failure law.
2. In THIS zoo, x-dispersion alone also reaches 1.00 — because 3 of the 4
   out-of-class agents are x-bucketed, so the x-channel happens to proxy for
   general out-of-class-ness. The battery framing (worst channel) is the
   defensible general claim; single-channel sufficiency is a zoo artifact.
3. Deeper lesson for the theory section: a *smooth* regime-blind agent (the
   lambda=0 blend from experiment2d) fails LITTLE (RMSE 0.015) because the
   estimator correctly fits theta2 ~ 0 for it — c-blindness inside the model
   class is benign. What kills counterfactuals is internals *outside the
   model class*. Audits should therefore target model-class conformity of
   the mechanism, channel by channel — which is precisely what a battery of
   specification tests means (Kalouptsidi et al. resonance: which
   counterfactual fails depends on which channel is broken).

**Status of the paper's central claim:** demonstrated in pilot form — an
internal, causal, in-distribution audit battery predicts (rank-orders) which
agents' structural counterfactuals fail, while behavioral fit (2-3% gaps)
and probe decoding (R^2 = 1.0 everywhere) cannot.
