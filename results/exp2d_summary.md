# 2-D experiment — internal audit vs counterfactual validity

Truth: theta1 = 0.05, theta2 = 0.8, rc = 10.0;
60 mileage bins x 5 cost regimes.
Counterfactual: permanent shift to the worst cost regime (c = max);
econometrician predicts from NFXP estimates, truth = agent's actual policy.

| lambda | R2 decode c | causal audit | theta1_hat | theta2_hat | RC_hat | CF RMSE |
|---|---|---|---|---|---|---|
| 0.00 | 1.000 | 0.0264 | 0.0702 | -0.003 | 9.86 | 0.0154 |
| 0.25 | 1.000 | 0.0045 | 0.0630 | 0.172 | 9.70 | 0.0129 |
| 0.50 | 1.000 | 0.0047 | 0.0595 | 0.314 | 9.79 | 0.0078 |
| 0.75 | 1.000 | 0.0062 | 0.0546 | 0.550 | 9.85 | 0.0071 |
| 1.00 | 1.000 | 0.0021 | 0.0505 | 0.783 | 10.00 | 0.0018 |

## Actual readings (first run, 2026-07-06)

**What worked — the structural side is a clean result:**
- theta2_hat tracks lambda almost linearly (-0.003 -> 0.172 -> 0.314 ->
  0.550 -> 0.783 vs true 0.8): NFXP recovers the *degree* of cost-regime use.
  At lambda=0 it correctly reports "this agent does not respond to the
  regime" (theta2 ~ 0) and compensates via upward-biased theta1 (0.070).
- CF RMSE is monotone decreasing in lambda (0.0154 -> 0.0018).

**Design lesson 1 — counterfactual failures are small everywhere.** The
lambda-blend family stays (approximately) INSIDE the structural model class:
the lambda=0 agent is close to a theta2=0 agent, so the estimator can fit it
and its counterfactual barely fails (RMSE 0.015). To produce the paper's
"confidently wrong counterfactual" in 2-D, the heuristic agents must live
OUTSIDE the model class (bucket/threshold rules on (x, c), hysteresis,
asymmetric regime responses) — like the 1-D bucket heuristic did.

**Design lesson 2 — the c-probe-direction audit is broken, exactly as
Exhibit 4 predicted.** R2(c) = 1.000 for every agent (c sits in the inputs),
and patching the ridge-probe c-direction produces tiny, non-monotone shifts
(0.026 at lambda=0 vs 0.002 at lambda=1): ridge directions are causally idle,
so an audit built on them measures nothing. The audit must target the
causally-read subspace (readout-propagated directions, attribution patching,
or multi-direction subspace patching), not a decoding direction.

Both lessons are consistent with, and strengthen, the paper's central theme:
decodability-based tools mislead; causal-mechanism tools and out-of-class
agents are where the action is. Next iteration: out-of-class 2-D heuristics x
readout-aware audit.
