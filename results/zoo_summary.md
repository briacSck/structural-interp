# Out-of-class zoo — readout-aware audit vs counterfactual failure

Truth: theta1=0.05, theta2=0.8, rc=10.0.
Counterfactual: permanent worst regime (c=max), prediction from NFXP estimates
vs the agent's actual policy.

| agent | ergodic behavior gap | audit (readout) | audit (probe dir) | theta1_hat | theta2_hat | RC_hat | CF RMSE |
|---|---|---|---|---|---|---|---|
| clone | 0.0010 | 0.2088 | 0.0021 | 0.0500 | 0.873 | 10.13 | 0.0113 |
| fine2d | 0.0056 | 0.1597 | 0.0214 | 0.0417 | 0.721 | 8.53 | 0.0262 |
| coarse2d | 0.0238 | 0.2074 | 0.0973 | 0.0310 | 0.596 | 6.71 | 0.0963 |
| blindbucket | 0.0275 | 0.0457 | 0.0751 | 0.0308 | 0.006 | 5.88 | 0.1636 |
| asym | 0.0118 | 0.0817 | 0.0073 | 0.0629 | 0.302 | 9.69 | 0.0290 |

## Actual readings (run of 2026-07-06)

**The 2-D "confidently wrong" exhibit is restored.** Out-of-class rules break
the estimator the way the 1-D bucket agent did: coarse2d gets RC_hat = 6.71
and blindbucket RC_hat = 5.88 (truth 10.0), with regime-shift CF RMSE up to
0.164 — an order of magnitude above the clone (0.011) — despite ergodic
behavior gaps of only 2-3%.

**The readout audit works where decoding could not — for the channel it
measures.** Probe decoding of c gave R^2 = 1.0 for every agent (useless), and
the probe-direction audit is anti-informative (clone 0.002 < blindbucket
0.075). The readout audit cleanly separates c-using agents (0.16-0.21) from
the c-blind one (0.046), and the c-blind agent is exactly the worst CF
failure. Q1 of the lit review is answered positively: a causal, internal,
in-distribution audit discriminates where probing fails.

**But one audit is not enough — failure is multi-channel.** coarse2d scores
HIGH on the c-use audit (0.207, it genuinely responds to regimes across its
bucket boundaries) yet still fails badly (CF RMSE 0.096): its failure comes
from the out-of-class x-structure, not from regime-blindness. A single
mechanism audit predicts the failure component it targets, not total failure.

**Paper framing this suggests (Kalouptsidi et al. resonance):** interp audits
as a *battery of specification tests*, one per mechanism channel (c-use
audit, x-smoothness/value-curvature audit, ...), where each audit flags the
counterfactuals that stress its channel. Next iteration: add an x-structure
audit and show the pair jointly orders total CF failure.
