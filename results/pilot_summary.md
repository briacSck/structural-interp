# Pilot exhibits — run summary

Truth: theta1 = 0.05, RC = 10.0, beta = 0.95, 90 states.

## Exhibit 1 — behavioral similarity
- Ergodic-weighted mean |CCP(clone) - CCP(heuristic)|: **0.0209**
- Simulated replacement rates: clone 0.0373, heuristic 0.0418

## Exhibit 2 — linear probes for V(x), held-out R^2
| layer | clone | heuristic |
|---|---|---|
| 0 | 0.9881 | 0.9944 |
| 1 | 0.9779 | 0.9766 |

## Exhibit 3 — NFXP estimates and transport
| agent | theta1_hat | RC_hat | RMSE in-dist | RMSE off-dist |
|---|---|---|---|---|
| clone | 0.0480 | 9.752 | 0.0056 | 0.0123 |
| heuristic | 0.0173 | 5.178 | 0.0412 | 0.3219 |

## Caveats / next design decisions (week 4-5)
- Probe R^2 on a 1-D state is weak evidence by itself (any monotone encoding
  decodes well); the discriminating tests are activation patching and/or a
  2-D state extension.
- The "off-distribution transport" exhibit is a transportability check, not a
  full counterfactual; the retraining counterfactual (agents re-adapting to a
  changed RC) is a research-design decision to make deliberately.
- Hotz-Miller estimator still to add alongside NFXP.
