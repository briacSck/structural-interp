# C-channel conformity audit (raw)

Conformity score = max(normalized cv_x, normalized cv_c).
Spearman with CF RMSE: conformity 0.87,
x-only 0.53, c-only 0.18 (n=13).

| agent | family | cv_x | cv_c | conformity | CF RMSE |
|---|---|---|---|---|---|
| clone | optimal | 0.626 | 0.605 | 0.452 | 0.0081 |
| cat4x1 | Fryer-Jackson categories | 1.148 | 0.000 | 1.000 | 0.2182 |
| cat6x1 | Fryer-Jackson categories | 1.025 | 0.000 | 0.801 | 0.1182 |
| cat6x2 | Fryer-Jackson categories | 0.815 | 1.338 | 1.000 | 0.1069 |
| cat8x2 | Fryer-Jackson categories | 0.769 | 1.301 | 0.973 | 0.0755 |
| cat12x3 | Fryer-Jackson categories | 0.664 | 0.625 | 0.467 | 0.0233 |
| cat16x5 | Fryer-Jackson categories | 0.528 | 0.601 | 0.449 | 0.0169 |
| sparse_m0.0 | Gabaix sparse-max | 0.815 | 0.000 | 0.464 | 0.0048 |
| sparse_m0.5 | Gabaix sparse-max | 0.637 | 0.557 | 0.416 | 0.0079 |
| ri_lam0.25 | Matejka-McKay RI | 0.738 | 0.000 | 0.338 | 0.0080 |
| ri_lam0.75 | Matejka-McKay RI | 0.663 | 0.587 | 0.439 | 0.0054 |
| asym_low | salience asym | 0.715 | 0.936 | 0.700 | 0.0264 |
| asym_high | salience asym | 0.697 | 0.702 | 0.525 | 0.0230 |
