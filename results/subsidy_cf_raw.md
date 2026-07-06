# Subsidy counterfactual with re-learning (raw)

50% proportional RC subsidy (10 -> 5); stable-cognitive-primitive adaptation;
prediction from pre-policy NFXP estimates at R-hat_C/2.
Spearman(pre-policy conformity, grid RMSE) = 0.80;
(conformity, replacement-rate error) = 0.63. n = 13.

| agent | family | RMSE (ergodic-w) | RMSE (grid) | rate pred | rate actual | rate err | pre conformity |
|---|---|---|---|---|---|---|---|
| clone | optimal | 0.0056 | 0.0126 | 0.0859 | 0.0839 | 0.0020 | 0.452 |
| cat4x1 | Fryer-Jackson categories | 0.0695 | 0.2515 | 0.1341 | 0.0940 | 0.0400 | 1.000 |
| cat6x1 | Fryer-Jackson categories | 0.0513 | 0.1302 | 0.1059 | 0.0915 | 0.0144 | 0.801 |
| cat6x2 | Fryer-Jackson categories | 0.0537 | 0.1223 | 0.1039 | 0.0922 | 0.0117 | 1.000 |
| cat8x2 | Fryer-Jackson categories | 0.0431 | 0.0955 | 0.0998 | 0.0881 | 0.0116 | 0.973 |
| cat12x3 | Fryer-Jackson categories | 0.0133 | 0.0313 | 0.0891 | 0.0864 | 0.0028 | 0.467 |
| cat16x5 | Fryer-Jackson categories | 0.0091 | 0.0198 | 0.0872 | 0.0850 | 0.0021 | 0.449 |
| sparse_m0.0 | Gabaix sparse-max | 0.0032 | 0.0037 | 0.0844 | 0.0842 | 0.0002 | 0.464 |
| sparse_m0.5 | Gabaix sparse-max | 0.0026 | 0.0071 | 0.0841 | 0.0842 | 0.0001 | 0.416 |
| ri_lam0.25 | Matejka-McKay RI | 0.0038 | 0.0075 | 0.0857 | 0.0844 | 0.0013 | 0.338 |
| ri_lam0.75 | Matejka-McKay RI | 0.0038 | 0.0044 | 0.0847 | 0.0841 | 0.0006 | 0.439 |
| asym_low | salience asym | 0.0083 | 0.0159 | 0.0877 | 0.0875 | 0.0002 | 0.700 |
| asym_high | salience asym | 0.0082 | 0.0191 | 0.0812 | 0.0806 | 0.0006 | 0.525 |
