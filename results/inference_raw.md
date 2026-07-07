# Inference on the headline rank correlations (raw)

Method: one-sided Monte-Carlo permutation test (100k permutations),
bootstrap 95% percentile CI (10k agent resamples), leave-one-out Spearman
range (composition stability).

| headline | rho = 0.87 | perm p | 95% CI | LOO range |
|---|---|---|---|---|
| conformity -> regime-shift CF | 0.87 | 0.0001 | [0.49, 0.97] | [0.82, 0.92] |
| conformity -> subsidy CF | 0.80 | 0.0007 | [0.39, 0.92] | [0.75, 0.85] |

Note: the within-category "Spearman = 1.00" (n = 6) has exact one-sided
p = 1/6! = 0.0014 under random ranking, but at n = 6 is fragile to a single
swap — present as illustration only (review §4).
