"""Statistical inference for the headline Spearman correlations.

Feynman review must-do #1 (2026-07-07): the note reported rank correlations
with no uncertainty. For each headline pair this script computes:
- Monte-Carlo permutation p-value (one-sided, 100k permutations);
- bootstrap 95% percentile CI (10k resamples of agents);
- leave-one-out Spearman range (composition stability — review must-do #2's
  cheap variant: how much can one agent move the number?).

Pairs:
1. conformity audit -> regime-shift CF RMSE (n = 13; audits from
   c_conformity_raw.md, CF from zoo_v2_rows.csv)
2. conformity audit -> subsidy CF RMSE (n = 13; from subsidy_cf_raw.md)

Output: results/inference_raw.md.
"""

import csv
import re
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def infer(x: np.ndarray, y: np.ndarray, n_perm: int = 100_000,
          n_boot: int = 10_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    rho = spearman(x, y)
    perm = np.array([spearman(x, rng.permutation(y))
                     for _ in range(n_perm)])
    p_one_sided = float((np.sum(perm >= rho) + 1) / (n_perm + 1))
    n = len(x)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        boot.append(spearman(x[idx], y[idx]))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    loo = [spearman(np.delete(x, i), np.delete(y, i)) for i in range(n)]
    return dict(rho=rho, p=p_one_sided, ci=ci,
                loo_min=float(min(loo)), loo_max=float(max(loo)), n=n)


def load_conformity() -> dict:
    """Agent -> conformity score from c_conformity_raw.md's table."""
    text = (RESULTS / "c_conformity_raw.md").read_text(encoding="utf-8")
    conf = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        # | agent | family | cv_x | cv_c | conformity | CF RMSE |
        if len(parts) >= 7 and re.match(r"^[a-z]", parts[1] or ""):
            try:
                conf[parts[1]] = float(parts[5])
            except ValueError:
                continue
    return conf


def main() -> None:
    conf = load_conformity()

    with (RESULTS / "zoo_v2_rows.csv").open(newline="", encoding="utf-8") as f:
        cf_regime = {r["name"]: float(r["cf_rmse"]) for r in csv.DictReader(f)}

    subsidy = {}
    for line in (RESULTS / "subsidy_cf_raw.md").read_text(
            encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split("|")]
        # | agent | family | RMSE(w) | RMSE(grid) | ... |
        if len(parts) >= 9 and parts[1] in conf:
            subsidy[parts[1]] = float(parts[4])

    names = [n for n in conf if n in cf_regime and n in subsidy]
    x = np.array([conf[n] for n in names])
    y_regime = np.array([cf_regime[n] for n in names])
    y_subsidy = np.array([subsidy[n] for n in names])

    r1 = infer(x, y_regime)
    r2 = infer(x, y_subsidy)

    def fmt(r):
        return (f"rho = {r['rho']:.2f}, permutation p = {r['p']:.4f}, "
                f"bootstrap 95% CI [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}], "
                f"leave-one-out range [{r['loo_min']:.2f}, "
                f"{r['loo_max']:.2f}], n = {r['n']}")

    print("conformity -> regime-shift CF:", fmt(r1))
    print("conformity -> subsidy CF:     ", fmt(r2))

    summary = f"""# Inference on the headline rank correlations (raw)

Method: one-sided Monte-Carlo permutation test (100k permutations),
bootstrap 95% percentile CI (10k agent resamples), leave-one-out Spearman
range (composition stability).

| headline | {fmt(r1).split(',')[0]} | perm p | 95% CI | LOO range |
|---|---|---|---|---|
| conformity -> regime-shift CF | {r1['rho']:.2f} | {r1['p']:.4f} | [{r1['ci'][0]:.2f}, {r1['ci'][1]:.2f}] | [{r1['loo_min']:.2f}, {r1['loo_max']:.2f}] |
| conformity -> subsidy CF | {r2['rho']:.2f} | {r2['p']:.4f} | [{r2['ci'][0]:.2f}, {r2['ci'][1]:.2f}] | [{r2['loo_min']:.2f}, {r2['loo_max']:.2f}] |

Note: the within-category "Spearman = 1.00" (n = 6) has exact one-sided
p = 1/6! = 0.0014 under random ranking, but at n = 6 is fragile to a single
swap — present as illustration only (review §4).
"""
    (RESULTS / "inference_raw.md").write_text(summary, encoding="utf-8")
    print(f"summary -> {RESULTS / 'inference_raw.md'}")


if __name__ == "__main__":
    main()
