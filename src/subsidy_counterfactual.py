"""Replacement-subsidy counterfactual with agent re-learning.

Pre-registered design: paper/prereg_subsidy_cf.md. True RC: 10 -> 5 (50%
proportional subsidy). Agents adapt under the STABLE-COGNITIVE-PRIMITIVE
rule (same categories / attention / clamp applied to the new optimal
policy). The econometrician predicts with pre-policy NFXP estimates at
R-hat_C / 2. Failure metrics are compared with the PRE-policy conformity
audit — the note's final exhibit.

Outputs: results/subsidy_cf_raw.md (+ console).
"""

import csv
from pathlib import Path

import numpy as np

from mdp2d import RustMDP2D
from train_agents2d import agent_policy
from experiment2d_zoo import train_on_target
from zoo_v2 import CSV_PATH, zoo_v2_targets

RESULTS = Path(__file__).resolve().parents[1] / "results"
SUBSIDY = 0.5  # proportional replacement-cost subsidy


def ergodic_and_targets(truth: RustMDP2D):
    sol = truth.solve()
    ref = sol.simulate(n_buses=2000, n_periods=200,
                       rng=np.random.default_rng(9))
    _, counts = ref.empirical_ccp(truth.n_states)
    ergodic = counts / counts.sum()
    grid = ergodic.reshape(truth.n_mileage, truth.n_cost)
    return sol, ergodic, zoo_v2_targets(sol, grid)


def replacement_rate(mdp: RustMDP2D, policy_flat: np.ndarray) -> float:
    """Long-run replacement rate under `policy_flat` in environment `mdp`."""
    sol = mdp.solve()
    panel = sol.simulate(n_buses=1000, n_periods=300,
                         rng=np.random.default_rng(77), policy=policy_flat)
    return float(panel.choices.mean())


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    pre = RustMDP2D()                       # RC = 10
    post = RustMDP2D(rc=pre.rc * SUBSIDY)   # RC = 5

    print("solving pre and post environments, building adapted targets...")
    _, _, pre_targets = ergodic_and_targets(pre)
    _, post_ergodic, post_targets = ergodic_and_targets(post)

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        est = {r["name"]: (float(r["theta1"]), float(r["theta2"]),
                           float(r["rc"])) for r in csv.DictReader(f)}

    # Pre-policy conformity scores (from the c_conformity run, same seeds).
    conformity_raw = (RESULTS / "c_conformity_raw.md").read_text(
        encoding="utf-8")
    conf = {}
    for line in conformity_raw.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 7 and parts[1] in est:
            conf[parts[1]] = float(parts[5])  # conformity column

    rows = []
    for name, (family, target_post) in post_targets.items():
        print(f"{name} [{family}]: adapting (retrain on post-subsidy target)...")
        agent_post = train_on_target(target_post, post)
        pol_adapted = agent_policy(agent_post, post)

        t1, t2, rc_hat = est[name]
        mdp_hat_post = RustMDP2D(n_mileage=post.n_mileage,
                                 n_cost=post.n_cost, theta1=t1, theta2=t2,
                                 rc=rc_hat * SUBSIDY, beta=post.beta,
                                 mileage_probs=post.mileage_probs,
                                 cost_persist=post.cost_persist)
        pol_pred = mdp_hat_post.solve().ccp_replace

        rmse_w = float(np.sqrt(np.sum(post_ergodic
                                      * (pol_pred - pol_adapted) ** 2)
                               / post_ergodic.sum()))
        rmse_grid = float(np.sqrt(np.mean((pol_pred - pol_adapted) ** 2)))
        rate_pred = replacement_rate(post, pol_pred)
        rate_actual = replacement_rate(post, pol_adapted)
        rate_err = abs(rate_pred - rate_actual)

        rows.append(dict(name=name, family=family, rmse_w=rmse_w,
                         rmse_grid=rmse_grid, rate_pred=rate_pred,
                         rate_actual=rate_actual, rate_err=rate_err,
                         conformity=conf.get(name, np.nan)))
        print(f"  RMSE(w)={rmse_w:.4f}, grid={rmse_grid:.4f}, "
              f"rate pred/actual={rate_pred:.4f}/{rate_actual:.4f}, "
              f"conformity={conf.get(name, float('nan')):.3f}")

    c = np.array([r["conformity"] for r in rows])
    rmse = np.array([r["rmse_grid"] for r in rows])
    rerr = np.array([r["rate_err"] for r in rows])
    print(f"n={len(rows)}: Spearman(conformity, grid RMSE)={spearman(c, rmse):.2f}, "
          f"(conformity, rate error)={spearman(c, rerr):.2f}")

    table = "\n".join(
        f"| {r['name']} | {r['family']} | {r['rmse_w']:.4f} | "
        f"{r['rmse_grid']:.4f} | {r['rate_pred']:.4f} | "
        f"{r['rate_actual']:.4f} | {r['rate_err']:.4f} | "
        f"{r['conformity']:.3f} |" for r in rows)
    summary = f"""# Subsidy counterfactual with re-learning (raw)

50% proportional RC subsidy (10 -> 5); stable-cognitive-primitive adaptation;
prediction from pre-policy NFXP estimates at R-hat_C/2.
Spearman(pre-policy conformity, grid RMSE) = {spearman(c, rmse):.2f};
(conformity, replacement-rate error) = {spearman(c, rerr):.2f}. n = {len(rows)}.

| agent | family | RMSE (ergodic-w) | RMSE (grid) | rate pred | rate actual | rate err | pre conformity |
|---|---|---|---|---|---|---|---|
{table}
"""
    (RESULTS / "subsidy_cf_raw.md").write_text(summary, encoding="utf-8")
    print(f"summary -> {RESULTS / 'subsidy_cf_raw.md'}")


if __name__ == "__main__":
    main()
