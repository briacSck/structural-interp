"""C-channel conformity audit — closing the "one test per channel" story.

zoo_v2 left one gap: the salience-asymmetric agents (asym_low/high) are
out-of-class in the c-dimension only (their regime response is clamped below
the median — piecewise in c), and the x-dispersion audit only partially sees
them. This audit is the c-analog of x-dispersion:

  For in-distribution mileage bins, patch the readout-component of the
  activation shift between CONSECUTIVE regimes (x, c) -> (x, c+1) and record
  |dP|. A mechanism that responds to regimes the way the model class allows
  (smooth, monotone scaling) spreads sensitivity evenly across regime steps;
  a clamped/categorical mechanism is flat on some steps and jumps on others.

  Statistic: cv_c = std / (mean + floor). Uniformly-zero sensitivity (a
  regime-blind but smooth agent, e.g. sparse_m0.0) gives cv ~ 0 — correctly
  read as CONFORMING, since a theta2 = 0 model fits it. The floor keeps the
  ratio stable at near-zero means.

Conformity score per agent = max(normalized cv_x, normalized cv_c).
Reuses CF RMSE from results/zoo_v2_rows.csv (same seeds -> same agents),
so no NFXP is re-run. Outputs: results/c_conformity_raw.md.
"""

import csv
from pathlib import Path

import numpy as np
import torch

from audit_battery import x_dispersion_audit
from mdp2d import RustMDP2D
from probes import _forward_from, readout_direction
from train_agents2d import all_state_features
from experiment2d_zoo import train_on_target
from zoo_v2 import CSV_PATH, zoo_v2_targets

RESULTS = Path(__file__).resolve().parents[1] / "results"


def c_dispersion_audit(agent, mdp: RustMDP2D, layer: int = 0,
                       floor: float = 1e-3) -> dict:
    """CV of local causal regime-sensitivity along the readout direction."""
    d = readout_direction(agent, layer)
    u = torch.tensor(d / np.linalg.norm(d), dtype=torch.float32)
    feats = all_state_features(mdp)
    sens = []
    with torch.no_grad():
        acts = agent.hidden_activations(feats)[layer]
        for x in range(5, 45, 5):
            for c in range(mdp.n_cost - 1):
                a_lo = acts[x * mdp.n_cost + c]
                a_hi = acts[x * mdp.n_cost + c + 1]
                patched = a_lo + torch.dot(a_hi - a_lo, u) * u
                p_lo = torch.sigmoid(_forward_from(agent, layer, a_lo)).item()
                p_pa = torch.sigmoid(_forward_from(agent, layer, patched)).item()
                sens.append(abs(p_pa - p_lo))
    sens = np.array(sens)
    mean = float(sens.mean())
    # Gate: a channel the agent is uniformly insensitive to is CONFORMING
    # (some parameter value fits it exactly); its CV is pure noise.
    cv = float(sens.std() / (mean + floor)) if mean >= 0.01 else 0.0
    return {"c_dispersion": cv, "c_mean_sens": mean}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    truth = RustMDP2D()
    sol = truth.solve()
    ref = sol.simulate(n_buses=2000, n_periods=200,
                       rng=np.random.default_rng(9))
    _, counts = ref.empirical_ccp(truth.n_states)
    ergodic_grid = (counts / counts.sum()).reshape(truth.n_mileage,
                                                   truth.n_cost)

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        cf_by_name = {r["name"]: float(r["cf_rmse"])
                      for r in csv.DictReader(f)}

    rows = []
    for name, (family, target) in zoo_v2_targets(sol, ergodic_grid).items():
        print(f"{name}: training (same seed as zoo_v2)...")
        agent = train_on_target(target, truth)
        cv_x = x_dispersion_audit(agent, truth)["x_dispersion"]
        out = c_dispersion_audit(agent, truth)
        rows.append(dict(name=name, family=family, cv_x=cv_x,
                         cv_c=out["c_dispersion"],
                         c_mean=out["c_mean_sens"],
                         cf=cf_by_name[name]))
        print(f"  cv_x={cv_x:.3f}, cv_c={out['c_dispersion']:.3f}, "
              f"mean_c_sens={out['c_mean_sens']:.4f}, CF={cf_by_name[name]:.4f}")

    cv_x = np.array([r["cv_x"] for r in rows])
    cv_c = np.array([r["cv_c"] for r in rows])
    cf = np.array([r["cf"] for r in rows])
    nx = (cv_x - cv_x.min()) / (cv_x.max() - cv_x.min())
    nc = (cv_c - cv_c.min()) / (cv_c.max() - cv_c.min())
    conformity = np.maximum(nx, nc)
    print(f"n={len(rows)}: Spearman conformity={spearman(conformity, cf):.2f}, "
          f"x-only={spearman(nx, cf):.2f}, c-only={spearman(nc, cf):.2f}")

    table = "\n".join(
        f"| {r['name']} | {r['family']} | {r['cv_x']:.3f} | {r['cv_c']:.3f} | "
        f"{n_:.3f} | {r['cf']:.4f} |"
        for r, n_ in zip(rows, conformity))
    summary = f"""# C-channel conformity audit (raw)

Conformity score = max(normalized cv_x, normalized cv_c).
Spearman with CF RMSE: conformity {spearman(conformity, cf):.2f},
x-only {spearman(nx, cf):.2f}, c-only {spearman(nc, cf):.2f} (n={len(rows)}).

| agent | family | cv_x | cv_c | conformity | CF RMSE |
|---|---|---|---|---|---|
{table}
"""
    (RESULTS / "c_conformity_raw.md").write_text(summary, encoding="utf-8")
    print(f"summary -> {RESULTS / 'c_conformity_raw.md'}")


if __name__ == "__main__":
    main()
