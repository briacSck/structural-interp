"""Microfounded zoo v2 — the paper's headline experiment.

Every agent is now an instance of a named bounded-rationality primitive
(citations in paper/positioning.md), not an ad hoc bucket rule:

  clone            full-information optimal policy (the in-class control)
  cat{nx}x{nc}     Fryer & Jackson (2008) categorical cognition: act on
                   ergodic-weighted category averages over an nx x nc grid
  sparse_m{m}      Gabaix (2014) sparse-max: perceive c' = m*c + (1-m)*cbar
                   (attention weight m on the cost regime), act optimally
                   on the perceived state
  ri_lam{l}        Matejka & McKay (2015) RI-logit attenuation toward the
                   regime-blind policy (Fosgerau et al. 2020 equivalence)
  asym_low/high    salience/default asymmetry: ignore regimes below (above)
                   the median by clamping to it

Audits (internal, causal, in-distribution states only):
  audit_c  amnesic LEACE erasure of the cost regime at layer 0 (leace.py)
  audit_x  dispersion (CV) of local causal mileage-sensitivity (audit_battery)
Battery = worst normalized channel. Outcome = regime-shift counterfactual
RMSE from NFXP estimates. Headline statistic: Spearman(battery, CF RMSE)
across the zoo.

Resumable: per-agent rows are appended to results/zoo_v2_rows.csv; agents
already present are skipped, so an interrupted run continues where it left
off. Summary/figure are rebuilt from the CSV at the end of any run.
Seeding: zlib.crc32 (process-stable).
"""

import csv
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from audit_battery import x_dispersion_audit
from estimate2d import estimate_nfxp_2d
from leace import amnesic_audit
from mdp2d import RustMDP2D
from train_agents2d import agent_policy, all_state_features
from experiment2d_zoo import bucket_average, train_on_target

RESULTS = Path(__file__).resolve().parents[1] / "results"
CSV_PATH = RESULTS / "zoo_v2_rows.csv"
FIELDS = ["name", "family", "beh_gap", "audit_c", "audit_x",
          "theta1", "theta2", "rc", "cf_rmse"]


def sparse_attention(grid: np.ndarray, m: float) -> np.ndarray:
    """Gabaix sparse-max on the cost regime: act on c' = m*c + (1-m)*cbar."""
    kx, kc = grid.shape
    cbar = (kc - 1) / 2
    out = np.empty_like(grid)
    for c in range(kc):
        cp = m * c + (1 - m) * cbar
        lo, hi = int(np.floor(cp)), min(int(np.floor(cp)) + 1, kc - 1)
        w = cp - np.floor(cp)
        out[:, c] = (1 - w) * grid[:, lo] + w * grid[:, hi]
    return out


def zoo_v2_targets(solution, ergodic_grid: np.ndarray) -> dict:
    grid = solution.ccp_grid()
    kc = grid.shape[1]
    blind = grid.mean(axis=1, keepdims=True)
    targets = {"clone": ("optimal", grid)}
    for nx, nc in ((4, 1), (6, 1), (6, 2), (8, 2), (12, 3), (16, 5)):
        targets[f"cat{nx}x{nc}"] = (
            "Fryer-Jackson categories",
            bucket_average(grid, ergodic_grid, nx, nc))
    for m in (0.0, 0.5):
        targets[f"sparse_m{m}"] = ("Gabaix sparse-max",
                                   sparse_attention(grid, m))
    for lam in (0.25, 0.75):
        targets[f"ri_lam{lam}"] = ("Matejka-McKay RI",
                                   lam * grid + (1 - lam) * blind)
    targets["asym_low"] = ("salience asym",
                           grid[:, np.maximum(np.arange(kc), kc // 2)])
    targets["asym_high"] = ("salience asym",
                            grid[:, np.minimum(np.arange(kc), kc // 2)])
    return targets


def load_done() -> dict:
    if not CSV_PATH.exists():
        return {}
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def append_row(row: dict) -> None:
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    truth = RustMDP2D()
    sol = truth.solve()
    ref = sol.simulate(n_buses=2000, n_periods=200,
                       rng=np.random.default_rng(9))
    _, counts = ref.empirical_ccp(truth.n_states)
    ergodic = counts / counts.sum()
    ergodic_grid = ergodic.reshape(truth.n_mileage, truth.n_cost)
    in_dist = ergodic > 1e-4
    _, c_grid = truth.state_grid()
    feats = all_state_features(truth)

    done = load_done()
    for name, (family, target) in zoo_v2_targets(sol, ergodic_grid).items():
        if name in done:
            print(f"{name}: already done, skipping")
            continue
        print(f"{name} [{family}]: training...")
        agent = train_on_target(target, truth)
        pol = agent_policy(agent, truth)
        pol_grid = pol.reshape(truth.n_mileage, truth.n_cost)
        beh_gap = float(np.sum(ergodic * np.abs(pol - sol.ccp_replace)))

        with torch.no_grad():
            acts = agent.hidden_activations(feats)[0].numpy()
        audit_c = amnesic_audit(agent, 0, acts, c_grid.astype(float), in_dist)
        audit_x = x_dispersion_audit(agent, truth)["x_dispersion"]

        panel = sol.simulate(n_buses=500, n_periods=200,
                             rng=np.random.default_rng(zlib.crc32(name.encode())),
                             policy=pol)
        print("  estimating NFXP...")
        est = estimate_nfxp_2d(panel, truth)
        mdp_hat = RustMDP2D(n_mileage=truth.n_mileage, n_cost=truth.n_cost,
                            theta1=est.theta1, theta2=est.theta2, rc=est.rc,
                            beta=truth.beta, mileage_probs=truth.mileage_probs,
                            cost_persist=truth.cost_persist)
        pred_worst = mdp_hat.solve().ccp_grid()[:, -1]
        cf_rmse = float(np.sqrt(np.mean((pred_worst - pol_grid[:, -1]) ** 2)))

        row = dict(name=name, family=family, beh_gap=f"{beh_gap:.5f}",
                   audit_c=f"{audit_c:.5f}", audit_x=f"{audit_x:.4f}",
                   theta1=f"{est.theta1:.5f}", theta2=f"{est.theta2:.4f}",
                   rc=f"{est.rc:.3f}", cf_rmse=f"{cf_rmse:.5f}")
        append_row(row)
        print(f"  audit_c={audit_c:.4f}, audit_x={audit_x:.3f}, "
              f"theta=({est.theta1:.4f},{est.theta2:.3f},{est.rc:.2f}), "
              f"CF-RMSE={cf_rmse:.4f}")

    build_outputs()


def build_outputs() -> None:
    rows = list(load_done().values())
    if len(rows) < 3:
        print("not enough rows yet for summary/figure")
        return
    audit_c = np.array([float(r["audit_c"]) for r in rows])
    audit_x = np.array([float(r["audit_x"]) for r in rows])
    cf = np.array([float(r["cf_rmse"]) for r in rows])
    risk_c = 1 - audit_c / audit_c.max()
    risk_x = (audit_x - audit_x.min()) / (audit_x.max() - audit_x.min())
    battery = np.maximum(risk_c, risk_x)

    def spearman(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    rho_b, rho_c, rho_x = (spearman(battery, cf), spearman(risk_c, cf),
                           spearman(risk_x, cf))
    print(f"n={len(rows)}: Spearman battery={rho_b:.2f}, "
          f"c-only={rho_c:.2f}, x-only={rho_x:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(battery, cf, s=90, zorder=3)
    for r, b, y in zip(rows, battery, cf):
        ax.annotate(r["name"], (b, y), textcoords="offset points",
                    xytext=(7, 4), fontsize=8)
    ax.set_xlabel("battery score (worst channel: LEACE c-audit, x-dispersion)")
    ax.set_ylabel("regime-shift counterfactual RMSE")
    ax.set_title(f"Microfounded zoo (n={len(rows)}): "
                 f"Spearman = {rho_b:.2f}")
    fig.tight_layout()
    fig.savefig(RESULTS / "zoo_v2.png", dpi=150)

    table = "\n".join(
        f"| {r['name']} | {r['family']} | {r['beh_gap']} | {r['audit_c']} | "
        f"{r['audit_x']} | {r['theta1']} | {r['theta2']} | {r['rc']} | "
        f"{r['cf_rmse']} |" for r in rows)
    summary = f"""# Microfounded zoo v2 — LEACE amnesic audit x battery (raw)

n = {len(rows)} agents. Spearman with CF RMSE: battery {rho_b:.2f},
c-channel alone {rho_c:.2f}, x-channel alone {rho_x:.2f}.

| agent | family | beh gap | LEACE c-audit | x-dispersion | theta1 | theta2 | RC | CF RMSE |
|---|---|---|---|---|---|---|---|---|
{table}
"""
    (RESULTS / "zoo_v2_summary_raw.md").write_text(summary, encoding="utf-8")
    print(f"summary -> {RESULTS / 'zoo_v2_summary_raw.md'}")


if __name__ == "__main__":
    main()
