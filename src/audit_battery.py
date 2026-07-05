"""Audit battery: one specification test per mechanism channel.

The zoo run showed a single audit is not enough: the c-use audit flags
regime-blindness (blindbucket) but misses x-structure misspecification
(coarse2d fails with a HIGH c-use score). This experiment completes the
battery with an x-structure audit and asks whether the PAIR jointly orders
total counterfactual failure.

Audit 1 — c-use (from experiment2d_zoo): mean |dP| from patching the
  c-induced activation shift along the readout direction. Low = regime-blind.

Audit 2 — x-dispersion (new): local causal sensitivity to mileage. For
  adjacent in-distribution mileage bins (x -> x+2, median regime), patch the
  x-induced activation shift along the readout direction and record |dP|.
  A smooth value-based mechanism spreads sensitivity evenly; a lookup-table
  mechanism is flat inside buckets and spikes at boundaries. Statistic =
  coefficient of variation (std/mean) of the local sensitivities.
  High = lookup-like internals.

Battery verdict per agent: fails channel 1 if c-use below threshold; fails
channel 2 if x-dispersion above threshold. Prediction: total regime-shift
CF failure is high iff at least one channel fails, and the battery rank
(worst channel) orders CF RMSE better than either audit alone.

Outputs: results/battery_summary.md, results/battery.png.
"""

import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from estimate2d import estimate_nfxp_2d
from mdp2d import RustMDP2D
from probes import _forward_from, readout_direction
from train_agents2d import agent_policy, all_state_features
from experiment2d_zoo import c_shift_audit, train_on_target, zoo_targets

RESULTS = Path(__file__).resolve().parents[1] / "results"


def x_dispersion_audit(agent, mdp: RustMDP2D, layer: int = 0,
                       step: int = 2) -> dict:
    """CV of local causal mileage-sensitivity along the readout direction."""
    d = readout_direction(agent, layer)
    u = torch.tensor(d / np.linalg.norm(d), dtype=torch.float32)
    feats = all_state_features(mdp)
    c_mid = mdp.n_cost // 2
    sens = []
    with torch.no_grad():
        acts = agent.hidden_activations(feats)[layer]
        for x in range(4, 46 - step):
            a_lo = acts[x * mdp.n_cost + c_mid]
            a_hi = acts[(x + step) * mdp.n_cost + c_mid]
            patched = a_lo + torch.dot(a_hi - a_lo, u) * u
            p_clean = torch.sigmoid(_forward_from(agent, layer, a_lo)).item()
            p_patch = torch.sigmoid(_forward_from(agent, layer, patched)).item()
            sens.append(abs(p_patch - p_clean))
    sens = np.array(sens)
    mean = sens.mean()
    cv = float(sens.std() / mean) if mean > 1e-9 else np.inf
    return {"x_dispersion": cv, "x_mean_sens": float(mean)}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    truth = RustMDP2D()
    sol = truth.solve()
    ref = sol.simulate(n_buses=2000, n_periods=200,
                       rng=np.random.default_rng(9))
    _, counts = ref.empirical_ccp(truth.n_states)
    ergodic_grid = (counts / counts.sum()).reshape(truth.n_mileage,
                                                   truth.n_cost)

    rows = []
    for name, target in zoo_targets(sol, ergodic_grid).items():
        print(f"{name}: training...")
        agent = train_on_target(target, truth)  # same seeds => same zoo agents
        pol = agent_policy(agent, truth)
        pol_grid = pol.reshape(truth.n_mileage, truth.n_cost)

        audit_c = c_shift_audit(agent, truth, readout_direction(agent, 0), 0)
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

        rows.append(dict(name=name, audit_c=audit_c, audit_x=audit_x,
                         cf_rmse=cf_rmse))
        print(f"  audit_c={audit_c:.4f}, audit_x_cv={audit_x:.3f}, "
              f"CF-RMSE={cf_rmse:.4f}")

    # Battery score: worst normalized channel. Channel 1 risk = 1 - c_use
    # (normalized to the best agent); channel 2 risk = x-dispersion
    # (normalized). Total predicted risk = max of the two.
    c_vals = np.array([r["audit_c"] for r in rows])
    x_vals = np.array([r["audit_x"] for r in rows])
    risk_c = 1 - c_vals / c_vals.max()
    risk_x = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())
    for r, rc_, rx in zip(rows, risk_c, risk_x):
        r["risk_c"], r["risk_x"] = float(rc_), float(rx)
        r["battery"] = float(max(rc_, rx))

    battery = np.array([r["battery"] for r in rows])
    cf = np.array([r["cf_rmse"] for r in rows])
    rho = float(np.corrcoef(np.argsort(np.argsort(battery)),
                            np.argsort(np.argsort(cf)))[0, 1])  # Spearman
    rho_c = float(np.corrcoef(np.argsort(np.argsort(risk_c)),
                              np.argsort(np.argsort(cf)))[0, 1])
    rho_x = float(np.corrcoef(np.argsort(np.argsort(risk_x)),
                              np.argsort(np.argsort(cf)))[0, 1])
    print(f"Spearman(battery, CF) = {rho:.2f}; c-only {rho_c:.2f}; "
          f"x-only {rho_x:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    ax = axes[0]
    for r in rows:
        ax.scatter(r["risk_c"], r["risk_x"], s=120,
                   c=[[min(r["cf_rmse"] / cf.max(), 1), 0.2, 0.2]], zorder=3)
        ax.annotate(f"{r['name']}\nCF={r['cf_rmse']:.3f}",
                    (r["risk_c"], r["risk_x"]), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)
    ax.set_xlabel("channel-1 risk: 1 - normalized c-use audit")
    ax.set_ylabel("channel-2 risk: normalized x-dispersion audit")
    ax.set_title("Audit battery map (darker red = worse counterfactual)")

    ax = axes[1]
    ax.scatter(battery, cf, s=110, zorder=3)
    for r in rows:
        ax.annotate(r["name"], (r["battery"], r["cf_rmse"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("battery score (worst channel)")
    ax.set_ylabel("regime-shift counterfactual RMSE")
    ax.set_title(f"Battery vs failure (Spearman = {rho:.2f})")
    fig.tight_layout()
    fig.savefig(RESULTS / "battery.png", dpi=150)
    print(f"figure -> {RESULTS / 'battery.png'}")

    table = "\n".join(
        f"| {r['name']} | {r['audit_c']:.4f} | {r['audit_x']:.3f} | "
        f"{r['risk_c']:.3f} | {r['risk_x']:.3f} | {r['battery']:.3f} | "
        f"{r['cf_rmse']:.4f} |" for r in rows)
    summary = f"""# Audit battery — do two channel audits jointly order CF failure?

| agent | c-use audit | x-dispersion (CV) | risk_c | risk_x | battery (max) | CF RMSE |
|---|---|---|---|---|---|---|
{table}

Rank correlations with CF RMSE: battery {rho:.2f}; c-channel alone {rho_c:.2f};
x-channel alone {rho_x:.2f}.

(Annotate after inspection.)
"""
    # _raw suffix: never overwrite the hand-annotated battery_summary.md
    (RESULTS / "battery_summary_raw.md").write_text(summary, encoding="utf-8")
    print(f"summary -> {RESULTS / 'battery_summary_raw.md'}")


if __name__ == "__main__":
    main()
