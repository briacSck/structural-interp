"""Pilot experiment: the paper's three core exhibits in one run.

Exhibit 1 — behavioral similarity: the clone (value-based) and the heuristic
            (bucket rule) are hard to tell apart on the ergodic state
            distribution.
Exhibit 2 — probe contrast: linear probes for the true value function V(x)
            separate the two agents' internals.
Exhibit 3 — structural estimation + off-distribution transport: NFXP fits
            both agents' panels, but its predicted policy transports to
            rarely-visited states only for the agent that internalized the
            value structure.

Outputs: results/pilot_exhibits.png and results/pilot_summary.md.

Deferred by design (do NOT bolt on silently — research-design decisions):
activation patching, Hotz-Miller estimator, retraining counterfactuals.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from estimate import estimate_nfxp
from mdp import RustMDP
from probes import probe_agent_for_value
from train_agents import (MLPAgent, simulate_agent, state_features,
                          train_clone, train_heuristic)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def agent_ccp(agent: MLPAgent, n_states: int) -> np.ndarray:
    feats = torch.tensor(state_features(np.arange(n_states), n_states),
                         dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(agent(feats)).numpy()


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    truth = RustMDP()
    sol = truth.solve()
    k = truth.n_states

    # Ergodic state distribution (visit frequencies under the optimal policy).
    ref_panel = sol.simulate(n_buses=2000, n_periods=200,
                             rng=np.random.default_rng(123))
    _, counts = ref_panel.empirical_ccp(k)
    ergodic = counts / counts.sum()
    in_dist = ergodic > 1e-3          # states an econometrician actually sees
    off_dist = ~in_dist

    print("training agents...")
    clone = train_clone(sol)
    heuristic = train_heuristic(sol)
    ccp_clone = agent_ccp(clone, k)
    ccp_heur = agent_ccp(heuristic, k)

    # ---- Exhibit 1: behavioral similarity -------------------------------
    weighted_gap = float(np.sum(ergodic * np.abs(ccp_clone - ccp_heur)))
    panel_clone = simulate_agent(clone, truth, n_buses=500, n_periods=200, seed=11)
    panel_heur = simulate_agent(heuristic, truth, n_buses=500, n_periods=200, seed=11)
    rate_clone = float(panel_clone.choices.mean())
    rate_heur = float(panel_heur.choices.mean())
    print(f"[1] ergodic-weighted |CCP gap| = {weighted_gap:.4f}; "
          f"replacement rates: clone {rate_clone:.4f}, heuristic {rate_heur:.4f}")

    # ---- Exhibit 2: probe contrast --------------------------------------
    probes_clone = probe_agent_for_value(clone, sol)
    probes_heur = probe_agent_for_value(heuristic, sol)
    print("[2] held-out R^2 for V(x):",
          {f"clone L{r['layer']}": round(r["r2_test"], 4) for r in probes_clone},
          {f"heur L{r['layer']}": round(r["r2_test"], 4) for r in probes_heur})

    # ---- Exhibit 3: structural estimation + off-distribution transport ---
    print("[3] running NFXP on both panels (takes ~1 min)...")
    est_clone = estimate_nfxp(panel_clone, truth)
    est_heur = estimate_nfxp(panel_heur, truth)

    def implied_ccp(est):
        return RustMDP(n_states=k, theta1=est.theta1, rc=est.rc,
                       beta=truth.beta, trans_probs=truth.trans_probs
                       ).solve().ccp_replace

    pred_clone, pred_heur = implied_ccp(est_clone), implied_ccp(est_heur)

    def rmse(pred, actual, mask):
        return float(np.sqrt(np.mean((pred[mask] - actual[mask]) ** 2)))

    transport = {
        "clone_in": rmse(pred_clone, ccp_clone, in_dist),
        "clone_off": rmse(pred_clone, ccp_clone, off_dist),
        "heur_in": rmse(pred_heur, ccp_heur, in_dist),
        "heur_off": rmse(pred_heur, ccp_heur, off_dist),
    }
    print(f"    clone: theta1={est_clone.theta1:.4f}, rc={est_clone.rc:.3f} | "
          f"heuristic: theta1={est_heur.theta1:.4f}, rc={est_heur.rc:.3f} "
          f"(truth 0.05, 10.0)")
    print(f"    structural-prediction RMSE in/off distribution: "
          f"clone {transport['clone_in']:.4f}/{transport['clone_off']:.4f}, "
          f"heuristic {transport['heur_in']:.4f}/{transport['heur_off']:.4f}")

    # ---- Figure ----------------------------------------------------------
    x = np.arange(k)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(x, sol.ccp_replace, "k--", label="optimal policy", lw=1)
    ax.plot(x, ccp_clone, label="clone", lw=2)
    ax.plot(x, ccp_heur, label="heuristic", lw=2)
    ax2 = ax.twinx()
    ax2.fill_between(x, ergodic, alpha=0.15, color="gray")
    ax2.set_ylabel("ergodic state density (shaded)")
    ax.set_title("Exhibit 1: behaviorally similar where data lives")
    ax.set_xlabel("mileage bin"); ax.set_ylabel("P(replace | x)"); ax.legend()

    ax = axes[0, 1]
    layers = [r["layer"] for r in probes_clone]
    width = 0.35
    ax.bar(np.array(layers) - width / 2, [r["r2_test"] for r in probes_clone],
           width, label="clone")
    ax.bar(np.array(layers) + width / 2, [r["r2_test"] for r in probes_heur],
           width, label="heuristic")
    ax.set_title("Exhibit 2: probing hidden layers for V(x)")
    ax.set_xlabel("hidden layer"); ax.set_ylabel("held-out $R^2$")
    ax.set_xticks(layers); ax.set_ylim(0, 1.05); ax.legend()

    ax = axes[1, 0]
    ax.plot(x, ccp_clone, label="clone actual", lw=2)
    ax.plot(x, pred_clone, "--", label="structural prediction (clone $\\hat\\theta$)")
    ax.plot(x, ccp_heur, label="heuristic actual", lw=2)
    ax.plot(x, pred_heur, "--", label="structural prediction (heur $\\hat\\theta$)")
    first_off = int(np.argmax(off_dist)) if off_dist.any() else k
    ax.axvspan(first_off, k - 1, alpha=0.1, color="red",
               label="off-distribution")
    ax.set_title("Exhibit 3: transport of the estimated model")
    ax.set_xlabel("mileage bin"); ax.set_ylabel("P(replace | x)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        "Structural estimates (truth: theta1=0.05, RC=10.0)",
        f"  clone:      theta1={est_clone.theta1:.4f},  RC={est_clone.rc:.3f}",
        f"  heuristic:  theta1={est_heur.theta1:.4f},  RC={est_heur.rc:.3f}",
        "",
        "Prediction RMSE (in-dist / off-dist)",
        f"  clone:      {transport['clone_in']:.4f} / {transport['clone_off']:.4f}",
        f"  heuristic:  {transport['heur_in']:.4f} / {transport['heur_off']:.4f}",
        "",
        f"Ergodic-weighted behavioral gap: {weighted_gap:.4f}",
        f"Replacement rates: clone {rate_clone:.4f}, heur {rate_heur:.4f}",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), family="monospace", fontsize=10,
            va="top", transform=ax.transAxes)

    fig.tight_layout()
    fig_path = RESULTS / "pilot_exhibits.png"
    fig.savefig(fig_path, dpi=150)
    print(f"figure -> {fig_path}")

    # ---- Summary ---------------------------------------------------------
    summary = f"""# Pilot exhibits — run summary

Truth: theta1 = {truth.theta1}, RC = {truth.rc}, beta = {truth.beta}, {k} states.

## Exhibit 1 — behavioral similarity
- Ergodic-weighted mean |CCP(clone) - CCP(heuristic)|: **{weighted_gap:.4f}**
- Simulated replacement rates: clone {rate_clone:.4f}, heuristic {rate_heur:.4f}

## Exhibit 2 — linear probes for V(x), held-out R^2
| layer | clone | heuristic |
|---|---|---|
""" + "\n".join(
        f"| {rc_['layer']} | {rc_['r2_test']:.4f} | {rh['r2_test']:.4f} |"
        for rc_, rh in zip(probes_clone, probes_heur)) + f"""

## Exhibit 3 — NFXP estimates and transport
| agent | theta1_hat | RC_hat | RMSE in-dist | RMSE off-dist |
|---|---|---|---|---|
| clone | {est_clone.theta1:.4f} | {est_clone.rc:.3f} | {transport['clone_in']:.4f} | {transport['clone_off']:.4f} |
| heuristic | {est_heur.theta1:.4f} | {est_heur.rc:.3f} | {transport['heur_in']:.4f} | {transport['heur_off']:.4f} |

## Caveats / next design decisions (week 4-5)
- Probe R^2 on a 1-D state is weak evidence by itself (any monotone encoding
  decodes well); the discriminating tests are activation patching and/or a
  2-D state extension.
- The "off-distribution transport" exhibit is a transportability check, not a
  full counterfactual; the retraining counterfactual (agents re-adapting to a
  changed RC) is a research-design decision to make deliberately.
- Hotz-Miller estimator still to add alongside NFXP.
"""
    summary_path = RESULTS / "pilot_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
