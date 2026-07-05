"""Out-of-class agent zoo x readout-aware audit — the paper's target exhibit.

The lambda-blend family (experiment2d.py) stayed inside the structural model
class, so counterfactuals barely failed. This experiment builds agents whose
decision RULES are outside the model class, then asks whether an internal
causal audit computed on IN-DISTRIBUTION states predicts the regime-shift
counterfactual failure.

Zoo (all same MLP architecture and inputs, trained on different target rules):
  clone        optimal CCP (inside the model class; the control)
  fine2d       12 x-buckets x 3 c-buckets step rule (close to optimal, but
               piecewise-constant: out-of-class in fine structure)
  coarse2d     6 x-buckets x 2 c-buckets (coarser lookup table)
  blindbucket  5 x-buckets, ignores the cost regime entirely
  asym         clamps low regimes to the median: target(x,c) = CCP*(x, max(c,2))
               (responds to the regime only when it is bad — asymmetric rule)

Audits (internal, causal, in-distribution states only):
  audit_readout  patch the component of the c-induced activation shift that
                 lies along the network's readout direction; mean |dP(replace)|
  audit_probe    same but along the ridge c-probe direction (known broken —
                 kept as the contrast)

Counterfactual: permanent shift to the worst regime (c = max); econometrician
predicts from NFXP(theta1, theta2, rc); truth = agent's actual policy there.

Outputs: results/zoo_summary.md, results/zoo_audit_vs_failure.png.
"""

import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from estimate2d import estimate_nfxp_2d
from mdp2d import RustMDP2D, Solution2D
from probes import readout_direction, ridge_probe
from train_agents import MLPAgent
from train_agents2d import agent_policy, all_state_features

RESULTS = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------- targets --
def bucket_average(grid: np.ndarray, weights: np.ndarray,
                   n_xb: int, n_cb: int) -> np.ndarray:
    """Replace grid values by their (ergodic-weighted) bucket averages."""
    kx, kc = grid.shape
    out = np.empty_like(grid)
    xe = np.linspace(0, kx, n_xb + 1).astype(int)
    ce = np.linspace(0, kc, n_cb + 1).astype(int)
    for i in range(n_xb):
        for j in range(n_cb):
            sl = (slice(xe[i], xe[i + 1]), slice(ce[j], ce[j + 1]))
            w = weights[sl] + 1e-9
            out[sl] = np.average(grid[sl], weights=w)
    return out


def zoo_targets(solution: Solution2D, ergodic_grid: np.ndarray) -> dict:
    grid = solution.ccp_grid()
    kc = solution.mdp.n_cost
    asym = grid[:, np.maximum(np.arange(kc), kc // 2)]  # clamp low regimes
    return {
        "clone": grid,
        "fine2d": bucket_average(grid, ergodic_grid, 12, 3),
        "coarse2d": bucket_average(grid, ergodic_grid, 6, 2),
        "blindbucket": bucket_average(grid, ergodic_grid, 5, 1),
        "asym": asym,
    }


def train_on_target(target_grid: np.ndarray, mdp: RustMDP2D,
                    n_epochs: int = 3000, lr: float = 1e-3,
                    seed: int = 0) -> MLPAgent:
    torch.manual_seed(seed)
    feats = all_state_features(mdp)
    target = torch.tensor(np.clip(target_grid.ravel(), 1e-6, 1 - 1e-6),
                          dtype=torch.float32)
    agent = MLPAgent(n_features=feats.shape[1])
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = loss_fn(agent(feats), target)
        loss.backward()
        opt.step()
    return agent


# ----------------------------------------------------------------- audits --
def c_shift_audit(agent, mdp: RustMDP2D, direction: np.ndarray,
                  layer: int = 0) -> float:
    """Mean |dP(replace)| from patching the c-induced activation shift
    projected on `direction`, over in-distribution mileage bins."""
    from probes import _forward_from

    u = direction / np.linalg.norm(direction)
    u_t = torch.tensor(u, dtype=torch.float32)
    feats = all_state_features(mdp)
    shifts = []
    with torch.no_grad():
        acts_all = agent.hidden_activations(feats)[layer]
        for x in range(5, 45, 5):
            a_lo = acts_all[x * mdp.n_cost + 0]
            a_hi = acts_all[x * mdp.n_cost + (mdp.n_cost - 1)]
            patched = a_lo + torch.dot(a_hi - a_lo, u_t) * u_t
            p_clean = torch.sigmoid(_forward_from(agent, layer, a_lo)).item()
            p_patch = torch.sigmoid(_forward_from(agent, layer, patched)).item()
            shifts.append(abs(p_patch - p_clean))
    return float(np.mean(shifts))


def run_audits(agent, mdp: RustMDP2D, layer: int = 0) -> dict:
    feats = all_state_features(mdp)
    with torch.no_grad():
        acts = agent.hidden_activations(feats)[layer].numpy()
    _, c = mdp.state_grid()
    probe_dir = ridge_probe(acts, c.astype(float))["weights"]
    return {
        "audit_readout": c_shift_audit(agent, mdp,
                                       readout_direction(agent, layer), layer),
        "audit_probe": c_shift_audit(agent, mdp, probe_dir, layer),
    }


# ----------------------------------------------------------- experiment ----
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
        agent = train_on_target(target, truth)
        pol = agent_policy(agent, truth)
        pol_grid = pol.reshape(truth.n_mileage, truth.n_cost)

        beh_gap = float(np.sum(ergodic_grid.ravel()
                               * np.abs(pol - sol.ccp_replace)))
        audits = run_audits(agent, truth)

        # zlib.crc32 is deterministic across processes (builtin hash() is
        # salted per-process and breaks reproducibility of the panel).
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

        rows.append(dict(name=name, beh_gap=beh_gap, **audits,
                         theta1=est.theta1, theta2=est.theta2, rc=est.rc,
                         cf_rmse=cf_rmse))
        print(f"  beh_gap={beh_gap:.4f}, audit_ro={audits['audit_readout']:.4f}, "
              f"audit_pr={audits['audit_probe']:.4f}, "
              f"theta=({est.theta1:.4f},{est.theta2:.3f},{est.rc:.2f}), "
              f"CF-RMSE={cf_rmse:.4f}")

    # -------- figure ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for r in rows:
        ax.scatter(r["audit_readout"], r["cf_rmse"], s=110, zorder=3)
        ax.annotate(r["name"], (r["audit_readout"], r["cf_rmse"]),
                    textcoords="offset points", xytext=(9, 5), fontsize=10)
    ax.set_xlabel("readout-aware internal audit of cost-regime use "
                  "(in-distribution states only)")
    ax.set_ylabel("regime-shift counterfactual RMSE")
    ax.set_title("Out-of-class agent zoo: internal audit vs counterfactual failure")
    fig.tight_layout()
    fig.savefig(RESULTS / "zoo_audit_vs_failure.png", dpi=150)
    print(f"figure -> {RESULTS / 'zoo_audit_vs_failure.png'}")

    table = "\n".join(
        f"| {r['name']} | {r['beh_gap']:.4f} | {r['audit_readout']:.4f} | "
        f"{r['audit_probe']:.4f} | {r['theta1']:.4f} | {r['theta2']:.3f} | "
        f"{r['rc']:.2f} | {r['cf_rmse']:.4f} |" for r in rows)
    summary = f"""# Out-of-class zoo — readout-aware audit vs counterfactual failure

Truth: theta1={truth.theta1}, theta2={truth.theta2}, rc={truth.rc}.
Counterfactual: permanent worst regime (c=max), prediction from NFXP estimates
vs the agent's actual policy.

| agent | ergodic behavior gap | audit (readout) | audit (probe dir) | theta1_hat | theta2_hat | RC_hat | CF RMSE |
|---|---|---|---|---|---|---|---|
{table}

(Annotate after inspection: does the readout audit order agents by CF failure
where the probe-direction audit does not? Which out-of-class rules produce
"confidently wrong" estimates?)
"""
    # _raw suffix: never overwrite the hand-annotated zoo_summary.md
    (RESULTS / "zoo_summary_raw.md").write_text(summary, encoding="utf-8")
    print(f"summary -> {RESULTS / 'zoo_summary_raw.md'}")


if __name__ == "__main__":
    main()
