"""2-D experiment: does an internal causal audit predict counterfactual failure?

Answers the lit review's two open questions:
Q1 — causal separation: linear probes decode the cost regime c from every
     agent (it sits in the inputs), but patching the c-carrying direction
     moves behavior only in agents that USE it. The audit is causal, internal,
     and computed on in-distribution states only.
Q3/Q2 — a one-parameter family of agents (lambda = how much the agent uses
     the cost regime) lets us plot audit statistic vs counterfactual RMSE.
     A monotone relationship is the paper's strongest possible exhibit.

Counterfactual: a permanent cost-regime shift (economy pinned at the worst
regime, c = max). The econometrician predicts P(replace | x, c=max) from
NFXP estimates; truth = the agent's actual policy at those states.

Outputs: results/exp2d_summary.md, results/exp2d_audit_vs_failure.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from estimate2d import estimate_nfxp_2d
from mdp2d import RustMDP2D
from probes import ridge_probe
from train_agents2d import agent_policy, all_state_features, train_agent_2d

RESULTS = Path(__file__).resolve().parents[1] / "results"
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def c_probe_and_audit(agent, mdp: RustMDP2D, layer: int = 0) -> dict:
    """Decode c from layer activations (R^2) and measure its causal use.

    Audit: for each in-distribution mileage bin, patch the c-probe direction
    of the hidden activations from (x, c=max) into the run at (x, c=0) and
    record the shift in P(replace). Mean absolute shift = audit statistic.
    """
    from probes import directional_patch

    feats = all_state_features(mdp)
    with torch.no_grad():
        acts = agent.hidden_activations(feats)[layer].numpy()
    _, c = mdp.state_grid()
    probe = ridge_probe(acts, c.astype(float))
    direction = probe["weights"]

    shifts = []
    for x in range(5, 45, 5):  # in-distribution mileage bins
        s_lo = x * mdp.n_cost + 0
        s_hi = x * mdp.n_cost + (mdp.n_cost - 1)
        out = _patch_2d(agent, layer, direction, s_hi, s_lo, mdp)
        shifts.append(abs(1 / (1 + np.exp(-out["logit_patched"]))
                          - 1 / (1 + np.exp(-out["logit_target"]))))
    return {"r2_c": probe["r2_test"], "audit": float(np.mean(shifts))}


def _patch_2d(agent, layer: int, direction: np.ndarray,
              source_flat: int, target_flat: int, mdp: RustMDP2D) -> dict:
    """directional_patch analog operating on flattened 2-D states."""
    from probes import _forward_from

    u = direction / np.linalg.norm(direction)
    feats = all_state_features(mdp)[[source_flat, target_flat]]
    with torch.no_grad():
        acts = agent.hidden_activations(feats)[layer]
        u_t = torch.tensor(u, dtype=torch.float32)
        shift = torch.dot(acts[0] - acts[1], u_t) * u_t
        return {
            "logit_target": _forward_from(agent, layer, acts[1]).item(),
            "logit_patched": _forward_from(agent, layer, acts[1] + shift).item(),
            "logit_source": _forward_from(agent, layer, acts[0]).item(),
        }


def counterfactual_rmse(est, agent_pol_grid: np.ndarray,
                        template: RustMDP2D) -> float:
    """Regime-shift counterfactual: predicted vs actual policy at c = max."""
    mdp_hat = RustMDP2D(n_mileage=template.n_mileage, n_cost=template.n_cost,
                        theta1=est.theta1, theta2=est.theta2, rc=est.rc,
                        beta=template.beta,
                        mileage_probs=template.mileage_probs,
                        cost_persist=template.cost_persist)
    pred = mdp_hat.solve().ccp_grid()[:, -1]
    actual = agent_pol_grid[:, -1]
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    truth = RustMDP2D()
    sol = truth.solve()

    rows = []
    for lam in LAMBDAS:
        print(f"lambda = {lam}: training...")
        agent = train_agent_2d(sol, lam)
        pol = agent_policy(agent, truth)
        pol_grid = pol.reshape(truth.n_mileage, truth.n_cost)

        interp = c_probe_and_audit(agent, truth)
        panel = sol.simulate(n_buses=500, n_periods=200,
                             rng=np.random.default_rng(int(lam * 100)),
                             policy=pol)
        print(f"  estimating NFXP...")
        est = estimate_nfxp_2d(panel, truth)
        cf_rmse = counterfactual_rmse(est, pol_grid, truth)
        rows.append((lam, interp["r2_c"], interp["audit"],
                     est.theta1, est.theta2, est.rc, cf_rmse))
        print(f"  R2(c)={interp['r2_c']:.3f}, audit={interp['audit']:.4f}, "
              f"theta=({est.theta1:.4f},{est.theta2:.3f},{est.rc:.2f}), "
              f"CF-RMSE={cf_rmse:.4f}")

    # ---- Figure: the audit -> failure plot -------------------------------
    lams = [r[0] for r in rows]
    audits = [r[2] for r in rows]
    rmses = [r[6] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(audits, rmses, c=lams, cmap="viridis", s=90,
                         zorder=3)
    for lam, a, r in zip(lams, audits, rmses):
        ax.annotate(f"$\\lambda$={lam}", (a, r), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)
    ax.set_xlabel("internal causal audit: mean |ΔP(replace)| from patching "
                  "the c-direction")
    ax.set_ylabel("regime-shift counterfactual RMSE")
    ax.set_title("Does the internal audit predict counterfactual failure?")
    fig.colorbar(scatter, label="$\\lambda$ (degree of cost-regime use)")
    fig.tight_layout()
    fig.savefig(RESULTS / "exp2d_audit_vs_failure.png", dpi=150)
    print(f"figure -> {RESULTS / 'exp2d_audit_vs_failure.png'}")

    table = "\n".join(
        f"| {lam:.2f} | {r2:.3f} | {aud:.4f} | {t1:.4f} | {t2:.3f} | "
        f"{rc_:.2f} | {cf:.4f} |"
        for lam, r2, aud, t1, t2, rc_, cf in rows)
    summary = f"""# 2-D experiment — internal audit vs counterfactual validity

Truth: theta1 = {truth.theta1}, theta2 = {truth.theta2}, rc = {truth.rc};
{truth.n_mileage} mileage bins x {truth.n_cost} cost regimes.
Counterfactual: permanent shift to the worst cost regime (c = max);
econometrician predicts from NFXP estimates, truth = agent's actual policy.

| lambda | R2 decode c | causal audit | theta1_hat | theta2_hat | RC_hat | CF RMSE |
|---|---|---|---|---|---|---|
{table}

Readings to check (see results/exp2d_summary.md for the annotated run):
- does theta2_hat track lambda (estimator recovers degree of regime use)?
- is CF RMSE monotone in lambda, and how large is it at lambda=0? (If small,
  the agent family sits inside the structural model class and cannot produce
  counterfactual failure — use out-of-class heuristics instead.)
- does the audit discriminate? (A ridge-probe-direction audit is expected to
  fail per Exhibit 4; a readout-aware audit is the fix.)
"""
    # _raw suffix: never overwrite the hand-annotated exp2d_summary.md
    (RESULTS / "exp2d_summary_raw.md").write_text(summary, encoding="utf-8")
    print(f"summary -> {RESULTS / 'exp2d_summary_raw.md'}")


if __name__ == "__main__":
    main()
