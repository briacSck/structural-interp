"""Interpretability tools for the neural agents (week 4).

Core question: do the agent's hidden layers *linearly encode* the objects the
structural model assumes it optimizes — the integrated value function V(x),
the choice-value difference v(x,1) - v(x,0), or just the raw state?

Method 1 — linear probes (Alain & Bengio 2016): ridge-regress a target
quantity on hidden activations across states; report held-out R^2. High R^2
for V(x) in the clone but not the heuristic agent is the paper's first
exhibit.

Method 2 — activation patching (week 4, TODO): replace an internal
activation computed at state x with the one from state x' and check whether
the output moves as the value function predicts (causal use, not just
decodability).
"""

import numpy as np


def ridge_probe(activations: np.ndarray, target: np.ndarray,
                l2: float = 1e-3, train_frac: float = 0.7,
                seed: int = 0) -> dict:
    """Fit target ~ activations by ridge regression; return held-out R^2.

    activations: (n_states, n_units) hidden activations, one row per state.
    target:      (n_states,) quantity to decode, e.g. Solution.v_bar.
    """
    rng = np.random.default_rng(seed)
    n = len(target)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    train, test = idx[:n_train], idx[n_train:]

    x_mean = activations[train].mean(axis=0)
    y_mean = target[train].mean()
    xc, yc = activations[train] - x_mean, target[train] - y_mean

    gram = xc.T @ xc + l2 * np.eye(xc.shape[1])
    weights = np.linalg.solve(gram, xc.T @ yc)

    pred = (activations[test] - x_mean) @ weights + y_mean
    ss_res = np.sum((target[test] - pred) ** 2)
    ss_tot = np.sum((target[test] - target[test].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"r2_test": float(r2), "weights": weights,
            "train_idx": train, "test_idx": test}


def probe_agent_for_value(agent, solution) -> list[dict]:
    """Probe every hidden layer of an MLPAgent for the integrated value function."""
    import torch
    from train_agents import state_features

    k = solution.mdp.n_states
    feats = torch.tensor(state_features(np.arange(k), k), dtype=torch.float32)
    with torch.no_grad():
        layer_acts = agent.hidden_activations(feats)

    results = []
    for layer_i, acts in enumerate(layer_acts):
        out = ridge_probe(acts.numpy(), solution.v_bar)
        results.append({"layer": layer_i, "r2_test": out["r2_test"]})
    return results


def activation_patch(agent, source_state: int, target_state: int):
    """TODO (week 4): patch hidden activations across states, measure output shift."""
    raise NotImplementedError


if __name__ == "__main__":
    from mdp import RustMDP
    from train_agents import train_clone

    sol = RustMDP().solve()
    agent = train_clone(sol)
    for row in probe_agent_for_value(agent, sol):
        print(f"layer {row['layer']}: held-out R^2 for V(x) = {row['r2_test']:.4f}")
