"""Exhibit 4: causal patching of the V(x) probe direction.

Exhibit 2 showed linear probes decode V(x) equally well from both agents
(the 1-D state artifact): decodability cannot separate them. This experiment
asks the causal question instead — when we transplant ONLY the probed V(x)
direction of the hidden activations from a source state into a target state,
how much of the agent's behavioral change does that single direction carry?

Metric: mean recovery of the clean source-target logit gap, probe direction
vs a random-direction control, per agent and layer.

Output: results/patching_summary.md (+ console table).
"""

from pathlib import Path

from mdp import RustMDP
from probes import patching_recovery
from train_agents import train_clone, train_heuristic

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    sol = RustMDP().solve()
    print("training agents...")
    agents = {"clone": train_clone(sol), "heuristic": train_heuristic(sol)}

    rows = []
    for name, agent in agents.items():
        for layer in (0, 1):
            rec = patching_recovery(agent, sol, layer)
            rows.append((name, layer, rec["probe"], rec["readout"],
                         rec["random"]))
            print(f"{name} layer {layer}: probe {rec['probe']:.3f}, "
                  f"readout {rec['readout']:.3f}, random {rec['random']:.3f}")

    table = "\n".join(f"| {n} | {l} | {p:.3f} | {ro:.3f} | {r:.3f} |"
                      for n, l, p, ro, r in rows)
    summary = f"""# Exhibit 4 — directional activation patching

Question: is the linearly-decodable V(x) direction *causally used* by each
agent, or merely present? We patch only that direction (source -> target
state) and measure the fraction of the clean behavioral (logit) gap it
carries, against a random-direction control.

| agent | layer | probe dir | readout dir | random dir |
|---|---|---|---|---|
{table}

Reading: recovery near 1 = the direction carries the decision signal; near 0
= causally idle. "readout dir" is the direction the network provably reads
(output weights, propagated first-order for layer 0) — the positive control.
A probe direction with high decoding R^2 but ~0 causal recovery is direct
evidence that decodability does not license causal claims, even in a 2-layer
MLP on a 1-D task.
"""
    path = RESULTS / "patching_summary.md"
    path.write_text(summary, encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
