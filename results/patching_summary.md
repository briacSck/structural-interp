# Exhibit 4 — directional activation patching

Question: is the linearly-decodable V(x) direction *causally used* by each
agent, or merely present? We patch only that direction (source -> target
state) and measure the fraction of the clean behavioral (logit) gap it
carries, against a random-direction control.

| agent | layer | probe dir | readout dir | random dir |
|---|---|---|---|---|
| clone | 0 | 0.026 | 0.809 | 0.006 |
| clone | 1 | 0.005 | 1.000 | 0.066 |
| heuristic | 0 | -0.002 | 1.467 | 0.417 |
| heuristic | 1 | -0.002 | 1.000 | 0.061 |

Reading: recovery near 1 = the direction carries the decision signal; near 0
= causally idle. "readout dir" is the direction the network provably reads
(output weights, propagated first-order for layer 0) — the positive control.
A probe direction with high decoding R^2 but ~0 causal recovery is direct
evidence that decodability does not license causal claims, even in a 2-layer
MLP on a 1-D task.
