# Structural Estimation Meets Ground Truth

**Research question.** When an econometrician estimates a structural model
(dynamic discrete choice) on a neural agent's behavior, do the recovered
parameters correspond to the agent's actual internal mechanism — and can
mechanistic interpretability tools detect, *ex ante*, when structural
counterfactuals will fail?

**Positioning.** Vafa–Rambachan–Mullainathan–Kleinberg (NeurIPS 2024) test
whether generative models have coherent world models *behaviorally*. We go
inside the network — "Othello-GPT for structural econometrics." This project
is fully separate from the `discrete_vibecoding` working paper; it uses the
*textbook* Rust (1987) bus-replacement model purely as a canonical laboratory.

> **Environment choice (revisit at week 2):** default is the textbook bus
> problem. Drop-in alternatives if we want more distance from anything
> Rust-flavored: consumer savings/borrowing, firm entry/exit, job search.
> The research question is environment-agnostic.

## Design

1. **Ground truth** (`src/mdp.py`): solve the MDP exactly (value iteration),
   giving the true value function, CCPs, and parameters θ = (θ₁, RC).
2. **Agents** (`src/train_agents.py`): (a) an MLP behaviorally cloned on the
   optimal policy; (b) a mechanistically different agent (noisy threshold
   heuristic or early-stopped RL) that matches (a)'s behavior in-distribution.
3. **Interpretability** (`src/probes.py`): linear probes for the value
   function in hidden layers; activation patching for causal use.
4. **Econometrics** (`src/estimate.py`): NFXP MLE (done) and Hotz–Miller CCP
   estimator (week 5) run on each agent's simulated choices. Compare recovered
   θ̂ across agents; run counterfactuals (e.g. change RC) and score predicted
   vs. actual agent behavior.
5. **Punchline**: structural estimation fits both agents; probes tell them
   apart; counterfactuals fail for exactly the agent whose internals lack the
   value representation.

## Pilot timeline (6 weeks)

| Week | Milestone |
|---|---|
| 1–2 | PyTorch ramp (see `learning/week1_checklist.md`) + ground truth module ✅ |
| 3 | Both agents trained, behavioral equivalence verified |
| 4 | Probe + patching results, clone vs heuristic contrast figure |
| 5 | Structural estimates + counterfactual failure exhibit |
| 6 | 6–10 page research note (arXiv/SSRN) + accessible post |

## Setup & verification

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
py -m pytest tests -v          # validates VI, simulation, and NFXP recovery
py src\mdp.py                  # quick ground-truth sanity print
```
