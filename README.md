# Do Structural Models Recover Mechanisms?
### Evidence from Neural Agents with Observable Internals

**Research question.** When an econometrician estimates a structural model
(dynamic discrete choice) on an agent's behavior, do the recovered parameters
capture the agent's actual decision *mechanism* — and can causal
interpretability audits detect, *ex ante*, which counterfactuals will fail?
With human data this is untestable; with neural agents the mechanism is fully
observable. **Draft note: [`paper/note.tex`](paper/note.tex).**

## Findings (as they actually came out)

1. **In-class bounded rationality is benign.** Inattentive agents (Gabaix
   sparse-max, Matějka–McKay RI) are absorbed into as-if parameters — θ̂₂
   tracks the attention parameter almost linearly — and their
   counterfactuals stay accurate, *including* a pre-registered Lucas-style
   subsidy experiment in which agents re-learn. The Fosgerau et al. (2020)
   RI–logit equivalence, made empirical.
2. **Out-of-class cognition fails silently.** Categorical agents
   (Fryer–Jackson lookup rules) with 2–3% behavioral gaps yield
   replacement-cost estimates of 4.6–7.0 vs a truth of 10, and a predicted
   policy response to the subsidy roughly **twice** the actual one — with
   smooth likelihoods and interior optima throughout.
3. **Probes can't see it; causal conformity audits can.** Linear probes
   decode everything from everyone (R² ≈ 1) and their directions are
   causally idle (the amnesic-probing lesson, Elazar et al. 2021). A
   **conformity audit** — dispersion of locally patched causal sensitivities
   per state variable, with an insensitivity gate — computed on
   in-distribution activations only, rank-predicts counterfactual failure:
   **Spearman 0.87** (regime-shift transport) and **0.80** (subsidy with
   re-learning) across 13 microfounded agents.

**Thesis.** Internal audits are specification tests on mechanisms: each
channel audit checks conformity of the agent's computation to the assumed
model class, and each counterfactual is threatened exactly by the channels it
stresses. Scope boundary (companion paper): under *endogenous* attention the
cognitive primitive itself moves with policy — the Lucas critique one level
up — and pre-policy audits cannot see it.

## Repo map

| Path | What it is |
|---|---|
| `paper/note.tex` | The research note (~9 pp., self-contained; compile on Overleaf) |
| `paper/positioning.md` | Novelty verdict, must-cites, referee traps (from verified lit reviews) |
| `paper/prereg_subsidy_cf.md` | Pre-registration of the subsidy experiment (committed before the run) |
| `src/mdp.py`, `src/mdp2d.py` | Ground-truth environments (Rust 1987; + observable cost regime), exact solutions |
| `src/estimate.py`, `src/estimate2d.py` | NFXP MLE and Hotz–Miller CCP estimators (recovery-tested) |
| `src/train_agents.py`, `src/train_agents2d.py` | MLP agents; λ-family |
| `src/zoo_v2.py` | **Main experiment**: 13 microfounded agents (categories, sparse-max, RI, salience), resumable |
| `src/probes.py`, `src/leace.py` | Linear probes, directional patching, LEACE erasure + amnesic audit |
| `src/audit_battery.py`, `src/c_conformity.py` | Per-channel dispersion audits + conformity score (the diagnostic) |
| `src/subsidy_counterfactual.py` | Pre-registered subsidy CF with stable-cognitive-primitive re-learning |
| `src/experiment.py`, `src/patching_experiment.py`, `src/experiment2d.py`, `src/experiment2d_zoo.py` | Earlier exhibits (1-D pilot; probe-direction patching; λ-family; ad-hoc zoo) |
| `results/*.md` | Annotated run summaries (`*_raw.md` = script output, never overwritten by reruns) |
| `learning/` | Mech-interp ramp checklist (historical) |

## Reproduce

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m pytest tests -v        # 12 tests: VI, simulation, NFXP + HM recovery, agents, 2-D env
.venv\Scripts\python src\zoo_v2.py             # main experiment (resumable; ~20 min CPU)
.venv\Scripts\python src\c_conformity.py       # conformity audit (Spearman 0.87)
.venv\Scripts\python src\subsidy_counterfactual.py  # pre-registered subsidy CF (Spearman 0.80)
```

Deterministic seeding throughout (`zlib.crc32`, torch seed 0). Laptop CPU
only; no GPU required.

## Positioning (one paragraph)

Vafa–Chang–Rambachan–Mullainathan (ICML 2025) show foundation models fit
orbits with heuristics that fail to generalize — *behaviorally*. We go inside
the network, in an economic laboratory with exact ground truth, and connect
the internals to the validity of an econometric procedure. The DDC
identification results (Magnac–Thesmar 2002; Kalouptsidi et al. 2021; Bugni–
Ura 2019) are the theory our lab makes visible and diagnosable; the
contribution is the synthesis plus the ex-ante audit. Full citation map:
`paper/positioning.md`.

*Separate from the author's `discrete_vibecoding` working paper; the Rust
(1987) model is used purely as a canonical laboratory.*
