# Positioning notes (distilled from the 2026-07-05 verified lit review)

Full review + provenance: `C:\Users\briac\feynman\outputs\interp-structural-ground-truth.md`
(all 17 citations URL-verified there).

## Novelty verdict
The intersection cell {internal mech-interp probes} × {structural counterfactual
validity} × {ground-truth solved MDP with distinct-mechanism agents} is **empty**
in the reachable literature. The claim holds **with a mandatory caveat**, stated
in the introduction, not buried:

> The econometric fact that DDC counterfactuals can be unidentified is known
> theory (Magnac & Thesmar 2002; Kalouptsidi, Scott & Souza-Rodrigues 2021), and
> "good fit ≠ correct mechanism" is known in ML (Vafa et al.; Othello-GPT; IRL
> identifiability). Our contribution is making the analytical failure
> *empirically visible at the mechanism level* and *prospectively diagnosable*
> via internal audits.

## Non-negotiable citations
- **Magnac & Thesmar (2002), Econometrica 70(2)** — DDC non-identification; the
  theoretical reason Exhibit 3's estimates can be confidently wrong. Missing it
  = desk-reject trigger.
- **Kalouptsidi, Scott & Souza-Rodrigues (2021), QE 12(2)** — some
  counterfactuals are identified even when the model isn't; sharpens which
  counterfactuals our diagnostic should target.
- **Vafa, Chang, Rambachan & Mullainathan (2025, ICML)** — nearest neighbor
  (orbits: good fit, heuristics that fail to generalize). Difference to draw
  precisely: their "inductive bias probe" is *behavioral adaptation*; ours are
  *internal activations + causal patching*, aimed at an *estimator's* validity.
- **Vafa et al. (2024, NeurIPS)**, **McGrath et al. (2022, PNAS)** (AlphaZero
  value probing — biggest "already done" risk on the interp axis; no
  econometrics), **Li et al. (2023)** + **Nanda et al. (2023)** (Othello-GPT),
  **Lucas (1976)**, **Heckman (2010)**, **Estrella & Fuhrer (1999)**,
  **Skalse & Abate (2024)** (IRL identifiability = same theorem in ML dialect),
  **Heimersheim & Nanda (2024)** (patching methodology), **Rust (1987)**.

## The three referee traps and their answers
1. *"Just DDC non-identification re-skinned?"* → We make an analytical result
   empirically visible and ex-ante diagnosable; see caveat framing above.
2. *"1-D toy"* → the 2-D cost-shifter extension (this repo, `src/*2d.py`) is the
   answer: probing degeneracy disappears, the estimator is used as in practice.
3. *"Probes are known to be non-causal"* → Exhibit 4 already concedes and
   demonstrates it; the 2-D causal-patching exhibit converts the concession into
   the paper's positive methodological claim (lit review Q1).

## The strongest-possible-result target (lit review Q2)
A monotone relationship between an internal-audit statistic and counterfactual
RMSE across a *family* of agents (not just two hand-built ones). Design the 2-D
lab so agent families (varying how much they use the cost shifter) are cheap to
generate.
