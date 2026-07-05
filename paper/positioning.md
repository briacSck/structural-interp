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

## Additions from the feynman code/lit audit (2026-07-06, verified)
Full audit: `C:\Users\briac\feynman\outputs\structure-x-probing-ddc-audit.md`.

**New must-cites (interp axis):**
- **Elazar, Ravfogel, Jacovi & Goldberg (2021), "Amnesic Probing", TACL**
  (arXiv:2006.00995) — decodability ≠ use is THEIR thesis; Exhibit 4 must
  cite it and claim novelty only for the structural-econometrics setting.
- **INLP (Ravfogel et al. 2020, ACL) → R-LACE → LEACE (Belrose et al.)** —
  principled concept-erasure directions. Design upgrade: define audit
  subspaces via LEACE (provably complete linear erasure) instead of the ad
  hoc ridge direction (whose "anti-informative" behavior our zoo already
  exhibited, as this literature predicts).

**New must-cites (econ axis) — the zoo is a bounded-rationality family:**
- blindbucket = **Gabaix (2014) sparse-max** (zero attention on c);
- coarse2d/fine2d = **Fryer & Jackson (2008)** categorical cognition
  (`bucket_average` IS their model);
- λ-family = **Matějka & McKay (2015, AER)** RI-logit attenuation;
  dynamics: **Steiner, Stewart & Matějka (2017, ECMA, DOI 10.3982/ECTA13636)**;
- **Fosgerau, Melo, de Palma & Shum (2020, IER)** equivalence licenses
  treating any bucketed/perturbed rule as an RI model — the zoo becomes
  microfounded, not ad hoc;
- **Fudenberg & Strzalecki (2015, ECMA, DOI 10.3982/ECTA11846)** for
  dynamic-logit foundations; asym needs an asymmetric default/salience
  structure (symmetric RI cannot produce it);
- **Bugni & Ura (2019, QE)** — local misspecification bias in DDC/CCP
  estimation: the theory behind our "confidently wrong" R̂C (5.9/6.7 vs 10).

**Known identification caveat to state:** the NFXP passes true transition
parameters (mileage process, cost chain) as known; estimating them is
standard but should be said explicitly.

## The strongest-possible-result target (lit review Q2)
A monotone relationship between an internal-audit statistic and counterfactual
RMSE across a *family* of agents (not just two hand-built ones). Design the 2-D
lab so agent families (varying how much they use the cost shifter) are cheap to
generate.
