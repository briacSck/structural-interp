# One-pager: Do Structural Models Recover Mechanisms? Evidence from Neural Agents with Observable Internals

**Author:** Briac Sockalingum · **Status:** pilot started July 2026

## The problem
Structural econometrics earns its keep through counterfactuals, which are valid
only if estimated parameters capture the agent's actual decision *mechanism*,
not merely fit its behavior. This assumption is untestable with human data:
we never observe the true decision process. Neural agents break the impasse —
their internals are fully inspectable with mechanistic-interpretability tools.

## The experiment
1. **Laboratory:** the canonical Rust (1987) bus-engine replacement MDP,
   solved exactly, so the true value function and parameters are known.
2. **Two agents, same behavior, different mechanisms:** an MLP cloned on the
   optimal policy vs. a behaviorally-matched heuristic (noisy mileage
   threshold). In-distribution choice data from the two are near-indistinguishable.
3. **The econometrician's exercise:** estimate the dynamic discrete choice
   model (NFXP and Hotz–Miller) on each agent's simulated panel. Both yield
   plausible θ̂ with good fit.
4. **Opening the box:** linear probes and activation patching test whether
   each agent internally represents and causally uses the value function.
5. **The payoff:** counterfactual experiments (change the replacement cost,
   shift the transition process) — structural predictions hold for the agent
   with value-function internals and fail for the heuristic agent, and the
   interpretability audit predicted *which* would fail before any
   counterfactual data existed.

## Contribution
- **To econometrics:** first ground-truth test of when structural estimates
  are mechanism-valid vs. merely behavior-fitting, and a proof-of-concept that
  interpretability can serve as an ex-ante specification test.
- **To interpretability:** a new model organism — value functions as world
  models — with an external scientific referee (econometric theory) instead of
  self-graded interpretations.
- **Adjacent literature:** Vafa, Rambachan, Mullainathan & Kleinberg (2024)
  evaluate implicit world models behaviorally; Li et al. (2023, Othello-GPT)
  and Nanda et al. probe game-board world models without an economic estimand.
  The intersection — structural estimation × mechanistic ground truth — is open.

## Feasibility
Toy-scale (90-state MDP, MLPs with ~1k parameters): laptop CPU, zero compute
budget. Pipeline (environment, simulator, NFXP estimator, probe tooling) is
already running with passing recovery tests. Six-week pilot to a research
note; extensions (small transformers + TransformerLens, misspecified
econometrician, RL-trained agents) scale gracefully.
