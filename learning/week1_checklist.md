# Week 1–2 learning checklist (mech-interp ramp)

Goal: enough PyTorch + interp fluency to own weeks 3–4 of the pilot. Budget
~15–20 focused hours across two weeks. Do these in order.

## PyTorch basics (~6h)
- [ ] Official "Learn the Basics" tutorial (tensors → autograd → nn.Module → training loop): https://pytorch.org/tutorials/beginner/basics/intro.html
- [ ] Rebuild `src/train_agents.py`'s `train_clone` from scratch without looking — if you can, you know enough PyTorch for this project.

## Mech interp orientation (~4h reading)
- [ ] Neel Nanda, "Concrete Steps to Get Started in Mechanistic Interpretability": https://www.neelnanda.io/mechanistic-interpretability/getting-started
- [ ] Nanda's glossary (skim, keep as reference): https://www.neelnanda.io/mechanistic-interpretability/glossary
- [ ] Olah et al., "Zoom In: An Introduction to Circuits" (2020): https://distill.pub/2020/circuits/zoom-in/
- [ ] Li et al. Othello-GPT (2023) + Nanda's follow-up "Actually, Othello-GPT Has A Linear Emergent World Representation" — *the* template for this project's probing exhibit.

## Hands-on interp (~6h)
- [ ] ARENA 3.0 curriculum, chapter 1.1 (transformers from scratch) — optional for the MLP pilot, required before the transformer extension: https://arena-chapter1-transformer-interp.streamlit.app/
- [ ] Alain & Bengio (2016) linear probes paper (skim §1–3), then read `src/probes.py` and check the method matches.
- [ ] Activation patching explainer (Nanda): https://www.neelnanda.io/mechanistic-interpretability/attribution-patching — we only need the basic patching idea, not attribution patching.

## Deferred (post-pilot, for the transformer extension)
- TransformerLens main demo notebook
- Elhage et al., "Toy Models of Superposition" (2022)
- Gemma Scope / SAEs (only relevant for LLM-scale follow-ups)

## Litmus test at end of week 2
You should be able to explain, in one paragraph each, to an economist:
1. What a linear probe shows and what it cannot show (decodability ≠ use).
2. Why activation patching is a *causal* claim about internals.
3. How the Othello-GPT result maps onto "does the agent represent V(x)?"
