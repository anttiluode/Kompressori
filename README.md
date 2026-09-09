# Kompressori

**Compress the state. Then ask whether you also compressed the future operator.**

Kompressori starts from the old PhiWorld-style nonlinear field, but the goal is no longer to make a "hologram" claim. The first question is narrower:

> A partial state can preserve part of the visible future while failing to preserve how the system responds to the *next* perturbation.

That distinction matters for any adaptive system. A memory guard, compressed world-model, recurrent substrate, or learned latent may reproduce today's answers yet have a very different local response geometry.

## Gate 0 — future fidelity is not response fidelity

We evolve the PhiWorld-like field to a structured state `s=(phi, phi_old)`, retain only a fraction of the cells, and zero the hidden state.

Two measurements are made on hidden cells over a 50-step future:

1. **future correlation** — does the compressed state follow the same future trajectory?
2. **response correlation** — after applying the same tiny Gaussian displacement to full and compressed states, do their *changes* follow the same trajectory?

The second is a finite-difference tangent-response test. The probe is small enough to be close to linear: halving it and doubling its response disagrees with the full probe by at most **0.043 relative error** across the three default probe seeds.

### The first interesting failure

At a **5% square-edge-shaped cue**:

- future correlation: **0.408**
- response correlation: **0.004**
- response gain ratio: **4.389** (`1.0` would match the full state's response magnitude)

So the cue retains a nontrivial resemblance to the future while its response to the next perturbation is essentially unrelated and about 4.4× too large.

```text
looks similar later
        !=
has the same future sensitivity
```

At **25% random** observation the two correlations become closer (`future=0.517`, `response=0.481`), though the response is still amplified (`gain=1.434`). At **50% energy-selected** cells the future correlation reaches `0.665` and response correlation `0.516`, with gain `0.827`. None is yet strong completion.

## Gate 1 — optimizing the future does not recover the response

Gate 1 fixes the observation budget at **5%**, generates **200 random masks**, and ranks them using **only future trajectory correlation**. It then surprises every compressed state with five perturbation packets that were not used for selection.

Result:

- Pearson correlation between future fidelity and response fidelity across the 200 masks: **0.026**
- best future-only mask: future `0.344`, unseen response `0.070`, response gain `3.293`
- best response-preserving mask: future `0.296`, unseen response `0.340`, response gain `2.693`
- the response-best mask ranked only **71st** by future fidelity

So, in this toy search, **future/replay fidelity is almost no guide to perturbation-response fidelity**. This is not a universal theorem and not evidence about brains. It is a concrete counterexample to a tempting engineering assumption: choosing a compressed state because it reproduces known trajectories can leave the local future operator badly wrong.

The full receipt is in `results/gate1/receipt.json`. The exact long run was:

```bash
python gate1_search.py --horizon 50 --candidates 200 --probe-seeds 5
```

The script defaults are smaller for quicker iteration.

## Why the old PhiWorld result belongs here

The predecessor probe already tested partial-field completion in two modes: one-shot partial initial state and continuous trajectory clamping. It explicitly measured only hidden cells. The old result did **not** show a holographic reconstruction attractor: distributed random samples helped more than compact center/ring cues, but even 50% random observation only reached modest hidden correlation, and no mask reached 0.75 mean correlation in the supplied run.

Kompressori keeps that negative result as the baseline and adds the missing question: **did the compressed state retain the response geometry?**

A detail worth making explicit: the simulation Laplacian uses `np.roll`, so the domain is periodic. The `edge` mask is a square-shaped observation geometry; it is **not a physical boundary condition**.

## Euler / Navier–Stokes inspiration, without overclaiming

OpenAI's September 2026 release proposes finite-time blowup results for Navier–Stokes and Euler and supplies Lean formalizations. The conceptual prompt we borrow is not "PhiWorld is a fluid". It is the more general idea that perturbations can be transformed by the current state into finite-time response geometries with very different amplification properties.

For Kompressori the operational translation is:

```text
preserve state / answers
        ↓
not enough
        ↓
preserve finite-time response geometry too
```

Official sources:

- https://openai.com/research/
- https://github.com/openai/NavierStokesAndEuler

## Run

```bash
python -m pip install numpy
python kompressori.py
python gate1_search.py
```

Gate 0 writes `results/gate0/receipt.json`; Gate 1 writes `results/gate1/receipt.json`.

## Repo map

- `kompressori.py` — Gate 0 future-vs-response experiment
- `gate1_search.py` — future-only mask search, followed by unseen response tests
- `results/gate0/receipt.json` — Gate 0 receipt
- `results/gate1/receipt.json` — Gate 1 receipt
- `legacy/phiworld_completion_probe.py` — predecessor experiment
- `legacy/completion_summary.json` — predecessor receipt supplied with the experiment
- `index.html` — dependency-free visual summary for GitHub Pages
- `test_kompressori.py` — smoke tests

## Next gates

- **G2 — response-aware compressor:** optimize a combined objective over state and tangent response. Does the selected mask become spatially distributed, multiscale, ring-like, or something stranger?
- **G3 — perturbation transfer:** ask whether perturbation A changes the field so perturbation B is amplified more strongly later — the toy parent→child cascade idea.
- **G4 — slow substrate:** let the response geometry itself change slowly and ask which compressed constraints prevent runaway self-amplification.
- **G5 — learning guard:** move the same distinction into an adaptive model: preserve stored answers versus preserve the Jacobian/impulse responses around them.

The repo succeeds if these gates kill the seductive story quickly or turn it into a measurable mechanism.
