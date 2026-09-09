# Kompressori

**Compress the state. Then ask whether you also compressed the future operator.**

Kompressori starts from the old PhiWorld-style nonlinear field, but the goal is no longer to make a "hologram" claim. The first experiment asks a narrower question that came out of the recent Euler/Navier–Stokes discussion:

> A partial state can preserve part of the visible future while failing to preserve how the system responds to the *next* perturbation.

That distinction matters for any adaptive system. A memory guard, compressed world-model, recurrent substrate, or learned latent may reproduce today's answers yet have a very different local response geometry.

## Gate 0 — future fidelity is not response fidelity

We evolve the PhiWorld-like field to a structured state `s=(phi, phi_old)`. We then keep only a fraction of the cells and zero the hidden state.

Two measurements are made on the hidden cells over a 50-step future:

1. **future correlation** — does the compressed state follow the same future trajectory?
2. **response correlation** — after applying the same tiny Gaussian displacement to the full and compressed states, do their *changes* follow the same trajectory?

The second is a finite-difference tangent-response test. It is intentionally small enough to be close to linear: in the default receipt, halving the probe and doubling its response disagrees with the full probe by at most **0.043 relative error** across the three probe seeds.

### The first interesting failure

At a **5% square-edge-shaped cue**:

- future correlation: **0.408**
- response correlation: **0.004**
- response gain ratio: **4.389** (`1.0` would match the full state's response magnitude)

So this cue retains a nontrivial resemblance to the future while the response to the next perturbation is essentially unrelated and about 4.4× too large.

That is the point of Kompressori:

```text
looks similar later
        !=
has the same future sensitivity
```

At **25% random** observation the two correlations become much closer (`future=0.517`, `response=0.481`), though the response is still amplified (`gain=1.434`). At **50% energy-selected** cells the future correlation reaches `0.665` and response correlation `0.516`, with gain `0.827`. None of these is yet "strong completion." That negative result is useful.

## Why the old PhiWorld result belongs here

The predecessor probe already tested partial-field completion in two modes: one-shot partial initial state and continuous trajectory clamping. It explicitly measured only the hidden cells. The old result did **not** show a holographic reconstruction attractor: distributed random samples helped more than compact center/ring cues, but even 50% random observation only reached modest hidden correlation, and no mask reached 0.75 mean correlation in the supplied run.

Kompressori keeps that result as the baseline and adds the missing question: **did the compressed state retain the response geometry?**

A detail worth making explicit: the simulation Laplacian uses `np.roll`, so the domain is periodic. The `edge` mask is just a square-shaped observation geometry; it is **not a physical boundary condition**.

## Euler / Navier–Stokes inspiration, without overclaiming

OpenAI's September 2026 release proposes finite-time blowup results for Navier–Stokes and Euler and supplies Lean formalizations. The conceptual prompt we borrow is not "PhiWorld is a fluid". It is the more general idea that a perturbation can be transformed by the current state into a future geometry with very different amplification properties.

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
```

Outputs:

```text
results/gate0/runs.csv
results/gate0/receipt.json
```

The default run is small: 64×64 grid, 350 settling steps, 50-step evaluation horizon, 5 mask geometries, 6 observation fractions, 3 probe seeds.

## Repo map

- `kompressori.py` — Gate 0 experiment
- `results/gate0/receipt.json` — compact machine-readable receipt
- `results/gate0/runs.csv` — every run
- `legacy/phiworld_completion_probe.py` — predecessor experiment
- `legacy/completion_summary.json` — predecessor receipt supplied with the experiment
- `index.html` — dependency-free visual summary for GitHub Pages
- `test_kompressori.py` — smoke tests

## Next gates

G0 gives us a clean target rather than a grand claim. The next experiments should attack it.

- **G1 — learned compressor:** choose the observed cells to maximize future fidelity, then test whether response fidelity was accidentally destroyed.
- **G2 — response-aware compressor:** optimize a combined objective over state and tangent response. Does the chosen mask become spatially distributed, multiscale, ring-like, or something stranger?
- **G3 — perturbation transfer:** ask whether one perturbation changes the field so another perturbation is amplified more strongly later — the toy parent→child cascade idea.
- **G4 — slow substrate:** allow the response geometry itself to change slowly and ask which compressed constraints prevent runaway self-amplification.

The repo succeeds if those gates kill the seductive story quickly or turn it into a measurable mechanism.
