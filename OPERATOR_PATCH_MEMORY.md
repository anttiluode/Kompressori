# Operator-patch memory — the positive hypothesis hiding in Kompressori

The useful object emerging from Gates 6–8 is not a stored answer and not the whole
response operator.

It is a **small edit to the operator through which the next perturbation will
travel**.

Write the finite-time response operator around the present state as

```math
J : \delta q \mapsto \delta y_{0:T}.
```

A local finite experience `e` changes it:

```math
J \longrightarrow J + \Delta J_e.
```

Gate 6 found that `J` is high-rank in the sampled 100-direction basis while a
single local `Delta J_e` is much lower-rank. Gate 7 found that separated edits add
while nearby finite edits interact. Gate 8 now finds that **overlap between the
single-event low-rank response subspaces predicts that later interaction**, even
among pairs at exactly the same physical distance.

That suggests a concrete memory model:

```math
\Delta J_e \approx U_e S_e V_e^T.
```

`V_e` is the small set of perturbation/cue directions to which the edit is
sensitive. `U_e` is the small set of future response directions in which its
consequence appears. `S_e` says how strongly those directions are coupled.

The phrase **operator patch** means that triple `(U_e, S_e, V_e)` plus whatever
physical structure realizes it.

## Why this is more useful than "protect old answers"

Suppose two systems still answer a stored question correctly. They can nevertheless
have different `J`, so the next incomplete cue or perturbation can be routed
differently. That is the exact gap exposed by the IttnasNoruen / Kompressori
counterexample.

A point answer constrains `f(q)`.

An operator patch constrains a local family of counterfactuals:

```text
what happens if the next cue moves this way?
what happens if another event arrives here?
which future directions are amplified or suppressed?
```

The patch is therefore a candidate unit for preserving **access**, not merely
preserving a previously observed output.

## Interference becomes a collision problem

For two learned edits `a` and `b`, define an overlap score from their independently
measured low-rank factors. Gate 8 uses input overlap, output overlap and a two-sided
score that asks how much of each edit lies inside the other's retained input and
output subspaces.

The new empirical pattern is:

```text
small patch overlap  -> edits approximately superpose
large patch overlap  -> finite nonlinear interaction becomes likely
```

Physical distance predicts this in PhiWorld because locality separates the patches.
But Gate 8's same-distance residual test shows that distance is not the whole
story. In the frozen 34-pair panel, output-subspace overlap gives `R^2 ~= 0.992`
for log non-additivity versus `R^2 ~= 0.974` for distance alone, and two-sided
overlap remains correlated with interference after exact-distance means are removed
(`r ~= 0.837`).

Those numbers belong only to this deterministic toy panel. The important object is
the testable relation, not the decimal places.

## This gives structural growth a job

The Gate-8 growth experiment in IttnasNoruen currently waits until ordinary
behavioral constraints block a proposed update. That is probably the wrong trigger
if this hypothesis is right.

Growth should happen when the learner predicts:

```math
\text{benefit of new edit} \gg 0
```

but every available realization of that edit has excessive collision with
protected operator patches.

Then new structure is not "extra parameters because the old network is full."

It is **a new place in response geometry where the edit can live with less
interference**.

That makes the initially-large control and the growing system meaningfully
different. A large unstructured network can have enough numerical capacity and
still write new experience into an overlapping patch. Growth has an advantage only
if its *placement* is informed by predicted collision.

## Reuse versus separation

Overlap is not automatically bad.

If two experiences should transfer to or recall one another, reusing an existing
operator patch can be desirable. If they require conflicting responses, separation
is desirable.

So the allocation problem is closer to:

```text
related experience?     reuse / overlap / link
conflicting experience? split / orthogonalize / grow
uncertain relation?     probe before committing
```

This turns catastrophic forgetting and memory linking into two sides of the same
allocation variable.

## Retrieval becomes projection into a patch

An incomplete cue `c` can be viewed as a perturbation direction. Its ability to
reach memory `e` depends partly on its projection into the patch's input subspace:

```math
\|V_e^T c\|.
```

One cue can therefore fail while another reaches the same underlying learned edit.
That is a limited mathematical version of the face/name intuition that motivated
IttnasNoruen: the answer need not have vanished merely because one route into it
has weak coupling.

A retrieval policy should not ask only "am I confident?" It should ask which
available next cue is expected to illuminate a different operator patch.

## Replay becomes collision-targeted measurement

A bounded learner cannot replay everything. If a candidate new edit has sketch
`Delta J_new`, spend the replay budget on old patches with the largest predicted
collision:

```math
score(e, new) = overlap(\Delta J_e, \Delta J_new).
```

This is more specific than replaying memories near a classification boundary. It
asks which old *response geometries* the proposed change is positioned to alter.

The guard still evaluates actual finite consequences before committing. The
low-rank sketch is a measurement-allocation heuristic, not a certificate.

## The next cross-repo experiment

The positive test should now move into IttnasNoruen.

1. Learn several old tasks or cue routes and measure compact `Delta J_e` sketches.
2. Propose a new learned edit.
3. Rank old memories by predicted operator-patch collision using no held-out label.
4. Give every method the same replay/probe budget.
5. Compare collision-targeted replay with random replay, answer-boundary replay and
   ordinary loss-based interfered retrieval at **matched new-task progress**.
6. Measure held-out cue-route survival and held-out local response geometry.
7. When all existing routes collide, compare collision-triggered structural
   allocation with the same final capacity supplied from the start.

A win there would be substantially stronger than another counterexample. It would
show that a quantity discovered in a nonlinear field can actually guide learning.

## What is and is not novel here

Low-rank adaptation, orthogonal task subspaces, gradient-subspace protection and
interference-aware replay already exist in machine learning. Dendritic clustering,
branch-specific plasticity and memory linking also have substantial neuroscience
literature.

The possible contribution of this project is narrower and testable:

> **Measure experience as a low-rank edit of future response geometry; predict
> interference from overlap between those edits; use that collision estimate to
> choose replay, routing and structural allocation.**

The project should claim no more until the adaptive cross-repo test works.
