# Kompressori prior-art kill ledger — September 2026

This file asks which claims survive a hostile comparison with established perturbation, Green-function and multiple-scattering theory.

`NOT KILLED IN THIS PASS` means only that this exact empirical test was not located. It is **not** a priority claim.

## Bottom line

The broad conceptual story is not new:

```text
localized event
    -> low-rank/local perturbation of a response operator
    -> propagated interaction with another event
    -> non-additive joint response
```

Finite-rank perturbation theory, Woodbury/Dyson identities, Foldy-Lax/T-matrix multiple scattering, and localized-defect Green-function theory already contain that architecture.

Kompressori's remaining value is empirical and narrower: it has a nonlinear state-changing toy field in which the **full sampled response operator is high-rank while finite local experience induces a much lower-rank response update**, and it can now test whether an independently measured propagated local coupling predicts later finite update interference better than simpler baselines.

## Kill table

| Kompressori claim | Prior art | Verdict |
|---|---|---|
| A local perturbation can change a response/Green operator by a low-rank update | Sherman-Morrison-Woodbury / finite-rank defect theory; Dutta et al. 2018 explicitly model a localized mutation as a small-rank defect Hamiltonian and compute the Green-function change with Woodbury | **KILLED as conceptual novelty** |
| Two local perturbations can interact non-additively through the response propagator | Dyson / multiple-scattering expansions; Dutta et al. identify two-mutation epistasis with Green-function nonlinearity and paths that visit both defects | **KILLED** |
| Local scatterers can be represented by a small interaction matrix coupled through propagated Green functions | Foldy-Lax and T-matrix multiple scattering | **KILLED** |
| Separated local perturbations may add while strongly coupled/overlapping perturbations interfere | generic locality plus multiple-scattering / defect interaction theory | **KILLED as broad principle** |
| The whole response operator may be high-rank while a particular local update is low-rank | low-rank perturbations of high-dimensional operators are standard; the measured numerical contrast in this toy field remains a legitimate result | **Not a new principle; retained as an empirical property of this model** |
| Future-state fidelity need not imply perturbation-response fidelity | observability, system identification, robust control and operator-learning literatures already distinguish trajectory/state approximation from input-output response approximation | **KILLED as general concept** |
| A few sampled response probes can overfit the operator | standard finite-sample/system-identification limitation | **KILLED as general concept** |
| A small-probe propagated coupling measured around the base trajectory predicts later finite nonlinear non-additivity better than Euclidean distance or static DeltaJ-subspace overlap | No exact matching result located in this pass; closely related to Green-function coupling / multiple-scattering path strengths | **NOT KILLED IN THIS PASS; empirical predictor question only** |

## Particularly close prior art

### Dutta et al. 2018

Dutta, Eckmann, Libchaber and Tlusty, *Green function of correlated genes in a minimal mechanical model of protein evolution*, PNAS 115, E4559–E4568 (2018), doi:10.1073/pnas.1716215115.

This paper is unusually close to Gates 6–7 in mathematical architecture:

- a mutation changes only a small local set of bonds;
- the associated defect Hamiltonian is therefore low rank;
- the perturbed Green function is evaluated with Woodbury;
- the Dyson expansion is interpreted as repeated scattering from the defect;
- the non-additivity of two mutations is called epistasis;
- long-range epistasis is written as multiple-scattering paths containing both local perturbations.

Therefore Kompressori should **not** claim that low-rank local operator changes or propagated non-additivity were discovered here.

### Foldy-Lax / T-matrix multiple scattering

Multiple-scattering methods solve mutual interactions among local scatterers by propagating fields through Green functions and solving a coupled finite-dimensional interaction system. This is a close structural ancestor of the `CausalHorizon` event-space matrix.

Useful examples:

- K. Green & K. Lumme, *Multiple scattering by the iterative Foldy-Lax scheme*, JOSA A 22, 1555–1558 (2005), doi:10.1364/JOSAA.22.001555.
- Generalized Foldy-Lax formulations couple point/extended scatterers through free-space Green-function propagation.

## What remains worth testing

Gate 7 currently selects only nearest and farthest pairs. It shows a spectacular distance contrast but cannot tell whether distance is the actual causal coordinate.

The next experiment therefore does **not** ask whether updates interfere. That is already measured and well precedented.

It asks whether the nonlinear field exposes a better *predictor* of that interference.

For localized packet `p_i`, use a tiny outcome-independent probe around the base state and measure how its delayed state response projects onto another packet `p_j`:

```math
\widehat\omega_{j\leftarrow i}
=
\frac{\langle (\delta\phi_i,\delta\phi^{old}_i),(p_j,p_j)\rangle}
     {\|(p_j,p_j)\|^2}.
```

This is a nonlinear-base, finite-difference analogue of a propagated coupling coefficient. It is **not** asserted to satisfy the exact CausalHorizon theorem.

For simultaneous events, freeze before outcome measurement the symmetric predictor

```math
P_{prop}(i,j)=\max(|\widehat\omega_{j\leftarrow i}|,|\widehat\omega_{i\leftarrow j}|).
```

Then compare it on outcome-independent held-out pairs against:

1. inverse physical distance;
2. cosine similarity of the individual whitened `DeltaJ` operators;
3. overlap of their dominant right singular subspaces.

The target is Gate-7-style finite non-additivity.

If propagated coupling does not beat those simple baselines, the `CausalHorizon -> Kompressori` handoff is killed as a useful empirical predictor even though the exact linear theorem remains true.

## Claim language after this audit

Reasonable:

> In this toy nonlinear field, a 100-direction sampled response operator is high-rank, while the response update caused by one local finite event is much lower-rank; nearby finite updates interfere strongly and distant ones add almost exactly.

> We test whether a separately measured small-probe propagated coupling predicts that finite interference better than physical distance or static update-subspace overlap.

Not reasonable:

> We discovered that local perturbations make low-rank operator updates.

> We discovered that low-rank updates can interfere.

> We discovered a new multiple-scattering interaction matrix.

> CausalHorizon proves the nonlinear Kompressori behavior.

## References

- Dutta S. et al. **Green function of correlated genes in a minimal mechanical model of protein evolution.** PNAS 115, E4559–E4568 (2018). doi:10.1073/pnas.1716215115
- Green K., Lumme K. **Multiple scattering by the iterative Foldy-Lax scheme.** J. Opt. Soc. Am. A 22, 1555–1558 (2005). doi:10.1364/JOSAA.22.001555
- Sherman J., Morrison W.J.; Woodbury M.A. — classical low-rank inverse/update identities.
