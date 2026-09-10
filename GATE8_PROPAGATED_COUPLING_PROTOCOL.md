# Gate 8 — propagated-coupling predictor contest

Status: **frozen before the first Gate-8 outcome run**.

Date: 2026-09-10

## Question

Gate 7 already established, in this toy field, that the closest finite local events can produce strongly non-additive response-operator updates while the farthest tested events add almost exactly.

That is not a new physical principle. Multiple-scattering and localized-defect theory already provide much of that architecture.

The narrower question is:

> Does an independently measured **propagated local coupling** around the unmodified base trajectory predict finite nonlinear update interference better than physical distance or static overlap of the individual low-rank `Delta J` updates?

This is a predictor contest, not a claim that the nonlinear field obeys the exact CausalHorizon theorem.

## Locked simulation

Reuse the Gate-6/7 configuration unchanged:

```text
grid             40 x 40 periodic
target_steps      350
A locations       first 30 deterministic Gate-5 sites
delay             24
B-response horizon 25
B packet grid      10 x 10 = 100 sampled directions
epsilon_A finite   0.1
epsilon_B          0.0005
```

The same input Gram whitening used by Gates 6–7 is used for all `Delta J` comparisons.

## Held-out pair panel

There are `30 choose 2 = 435` possible event pairs.

Gate 7 already inspected the **8 closest and 8 farthest** pairs. Gate 8 excludes those 16 pairs entirely.

Sort the remaining 419 pairs by `(torus distance, i, j)` and choose **48 pairs at evenly spaced rank positions** from the first through last remaining pair. Pair selection therefore depends only on geometry and the previously declared Gate-7 exclusion, never on Gate-8 non-additivity or any predictor value.

These 48 pairs are the frozen outcome panel.

## Outcome

For each held-out pair `(i,j)`, compute exactly the Gate-7 finite non-additivity:

```math
N_{ij}=
\frac{\|\Delta J_{ij}-(\Delta J_i+\Delta J_j)\|_W}
     {\|\Delta J_{ij}\|_W},
```

where `W` is the Gate-6 input whitening.

No pair is removed based on its outcome.

## Four predictors fixed before outcomes

### P0 — physical locality

```math
P_{dist}(i,j)=\frac{1}{1+d_{torus}(i,j)}.
```

This is the baseline Gate 7 implicitly favored.

### P1 — static operator cosine

For each finite single-event update, whiten the sampled operator:

```text
D_i = DeltaJ_i W.
```

Then

```math
P_{cos}(i,j)=
\frac{|\langle D_i,D_j\rangle_F|}
     {\|D_i\|_F\|D_j\|_F}.
```

### P2 — dominant input-subspace overlap

From `D_i^T D_i`, retain the smallest right-singular subspace containing at least **90%** of `D_i`'s spectral energy. If its orthonormal basis is `V_i`, define

```math
P_{sub}(i,j)=
\frac{\|V_i^T V_j\|_F^2}
     {\min(r_i,r_j)}.
```

This is a static low-rank-update overlap baseline.

### P3 — propagated small-probe coupling

Before measuring pair non-additivity, estimate how a tiny packet at one event site reaches the other event site after the same 24-step preparation interval.

Use the already-declared tiny amplitude

```text
epsilon_probe = 0.002
```

and for event `i` compute

```text
(delta_phi_i, delta_old_i)
 = [advance(base + eps*p_i) - advance(base)] / eps.
```

Project that delayed two-state perturbation onto the local packet direction at `j`:

```math
\widehat\omega_{j\leftarrow i}
=
\frac{\langle\delta\phi_i,p_j\rangle+\langle\delta old_i,p_j\rangle}
     {2\langle p_j,p_j\rangle}.
```

Because the finite Gate-8 events are simultaneous rather than strictly time ordered, freeze the symmetric primary predictor as

```math
P_{prop}(i,j)=
\max\bigl(|\widehat\omega_{j\leftarrow i}|,
          |\widehat\omega_{i\leftarrow j}|\bigr).
```

The directed `omega` values are retained in the receipt.

## Score

No fitted regression is used.

For each predictor, compute a tie-aware **Spearman rank correlation** with finite non-additivity across the 48 held-out pairs.

Primary classification:

```text
PROPAGATED_COUPLING_WINS
```

only if

1. `rho_prop >= 0.50`, and
2. `rho_prop >= max(rho_distance, rho_operator_cosine, rho_subspace) + 0.10`.

Otherwise:

```text
PROPAGATED_COUPLING_NOT_PRIVILEGED
```

The `+0.10` margin is frozen to require a practically visible improvement rather than declaring victory over tiny rank-correlation noise.

## Secondary diagnostics

Always report:

- all four Spearman coefficients;
- pair-level predictor/outcome table;
- top-12 pairs according to each predictor and their mean non-additivity;
- the two directed `omega` values;
- single-update 90%-energy ranks;
- Pearson correlation between physical distance and propagated coupling, to show whether `P_prop` is merely distance in disguise.

No secondary diagnostic can change the primary classification.

## Interpretation

If propagated coupling wins, the earned statement is only:

> In this nonlinear toy field, a small-probe measurement of how one local perturbation propagates toward another predicts later finite response-update interference better than physical distance or static low-rank-update overlap on a held-out geometry-selected pair panel.

If it does not win, the empirical `CausalHorizon -> Kompressori` handoff is killed as a privileged predictor. The exact CausalHorizon linear identity remains true, but it has not bought predictive power here.
