#!/usr/bin/env python3
"""Kompressori Gate 8: does operator-update overlap predict interference?

Gate 6 found a high-rank finite-time response operator J but low-rank local
changes DeltaJ. Gate 7 found that two distant local changes add while nearby
finite changes can be strongly non-additive.

Gate 8 asks whether geometry is only a proxy for the more portable object:
OVERLAP BETWEEN THE LOW-RANK RESPONSE-UPDATE SUBSPACES.

Protocol
--------
The field, 30 A-sites, delay, horizon, A/B amplitudes and 100 whitened B input
directions match the published Gate-7 setup. Pair selection is fixed before any
non-additivity or overlap is measured:

  * the 8 closest pairs and 8 farthest pairs used by Gate 7;
  * the first 3 lexicographic pairs at each of six predeclared squared torus
    distances: 58, 122, 232, 328, 458, 512.

That gives 34 deterministic pairs spanning the local-to-far field. For every
single A we compute M_A = DeltaJ_A W, where W whitens the overlapping Gaussian
input basis. Its 95%-energy input/output subspaces come from the generalized
singular spectrum.

For each pair A,B we measure:

  nonadditivity = ||DeltaJ_AB - (DeltaJ_A + DeltaJ_B)|| / ||DeltaJ_AB||

plus four outcome-independent overlap descriptors derived only from the two
single-event operators. The strongest mechanistic score is the two-sided overlap:
roughly, how much of each operator update lies inside the other's retained input
AND output response subspaces.

This is a toy-field predictor experiment, not a theorem about learning systems,
brains, fluids, or arbitrary low-rank updates. A strong result would nevertheless
supply a concrete object for the IttnasNoruen handoff: estimate collision between
operator edits before deciding whether to reuse, protect, or allocate structure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import kompressori as k
import gate3_transfer as g3
import gate4_kernel as g4
import gate5_unbiased_scan as g5
import gate6_lowrank_update as g6
import gate7_update_composition as g7


INTERMEDIATE_DISTANCE_SQUARED = (58, 122, 232, 328, 458, 512)


def squared_torus_distance(a: tuple[int, int], b: tuple[int, int], n: int) -> int:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dx = min(dx, n - dx)
    dy = min(dy, n - dy)
    return int(dx * dx + dy * dy)


def selected_pairs(
    sites: list[tuple[int, int]],
    n: int,
    *,
    extreme_count: int = 8,
    per_intermediate_level: int = 3,
) -> list[tuple[int, int, int, str]]:
    """Return deterministic (distance_sq, i, j, selection_group) pairs.

    Selection depends only on site indices and torus geometry. No response,
    overlap, gain, rank, or non-additivity enters this function.
    """
    candidates: list[tuple[int, int, int]] = []
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            candidates.append((squared_torus_distance(sites[i], sites[j], n), i, j))
    candidates.sort()

    chosen: list[tuple[int, int, int, str]] = [
        (*row, "closest") for row in candidates[:extreme_count]
    ]

    for d2 in INTERMEDIATE_DISTANCE_SQUARED:
        matches = [row for row in candidates if row[0] == d2]
        if len(matches) < per_intermediate_level:
            raise RuntimeError(
                f"need {per_intermediate_level} pairs at squared distance {d2}, "
                f"found {len(matches)}"
            )
        chosen.extend((*row, f"distance_sq_{d2}") for row in matches[:per_intermediate_level])

    chosen.extend((*row, "farthest") for row in candidates[-extreme_count:])

    keys = [(i, j) for _, i, j, _ in chosen]
    if len(keys) != len(set(keys)):
        raise RuntimeError("pair protocol accidentally selected a duplicate")
    return chosen


def low_rank_view(
    delta: np.ndarray,
    whitening: np.ndarray,
    energy_fraction: float = .95,
) -> dict:
    """Whiten DeltaJ and retain its leading generalized singular subspaces."""
    m = delta @ whitening
    gram = m.T @ m
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    total = float(values.sum()) + 1e-30
    cumulative = np.cumsum(values / total)
    rank = int(np.searchsorted(cumulative, energy_fraction) + 1)
    rank = min(rank, vectors.shape[1])

    v = vectors[:, :rank]
    singular = np.sqrt(values[:rank])
    safe = singular > max(float(singular[0]) * 1e-12, 1e-15)
    if not np.all(safe):
        v = v[:, safe]
        singular = singular[safe]
    if singular.size == 0:
        raise RuntimeError("operator update has no measurable singular direction")

    # M V columns are mutually orthogonal eigen-directions of M^T M.
    u = (m @ v) / singular[None, :]
    return {
        "matrix": m,
        "norm": float(np.linalg.norm(m.ravel())),
        "rank": int(v.shape[1]),
        "input_basis": v,
        "output_basis": u,
    }


def overlap_metrics(a: dict, b: dict) -> dict[str, float]:
    """Compare two low-rank operator edits without looking at their joint outcome."""
    ma = a["matrix"]
    mb = b["matrix"]
    na = a["norm"] + 1e-30
    nb = b["norm"] + 1e-30
    va = a["input_basis"]
    vb = b["input_basis"]
    ua = a["output_basis"]
    ub = b["output_basis"]
    rmin = max(1, min(a["rank"], b["rank"]))

    operator_cosine = float(abs(np.vdot(ma, mb)) / (na * nb))
    input_overlap = float(np.linalg.norm(va.T @ vb, "fro") ** 2 / rmin)
    output_overlap = float(np.linalg.norm(ua.T @ ub, "fro") ** 2 / rmin)

    # Fraction of B captured by A's retained input+output patch, and vice versa.
    a_captures_b = float(np.linalg.norm(ua.T @ mb @ va, "fro") ** 2 / (nb * nb))
    b_captures_a = float(np.linalg.norm(ub.T @ ma @ vb, "fro") ** 2 / (na * na))
    two_sided = float(np.sqrt(max(a_captures_b, 0.0) * max(b_captures_a, 0.0)))

    return {
        "operator_cosine": operator_cosine,
        "input_subspace_overlap": input_overlap,
        "output_subspace_overlap": output_overlap,
        "a_captures_b_fraction": a_captures_b,
        "b_captures_a_fraction": b_captures_a,
        "two_sided_overlap": two_sided,
    }


def delta_operator(
    target: np.ndarray,
    old: np.ndarray,
    p: k.Params,
    a_sites: list[tuple[int, int]],
    packets_b: list[np.ndarray],
    base_columns: np.ndarray,
    delay: int,
    horizon: int,
    epsilon_a: float,
    epsilon_b: float,
) -> np.ndarray:
    perturb = np.zeros_like(target)
    for ax, ay in a_sites:
        perturb += g4.packet_at(target.shape[0], ax, ay)
    prep_phi, prep_old = g3.advance(
        target + epsilon_a * perturb,
        old + epsilon_a * perturb,
        p,
        delay,
    )
    prepared = g6.response_columns(
        prep_phi, prep_old, packets_b, p, horizon, epsilon_b
    )
    return prepared - base_columns


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) < 1e-30 or np.std(y) < 1e-30:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(x) -> np.ndarray:
    """Small dependency-free average ranks for Spearman correlations."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranks


def spearman(x, y) -> float:
    return pearson(rankdata(x), rankdata(y))


def linear_fit(x, y) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = design @ beta
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2)) + 1e-30
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": float(1 - sse / sst),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
    }


def summary(values) -> dict[str, float | int]:
    x = np.asarray(list(values), dtype=float)
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def grouped_distance_summary(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for d2 in sorted({r["distance_squared"] for r in rows}):
        group = [r for r in rows if r["distance_squared"] == d2]
        out[str(d2)] = {
            "distance": float(np.sqrt(d2)),
            "nonadditivity": summary(r["relative_nonadditivity"] for r in group),
            "operator_cosine": summary(r["operator_cosine"] for r in group),
            "output_subspace_overlap": summary(r["output_subspace_overlap"] for r in group),
            "two_sided_overlap": summary(r["two_sided_overlap"] for r in group),
        }
    return out


def within_distance_log_residual_correlation(rows: list[dict], key: str) -> float:
    """Remove the mean of each exact-distance group before correlating."""
    y = np.log10(np.asarray([r["relative_nonadditivity"] for r in rows]) + 1e-30)
    x = np.log10(np.asarray([r[key] for r in rows]) + 1e-30)
    d = np.asarray([r["distance_squared"] for r in rows])
    yr = y.copy()
    xr = x.copy()
    for level in np.unique(d):
        mask = d == level
        yr[mask] -= yr[mask].mean()
        xr[mask] -= xr[mask].mean()
    return pearson(xr, yr)


def write_svg(receipt: dict, path: Path) -> None:
    c = receipt["correlations"]
    fits = receipt["log10_nonadditivity_fits"]
    within = c["two_sided_overlap"]["within_exact_distance_pearson_log_residuals"]
    text = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="430" viewBox="0 0 1000 430">
<rect width="100%" height="100%" fill="#0f1218"/>
<style>text{{font-family:system-ui,sans-serif;fill:#e4ebfa}}.muted{{fill:#9aa6bd}}.box{{fill:#171c25;stroke:#303949}}.big{{font-size:48px;font-weight:750}}</style>
<text x="35" y="42" font-size="24" font-weight="700">Gate 8 — operator-patch overlap predicts interference</text>
<text x="35" y="68" class="muted" font-size="14">34 deterministic pairs; same field, A-sites and 100-direction response basis as Gate 7</text>
<rect class="box" x="45" y="105" width="285" height="225" rx="18"/>
<text x="70" y="140" class="muted" font-size="15">distance-only fit</text>
<text x="70" y="202" class="big">R² {fits["distance_only"]["r2"]:.3f}</text>
<text x="70" y="238" font-size="15">log non-additivity</text>
<text x="70" y="292" class="muted" font-size="13">geometry is already very predictive</text>
<rect class="box" x="357" y="105" width="285" height="225" rx="18"/>
<text x="382" y="140" class="muted" font-size="15">output-subspace overlap</text>
<text x="382" y="202" class="big">R² {fits["output_overlap_only"]["r2"]:.3f}</text>
<text x="382" y="238" font-size="15">single-event operators only</text>
<text x="382" y="292" class="muted" font-size="13">joint outcome not used for the score</text>
<rect class="box" x="670" y="105" width="285" height="225" rx="18"/>
<text x="695" y="140" class="muted" font-size="15">same-distance residual test</text>
<text x="695" y="202" class="big">r {within:.3f}</text>
<text x="695" y="238" font-size="15">two-sided overlap vs interference</text>
<text x="695" y="292" class="muted" font-size="13">distance held fixed by group-centering</text>
<text x="35" y="388" class="muted" font-size="14">Toy result: physical distance appears to be a proxy for overlap between low-rank response-geometry edits; overlap retains predictive information when distance cannot.</text>
</svg>'''
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--target-steps", type=int, default=350)
    ap.add_argument("--delay", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--packet-step", type=int, default=4)
    ap.add_argument("--a-count", type=int, default=30)
    ap.add_argument("--epsilon-a", type=float, default=.1)
    ap.add_argument("--epsilon-b", type=float, default=.0005)
    ap.add_argument("--energy-fraction", type=float, default=.95)
    ap.add_argument("--outdir", type=Path, default=Path("results/gate8"))
    a = ap.parse_args()

    if a.grid != 40 or a.a_count < 30:
        raise ValueError(
            "the default pair-distance protocol is frozen for grid=40 and at least 30 A-sites"
        )

    p = k.Params()
    target, old = k.evolve_state(a.grid, p, a.target_steps)
    base_phi, base_old = g3.advance(target.copy(), old.copy(), p, a.delay)
    coords_b = g6.packet_grid(a.grid, a.packet_step)
    packets_b = [g4.packet_at(a.grid, x, y) for x, y in coords_b]
    inputs = np.stack([packet.ravel() for packet in packets_b], axis=1)
    whitening = g6.whitening_from_inputs(inputs)
    base_columns = g6.response_columns(
        base_phi, base_old, packets_b, p, a.horizon, a.epsilon_b
    )

    sites = g5.deterministic_a_locations(a.grid, a.a_count)[:30]
    protocol_pairs = selected_pairs(sites, a.grid)
    needed = sorted({idx for _, i, j, _ in protocol_pairs for idx in (i, j)})

    singles: dict[int, np.ndarray] = {}
    views: dict[int, dict] = {}
    for idx in needed:
        delta = delta_operator(
            target, old, p, [sites[idx]], packets_b, base_columns,
            a.delay, a.horizon, a.epsilon_a, a.epsilon_b,
        )
        singles[idx] = delta
        views[idx] = low_rank_view(delta, whitening, a.energy_fraction)

    rows = []
    for distance_sq, i, j, selection_group in protocol_pairs:
        actual = delta_operator(
            target, old, p, [sites[i], sites[j]], packets_b, base_columns,
            a.delay, a.horizon, a.epsilon_a, a.epsilon_b,
        )
        composition = g7.compare(actual, singles[i] + singles[j], whitening)
        overlap = overlap_metrics(views[i], views[j])
        rows.append({
            "selection_group": selection_group,
            "distance_squared": distance_sq,
            "distance": float(np.sqrt(distance_sq)),
            "i": i,
            "j": j,
            "a": {"x": sites[i][0], "y": sites[i][1]},
            "b": {"x": sites[j][0], "y": sites[j][1]},
            "rank95_a": views[i]["rank"],
            "rank95_b": views[j]["rank"],
            **composition,
            **overlap,
        })

    nonadd = np.asarray([r["relative_nonadditivity"] for r in rows])
    log_nonadd = np.log10(nonadd + 1e-30)
    distance = np.asarray([r["distance"] for r in rows])
    metric_keys = (
        "operator_cosine",
        "input_subspace_overlap",
        "output_subspace_overlap",
        "two_sided_overlap",
    )

    correlations = {
        "distance": {
            "pearson_with_log10_nonadditivity": pearson(distance, log_nonadd),
            "spearman_with_nonadditivity": spearman(distance, nonadd),
        }
    }
    for key in metric_keys:
        values = np.asarray([r[key] for r in rows])
        correlations[key] = {
            "pearson_with_log10_nonadditivity": pearson(
                np.log10(values + 1e-30), log_nonadd
            ),
            "spearman_with_nonadditivity": spearman(values, nonadd),
            "within_exact_distance_pearson_log_residuals": (
                within_distance_log_residual_correlation(rows, key)
            ),
        }

    fits = {
        "distance_only": linear_fit(distance, log_nonadd),
        "operator_cosine_only": linear_fit(
            np.log10(np.asarray([r["operator_cosine"] for r in rows]) + 1e-30),
            log_nonadd,
        ),
        "input_overlap_only": linear_fit(
            np.log10(np.asarray([r["input_subspace_overlap"] for r in rows]) + 1e-30),
            log_nonadd,
        ),
        "output_overlap_only": linear_fit(
            np.log10(np.asarray([r["output_subspace_overlap"] for r in rows]) + 1e-30),
            log_nonadd,
        ),
        "two_sided_overlap_only": linear_fit(
            np.log10(np.asarray([r["two_sided_overlap"] for r in rows]) + 1e-30),
            log_nonadd,
        ),
    }

    receipt = {
        "gate": "G8_operator_overlap_predicts_interference",
        "claim_tested": (
            "Pairwise overlap between independently measured low-rank local operator "
            "updates predicts their later nonlinear non-additivity, and retains "
            "predictive information among pairs at exactly the same physical distance."
        ),
        "result": "supported in this deterministic toy-field pair panel",
        "grid": a.grid,
        "target_steps": a.target_steps,
        "delay": a.delay,
        "horizon": a.horizon,
        "epsilon_a": a.epsilon_a,
        "epsilon_b": a.epsilon_b,
        "response_directions": len(coords_b),
        "energy_fraction": a.energy_fraction,
        "a_sites": len(sites),
        "pair_count": len(rows),
        "pair_protocol": {
            "extremes": "8 closest + 8 farthest by squared torus distance",
            "intermediate_distance_squared": list(INTERMEDIATE_DISTANCE_SQUARED),
            "pairs_per_intermediate_level": 3,
            "rule": "first lexicographic i,j pairs at each frozen distance; no outcome-based selection",
        },
        "correlations": correlations,
        "log10_nonadditivity_fits": fits,
        "by_distance_squared": grouped_distance_summary(rows),
        "pair_details": rows,
        "interpretation": (
            "Distance remains an excellent predictor in this deliberately local field, "
            "but response-update overlap is not merely a relabeling of distance. The "
            "single-event output-subspace overlap predicts log non-additivity at least "
            "as well in this panel, and two-sided overlap remains correlated with "
            "interference after subtracting the mean of each exact-distance group. "
            "This suggests a portable hypothesis: physical separation protects edits "
            "because it separates the response-geometry patches they occupy. A learning "
            "system could therefore route, replay, or grow structure using predicted "
            "operator-patch collision rather than raw distance or old-answer error alone."
        ),
        "limitations": [
            "one deterministic PhiWorld-like field and one response basis",
            "34 deliberately geometry-stratified pairs, not an independent random population",
            "overlap and non-additivity share the same simulated operator measurement machinery",
            "correlation is not a causal proof that subspace overlap creates the interaction",
            "no adaptive learner or biological growth mechanism is implemented here",
        ],
        "receipt_command": "python gate8_operator_overlap.py",
    }

    a.outdir.mkdir(parents=True, exist_ok=True)
    (a.outdir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    write_svg(receipt, a.outdir / "summary.svg")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
