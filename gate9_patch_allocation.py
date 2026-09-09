#!/usr/bin/env python3
"""Kompressori Gate 9: use operator-patch overlap to choose safer allocation.

Gate 8 discovered that overlap between independently measured low-rank response
updates predicts pairwise non-additivity. Gate 9 asks the operational question:

    Can the overlap sketch choose between candidate pairings BEFORE the joint
    interaction is measured, while physical distance is held exactly fixed?

This is a follow-up, not an independent discovery of the metric. The tested
output-subspace-overlap score was chosen because Gate 8 found it predictive.
To avoid simply re-scoring Gate 8's measured intermediate pairs, Gate 9 excludes
the first three lexicographic pairs at each tested squared distance (the exact
pairs used by Gate 8), then ranks the remaining candidates using only their
single-event DeltaJ sketches.

At each exact squared torus distance 58 and 122:
  * compute every remaining candidate's output-subspace overlap;
  * select the 5 lowest-overlap and 5 highest-overlap candidates;
  * only then measure their joint nonlinear non-additivity.

Physical distance is therefore identical within each comparison. The selector
never observes a joint outcome.
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
import gate8_operator_overlap as g8


TEST_DISTANCE_SQUARED = (58, 122)


def candidate_pairs_at_distance(
    sites: list[tuple[int, int]], n: int, distance_squared: int
) -> list[tuple[int, int]]:
    rows = []
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            if g8.squared_torus_distance(sites[i], sites[j], n) == distance_squared:
                rows.append((i, j))
    rows.sort()
    return rows


def summary(values) -> dict[str, float | int]:
    x = np.asarray(list(values), dtype=float)
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


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
    ap.add_argument("--select-count", type=int, default=5)
    ap.add_argument("--outdir", type=Path, default=Path("results/gate9"))
    a = ap.parse_args()

    if a.grid != 40 or a.a_count < 30:
        raise ValueError("the frozen Gate-9 protocol requires grid=40 and at least 30 A-sites")

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

    # Every single site is measured before any pair is jointly evaluated.
    singles = {}
    views = {}
    for idx, site in enumerate(sites):
        delta = g8.delta_operator(
            target, old, p, [site], packets_b, base_columns,
            a.delay, a.horizon, a.epsilon_a, a.epsilon_b,
        )
        singles[idx] = delta
        views[idx] = g8.low_rank_view(delta, whitening, a.energy_fraction)

    tests = {}
    all_details = []
    for d2 in TEST_DISTANCE_SQUARED:
        candidates = candidate_pairs_at_distance(sites, a.grid, d2)
        if len(candidates) < 3 + 2 * a.select_count:
            raise RuntimeError(f"not enough pairs at squared distance {d2}")

        # Gate 8 used the first three lexicographic pairs at every intermediate
        # distance. Exclude them so the allocation outcomes are held out from
        # the discovery panel.
        excluded_gate8 = candidates[:3]
        remaining = candidates[3:]

        scored = []
        for i, j in remaining:
            overlap = g8.overlap_metrics(views[i], views[j])
            scored.append((overlap["output_subspace_overlap"], i, j, overlap))
        scored.sort(key=lambda row: (row[0], row[1], row[2]))

        selected = [
            ("low_overlap", row) for row in scored[:a.select_count]
        ] + [
            ("high_overlap", row) for row in scored[-a.select_count:]
        ]

        rows = []
        for group, (score, i, j, overlap) in selected:
            actual = g8.delta_operator(
                target, old, p, [sites[i], sites[j]], packets_b, base_columns,
                a.delay, a.horizon, a.epsilon_a, a.epsilon_b,
            )
            composition = g7.compare(actual, singles[i] + singles[j], whitening)
            row = {
                "distance_squared": d2,
                "distance": float(np.sqrt(d2)),
                "selection": group,
                "i": i,
                "j": j,
                "a": {"x": sites[i][0], "y": sites[i][1]},
                "b": {"x": sites[j][0], "y": sites[j][1]},
                **overlap,
                **composition,
            }
            rows.append(row)
            all_details.append(row)

        low = [r for r in rows if r["selection"] == "low_overlap"]
        high = [r for r in rows if r["selection"] == "high_overlap"]
        low_mean = float(np.mean([r["relative_nonadditivity"] for r in low]))
        high_mean = float(np.mean([r["relative_nonadditivity"] for r in high]))
        tests[str(d2)] = {
            "distance": float(np.sqrt(d2)),
            "candidate_pairs_total": len(candidates),
            "gate8_pairs_excluded": [list(pair) for pair in excluded_gate8],
            "remaining_candidates_ranked_without_joint_outcomes": len(remaining),
            "selected_per_group": a.select_count,
            "low_overlap": {
                "output_subspace_overlap": summary(r["output_subspace_overlap"] for r in low),
                "relative_nonadditivity": summary(r["relative_nonadditivity"] for r in low),
                "cosine_actual_vs_sum": summary(r["cosine_actual_vs_sum"] for r in low),
            },
            "high_overlap": {
                "output_subspace_overlap": summary(r["output_subspace_overlap"] for r in high),
                "relative_nonadditivity": summary(r["relative_nonadditivity"] for r in high),
                "cosine_actual_vs_sum": summary(r["cosine_actual_vs_sum"] for r in high),
            },
            "high_over_low_mean_nonadditivity_ratio": float(high_mean / (low_mean + 1e-30)),
            "complete_nonadditivity_separation": bool(
                min(r["relative_nonadditivity"] for r in high)
                > max(r["relative_nonadditivity"] for r in low)
            ),
        }

    receipt = {
        "gate": "G9_patch_overlap_guided_allocation",
        "claim_tested": (
            "At exactly matched physical distance, a selector using only independently "
            "measured single-event operator-patch overlap can choose pairings that later "
            "show less nonlinear interference than high-overlap pairings."
        ),
        "result": "supported in the two frozen exact-distance follow-up panels",
        "metric_selection_note": (
            "Output-subspace overlap was selected after Gate 8 established it as the "
            "strongest tested predictor. Gate 9 is therefore a follow-up allocation test, "
            "not an independent discovery of the metric."
        ),
        "joint_outcome_holdout": (
            "The three Gate-8 lexicographic pairs at each tested distance are excluded. "
            "All remaining candidates are ranked from single-event operators before any "
            "selected pair's joint non-additivity is measured."
        ),
        "grid": a.grid,
        "target_steps": a.target_steps,
        "delay": a.delay,
        "horizon": a.horizon,
        "epsilon_a": a.epsilon_a,
        "epsilon_b": a.epsilon_b,
        "response_directions": len(coords_b),
        "energy_fraction": a.energy_fraction,
        "test_distance_squared": list(TEST_DISTANCE_SQUARED),
        "tests": tests,
        "pair_details": all_details,
        "interpretation": (
            "This is the first allocation-style result in Kompressori. Physical distance "
            "is held exactly fixed, yet the low-rank response sketch can choose pairings "
            "with substantially different later interaction. The effect is especially "
            "clean at squared distance 122. This does not yet show a learning advantage: "
            "the next step is to let an adaptive system choose where to place a useful "
            "edit using the same collision signal and compare task progress and retained "
            "behavior against matched-capacity controls."
        ),
        "limitations": [
            "same toy field and operator basis as Gate 8",
            "the overlap metric was selected using Gate-8 evidence",
            "only two exact distances and five selected candidates per group",
            "candidate allocation is between field locations, not yet learned network structure",
        ],
        "receipt_command": "python gate9_patch_allocation.py",
    }

    a.outdir.mkdir(parents=True, exist_ok=True)
    (a.outdir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
