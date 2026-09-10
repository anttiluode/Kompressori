#!/usr/bin/env python3
"""Kompressori Gate 8: does propagated small-probe coupling predict finite update interference?

Protocol: GATE8_PROPAGATED_COUPLING_PROTOCOL.md

This gate deliberately attacks the CausalHorizon handoff.  It compares a
small-probe, dynamically propagated local coupling with three preregistered
baselines on 48 geometry-selected pairs not used by Gate 7.
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


PAIR_COUNT = 48
GATE7_EXTREME_COUNT = 8
PROBE_EPS = 0.002


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks with exact ties sharing one rank."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        end = start + 1
        value = x[order[start]]
        while end < len(x) and x[order[end]] == value:
            end += 1
        rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x - np.mean(x)
    y = y - np.mean(y)
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / den) if den > 1e-30 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(average_ranks(np.asarray(a)), average_ranks(np.asarray(b)))


def heldout_pairs(sites: list[tuple[int, int]], n: int) -> list[dict]:
    candidates = []
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            candidates.append((g7.torus_distance(sites[i], sites[j], n), i, j))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    excluded = {
        (i, j)
        for _, i, j in candidates[:GATE7_EXTREME_COUNT] + candidates[-GATE7_EXTREME_COUNT:]
    }
    remaining = [x for x in candidates if (x[1], x[2]) not in excluded]
    positions = np.rint(np.linspace(0, len(remaining) - 1, PAIR_COUNT)).astype(int)
    if len(set(int(x) for x in positions)) != PAIR_COUNT:
        raise RuntimeError("even-rank pair selector did not produce 48 unique positions")
    selected = []
    for rank_position, pos in enumerate(positions):
        distance, i, j = remaining[int(pos)]
        selected.append(
            {
                "panel_rank": int(rank_position),
                "remaining_pair_rank": int(pos),
                "distance": float(distance),
                "i": int(i),
                "j": int(j),
                "a": {"x": int(sites[i][0]), "y": int(sites[i][1])},
                "b": {"x": int(sites[j][0]), "y": int(sites[j][1])},
            }
        )
    return selected


def dominant_right_subspace(matrix: np.ndarray, energy: float = 0.90) -> tuple[np.ndarray, int]:
    """Right singular subspace containing the requested Frobenius energy."""
    gram = np.asarray(matrix, dtype=float).T @ np.asarray(matrix, dtype=float)
    vals, vecs = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    total = float(np.sum(vals))
    if total <= 1e-30:
        return np.zeros((matrix.shape[1], 0), dtype=float), 0
    cumulative = np.cumsum(vals) / total
    rank = int(np.searchsorted(cumulative, energy) + 1)
    return vecs[:, :rank], rank


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[1] == 0 or b.shape[1] == 0:
        return 0.0
    denom = float(min(a.shape[1], b.shape[1]))
    return float(np.linalg.norm(a.T @ b, ord="fro") ** 2 / denom)


def operator_cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=float).ravel()
    bv = np.asarray(b, dtype=float).ravel()
    den = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(abs(av @ bv) / den) if den > 1e-30 else 0.0


def propagated_derivatives(
    target: np.ndarray,
    old: np.ndarray,
    p: k.Params,
    packets: list[np.ndarray],
    delay: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_phi, base_old = g3.advance(target.copy(), old.copy(), p, delay)
    dphi = []
    dold = []
    for packet in packets:
        phi_i, old_i = g3.advance(
            target + epsilon * packet,
            old + epsilon * packet,
            p,
            delay,
        )
        dphi.append((phi_i - base_phi) / epsilon)
        dold.append((old_i - base_old) / epsilon)
    return base_phi, base_old, np.stack(dphi), np.stack(dold)


def omega_matrix(dphi: np.ndarray, dold: np.ndarray, packets: list[np.ndarray]) -> np.ndarray:
    m = len(packets)
    omega = np.zeros((m, m), dtype=float)  # omega[j, i] = j <- i
    packet_norm2 = [float(np.sum(packet * packet)) for packet in packets]
    for i in range(m):
        for j, packet_j in enumerate(packets):
            numerator = float(np.sum(dphi[i] * packet_j) + np.sum(dold[i] * packet_j))
            omega[j, i] = numerator / (2.0 * packet_norm2[j] + 1e-30)
    return omega


def top_summary(rows: list[dict], key: str, count: int = 12) -> dict:
    ordered = sorted(rows, key=lambda r: float(r[key]), reverse=True)[:count]
    return {
        "predictor": key,
        "count": len(ordered),
        "mean_nonadditivity": float(np.mean([r["relative_nonadditivity"] for r in ordered])),
        "median_nonadditivity": float(np.median([r["relative_nonadditivity"] for r in ordered])),
        "pairs": [[int(r["i"]), int(r["j"])] for r in ordered],
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
    ap.add_argument("--probe-epsilon", type=float, default=PROBE_EPS)
    ap.add_argument("--out", type=Path, default=Path("results/gate8/receipt.json"))
    args = ap.parse_args()

    if args.a_count != 30:
        raise ValueError("frozen Gate 8 requires exactly 30 deterministic A locations")
    if abs(args.probe_epsilon - PROBE_EPS) > 1e-15:
        raise ValueError("frozen Gate 8 requires probe epsilon 0.002")

    p = k.Params()
    target, old = k.evolve_state(args.grid, p, args.target_steps)
    base_phi, base_old = g3.advance(target.copy(), old.copy(), p, args.delay)

    coords_b = g6.packet_grid(args.grid, args.packet_step)
    packets_b = [g4.packet_at(args.grid, x, y) for x, y in coords_b]
    inputs = np.stack([packet.ravel() for packet in packets_b], axis=1)
    whitening = g6.whitening_from_inputs(inputs)
    base_columns = g6.response_columns(
        base_phi, base_old, packets_b, p, args.horizon, args.epsilon_b
    )

    sites = g5.deterministic_a_locations(args.grid, args.a_count)
    packets_a = [g4.packet_at(args.grid, x, y) for x, y in sites]
    panel = heldout_pairs(sites, args.grid)

    # Predictor objects are measured without looking at any pair outcome.
    single_delta: dict[int, np.ndarray] = {}
    whitened_delta: dict[int, np.ndarray] = {}
    right_subspaces: dict[int, np.ndarray] = {}
    right_ranks: dict[int, int] = {}
    for idx in range(len(sites)):
        delta = g7.delta_operator(
            target, old, p, [sites[idx]], packets_b, base_columns,
            args.delay, args.horizon, args.epsilon_a, args.epsilon_b,
        )
        single_delta[idx] = delta
        dw = delta @ whitening
        whitened_delta[idx] = dw
        subspace, rank = dominant_right_subspace(dw, 0.90)
        right_subspaces[idx] = subspace
        right_ranks[idx] = rank
        print(f"single predictor {idx+1:02d}/{len(sites)} rank90={rank}", flush=True)

    _, _, dphi, dold = propagated_derivatives(
        target, old, p, packets_a, args.delay, args.probe_epsilon
    )
    omega = omega_matrix(dphi, dold, packets_a)

    # Now measure the previously frozen pair outcomes.
    rows = []
    for pi, meta in enumerate(panel):
        i, j = int(meta["i"]), int(meta["j"])
        actual = g7.delta_operator(
            target, old, p, [sites[i], sites[j]], packets_b, base_columns,
            args.delay, args.horizon, args.epsilon_a, args.epsilon_b,
        )
        outcome = g7.compare(actual, single_delta[i] + single_delta[j], whitening)
        omega_j_i = float(omega[j, i])
        omega_i_j = float(omega[i, j])
        row = {
            **meta,
            **outcome,
            "inverse_distance": float(1.0 / (1.0 + float(meta["distance"]))),
            "operator_cosine": operator_cosine(whitened_delta[i], whitened_delta[j]),
            "right_subspace_overlap": subspace_overlap(right_subspaces[i], right_subspaces[j]),
            "rank90_i": int(right_ranks[i]),
            "rank90_j": int(right_ranks[j]),
            "omega_j_from_i": omega_j_i,
            "omega_i_from_j": omega_i_j,
            "propagated_coupling": float(max(abs(omega_j_i), abs(omega_i_j))),
            "propagated_geometric_mean": float(np.sqrt(abs(omega_j_i * omega_i_j))),
        }
        rows.append(row)
        print(
            f"pair {pi+1:02d}/{len(panel)} ({i},{j}) d={meta['distance']:.3f} "
            f"nonadd={outcome['relative_nonadditivity']:.4f}",
            flush=True,
        )

    y = np.asarray([r["relative_nonadditivity"] for r in rows], dtype=float)
    predictors = {
        "distance": np.asarray([r["inverse_distance"] for r in rows], dtype=float),
        "operator_cosine": np.asarray([r["operator_cosine"] for r in rows], dtype=float),
        "subspace": np.asarray([r["right_subspace_overlap"] for r in rows], dtype=float),
        "propagated": np.asarray([r["propagated_coupling"] for r in rows], dtype=float),
    }
    rho = {name: spearman(values, y) for name, values in predictors.items()}
    baseline_best = max(rho["distance"], rho["operator_cosine"], rho["subspace"])
    margin = float(rho["propagated"] - baseline_best)
    wins = bool(rho["propagated"] >= 0.50 and margin >= 0.10)

    physical_distance = np.asarray([r["distance"] for r in rows], dtype=float)
    propagated = predictors["propagated"]
    summary = {
        "pair_count": len(rows),
        "spearman_nonadditivity": rho,
        "best_baseline_spearman": float(baseline_best),
        "propagated_margin_over_best_baseline": margin,
        "pearson_physical_distance_vs_propagated_coupling": pearson(
            physical_distance, propagated
        ),
        "top12": {
            "distance": top_summary(rows, "inverse_distance"),
            "operator_cosine": top_summary(rows, "operator_cosine"),
            "subspace": top_summary(rows, "right_subspace_overlap"),
            "propagated": top_summary(rows, "propagated_coupling"),
        },
    }

    receipt = {
        "gate": "G8_propagated_coupling_predictor_contest",
        "protocol": "GATE8_PROPAGATED_COUPLING_PROTOCOL.md",
        "result": (
            "PROPAGATED_COUPLING_WINS"
            if wins else "PROPAGATED_COUPLING_NOT_PRIVILEGED"
        ),
        "configuration": {
            "grid": args.grid,
            "target_steps": args.target_steps,
            "delay": args.delay,
            "horizon": args.horizon,
            "packet_step": args.packet_step,
            "response_directions": len(coords_b),
            "a_count": args.a_count,
            "epsilon_a": args.epsilon_a,
            "epsilon_b": args.epsilon_b,
            "probe_epsilon": args.probe_epsilon,
            "heldout_pair_count": len(panel),
            "gate7_extreme_pairs_excluded_each_side": GATE7_EXTREME_COUNT,
        },
        "primary_rule": {
            "rho_propagated_min": 0.50,
            "margin_over_best_baseline_min": 0.10,
        },
        "summary": summary,
        "omega_matrix": omega.tolist(),
        "single_update_rank90": {str(i): int(right_ranks[i]) for i in range(len(sites))},
        "pairs": rows,
        "interpretation": (
            "Gate 8 asks only whether an independently measured small-probe propagated "
            "coupling is a privileged predictor of finite nonlinear operator-update "
            "interference in this toy field. It does not claim a new multiple-scattering "
            "formalism and does not assert the exact CausalHorizon identity for the nonlinear dynamics."
        ),
        "receipt_command": "python gate8_propagated_coupling.py",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"result": receipt["result"], **summary}, indent=2))


if __name__ == "__main__":
    main()
