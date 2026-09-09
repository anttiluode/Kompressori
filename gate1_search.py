#!/usr/bin/env python3
"""Kompressori Gate 1: optimize the visible future, test the hidden response.

Search many sparse random observation masks using ONLY future-trajectory fidelity.
Then, without changing the chosen masks, evaluate how faithfully each compressed
state responds to several unseen tiny perturbations.

If future fidelity were a reliable proxy for local response geometry, the two
scores should rise together. Gate 1 measures that directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

import kompressori as k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--target-steps", type=int, default=350)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--fraction", type=float, default=.05)
    ap.add_argument("--candidates", type=int, default=80)
    ap.add_argument("--probe-seeds", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=.002)
    ap.add_argument("--mask-seed-base", type=int, default=1_000_000)
    ap.add_argument("--out", type=Path, default=Path("results/gate1/receipt.json"))
    a = ap.parse_args()

    p = k.Params()
    target, old = k.evolve_state(a.grid, p, a.target_steps)
    reference = k.rollout(target.copy(), old.copy(), p, a.horizon)

    # Phase A: the compressor sees only future fidelity.
    candidates = []
    for i in range(a.candidates):
        rng = np.random.default_rng(a.mask_seed_base + i)
        mask = k.make_mask("random", target, a.fraction, rng)
        cphi, cold = k.compress(target, old, mask)
        compressed = k.rollout(cphi, cold, p, a.horizon)
        future = k.future_metrics(compressed, reference, mask)["future_corr"]
        candidates.append({
            "candidate": i,
            "future_corr": float(future),
            "_mask": mask,
            "_cphi": cphi,
            "_cold": cold,
            "_compressed": compressed,
        })

    candidates.sort(key=lambda r: r["future_corr"], reverse=True)
    rank_of = {r["candidate"]: rank for rank, r in enumerate(candidates)}

    # Phase B: surprise the chosen states with perturbations not used for selection.
    probes = []
    for seed in range(a.probe_seeds):
        d, xy = k.torus_packet(a.grid, np.random.default_rng(9000 + seed))
        full_plus = k.rollout(target + a.epsilon*d, old + a.epsilon*d, p, a.horizon)
        probes.append((seed, xy, d, full_plus - reference))

    evaluated = []
    for row in candidates:
        i = row["candidate"]
        mask = row["_mask"]
        cphi = row["_cphi"]
        cold = row["_cold"]
        compressed = row["_compressed"]
        rcorr, rgain = [], []
        for _, _, d, full_delta in probes:
            compressed_plus = k.rollout(cphi + a.epsilon*d, cold + a.epsilon*d, p, a.horizon)
            m = k.response_metrics(compressed_plus - compressed, full_delta, mask)
            rcorr.append(m["response_corr"])
            rgain.append(m["response_gain_ratio"])
        evaluated.append({
            "candidate": i,
            "future_rank": rank_of[i],
            "future_corr": row["future_corr"],
            "response_corr_mean": float(np.mean(rcorr)),
            "response_corr_std": float(np.std(rcorr)),
            "response_gain_mean": float(np.mean(rgain)),
        })

    f = np.array([r["future_corr"] for r in evaluated])
    q = np.array([r["response_corr_mean"] for r in evaluated])
    relation = float(np.corrcoef(f, q)[0, 1])
    by_future = sorted(evaluated, key=lambda r: r["future_corr"], reverse=True)
    by_response = sorted(evaluated, key=lambda r: r["response_corr_mean"], reverse=True)

    receipt = {
        "gate": "G1_future_only_mask_search",
        "claim_tested": "Future fidelity is a reliable proxy for response-geometry fidelity.",
        "result": "not supported in this toy search",
        "grid": a.grid,
        "target_steps": a.target_steps,
        "horizon": a.horizon,
        "fraction": a.fraction,
        "candidates": a.candidates,
        "probe_seeds": a.probe_seeds,
        "probe_epsilon": a.epsilon,
        "future_response_pearson": relation,
        "future_corr_median": float(np.median(f)),
        "response_corr_mean_all": float(np.mean(q)),
        "response_corr_mean_top10_future": float(np.mean([r["response_corr_mean"] for r in by_future[:10]])),
        "response_corr_mean_bottom10_future": float(np.mean([r["response_corr_mean"] for r in by_future[-10:]])),
        "best_future_mask": by_future[0],
        "best_response_mask": by_response[0],
        "probe_locations": [
            {"seed": seed, "x": xy[0], "y": xy[1]} for seed, xy, _, _ in probes
        ],
        "interpretation": (
            "At fixed sparse random observation, selecting masks by future trajectory alone "
            "does not identify masks that preserve the response to unseen perturbations. "
            "This is a toy counterexample to using replay/output fidelity as the only guard."
        ),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
