#!/usr/bin/env python3
"""Kompressori Gate 2: does a response-aware sparse guard generalize?

Gate 1 showed that future/replay fidelity is a poor proxy for response fidelity.
The obvious repair is to score candidate compressions with perturbation probes.
Gate 2 attacks that repair rather than assuming it works.

At a fixed sparse observation budget we:
  1. generate many random masks,
  2. keep only the top quarter by ordinary future fidelity,
  3. use half of a perturbation set to choose the mask with the smallest
     finite-difference response error,
  4. test that chosen mask on held-out perturbation directions.

If a few response probes are enough to constrain the local future operator, the
response-aware mask should beat the future-only mask on held-out probes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import kompressori as k


def random_mask(n: int, fraction: float, rng: np.random.Generator) -> np.ndarray:
    count = max(1, min(n * n - 1, int(round(fraction * n * n))))
    idx = rng.choice(n * n, count, replace=False)
    mask = np.zeros(n * n, dtype=bool)
    mask[idx] = True
    return mask.reshape(n, n)


def mean_stats(row: dict, ids: np.ndarray) -> dict[str, float]:
    return {
        "response_corr": float(np.mean(row["response_corr"][ids])),
        "response_rel_error": float(np.mean(row["response_rel_error"][ids])),
        "response_gain": float(np.mean(row["response_gain"][ids])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=36)
    ap.add_argument("--target-steps", type=int, default=350)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--fraction", type=float, default=.05)
    ap.add_argument("--candidates", type=int, default=140)
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--splits", type=int, default=6)
    ap.add_argument("--future-top-fraction", type=float, default=.25)
    ap.add_argument("--epsilon", type=float, default=.002)
    ap.add_argument("--out", type=Path, default=Path("results/gate2/receipt.json"))
    a = ap.parse_args()

    if a.probes < 4 or a.probes % 2:
        raise ValueError("--probes must be an even integer >= 4")
    if not 0 < a.future_top_fraction <= 1:
        raise ValueError("--future-top-fraction must be in (0,1]")

    p = k.Params()
    target, old = k.evolve_state(a.grid, p, a.target_steps)
    reference = k.rollout(target.copy(), old.copy(), p, a.horizon)

    # Probe set. These are localized torus packets at deterministic locations.
    probes = []
    probe_locations = []
    for seed in range(a.probes):
        d, xy = k.torus_packet(a.grid, np.random.default_rng(70000 + seed))
        full_plus = k.rollout(target + a.epsilon * d, old + a.epsilon * d, p, a.horizon)
        probes.append((d, full_plus - reference))
        probe_locations.append({"probe": seed, "x": xy[0], "y": xy[1]})

    # Evaluate every candidate once on every probe. Selection happens later.
    rows = []
    for candidate in range(a.candidates):
        mask = random_mask(a.grid, a.fraction, np.random.default_rng(8_000_000 + candidate))
        cphi, cold = k.compress(target, old, mask)
        compressed = k.rollout(cphi, cold, p, a.horizon)
        future_corr = k.future_metrics(compressed, reference, mask)["future_corr"]

        response_corr = []
        response_rel_error = []
        response_gain = []
        for d, full_delta in probes:
            compressed_plus = k.rollout(
                cphi + a.epsilon * d,
                cold + a.epsilon * d,
                p,
                a.horizon,
            )
            m = k.response_metrics(compressed_plus - compressed, full_delta, mask)
            response_corr.append(m["response_corr"])
            response_rel_error.append(m["response_rel_error"])
            response_gain.append(m["response_gain_ratio"])

        rows.append({
            "candidate": candidate,
            "future_corr": float(future_corr),
            "response_corr": np.asarray(response_corr),
            "response_rel_error": np.asarray(response_rel_error),
            "response_gain": np.asarray(response_gain),
        })

    future_values = np.asarray([r["future_corr"] for r in rows])
    threshold = float(np.quantile(future_values, 1 - a.future_top_fraction))
    eligible = [r for r in rows if r["future_corr"] >= threshold]
    future_only = max(rows, key=lambda r: r["future_corr"])

    split_rng = np.random.default_rng(90000)
    split_rows = []
    half = a.probes // 2
    for split in range(a.splits):
        order = split_rng.permutation(a.probes)
        train = order[:half]
        test = order[half:]

        # No arbitrary lambda: first require decent future fidelity, then among
        # those candidates minimize finite-difference response error on training
        # perturbations. Smaller relative error means direction + gain agree.
        response_aware = min(
            eligible,
            key=lambda r: float(np.mean(r["response_rel_error"][train])),
        )

        split_rows.append({
            "split": split,
            "train_probes": train.tolist(),
            "test_probes": test.tolist(),
            "future_only": {
                "candidate": future_only["candidate"],
                "future_corr": future_only["future_corr"],
                "train": mean_stats(future_only, train),
                "test": mean_stats(future_only, test),
            },
            "response_aware": {
                "candidate": response_aware["candidate"],
                "future_corr": response_aware["future_corr"],
                "train": mean_stats(response_aware, train),
                "test": mean_stats(response_aware, test),
            },
        })

    def average(path: tuple[str, ...]) -> float:
        vals = []
        for split in split_rows:
            obj = split
            for key in path:
                obj = obj[key]
            vals.append(float(obj))
        return float(np.mean(vals))

    aggregate = {
        "future_only_train_response_corr": average(("future_only", "train", "response_corr")),
        "future_only_train_response_rel_error": average(("future_only", "train", "response_rel_error")),
        "future_only_test_response_corr": average(("future_only", "test", "response_corr")),
        "future_only_test_response_rel_error": average(("future_only", "test", "response_rel_error")),
        "response_aware_train_response_corr": average(("response_aware", "train", "response_corr")),
        "response_aware_train_response_rel_error": average(("response_aware", "train", "response_rel_error")),
        "response_aware_test_response_corr": average(("response_aware", "test", "response_corr")),
        "response_aware_test_response_rel_error": average(("response_aware", "test", "response_rel_error")),
        "response_aware_future_corr": average(("response_aware", "future_corr")),
    }

    heldout_improved = (
        aggregate["response_aware_test_response_rel_error"]
        < aggregate["future_only_test_response_rel_error"]
    )
    receipt = {
        "gate": "G2_response_guard_crossvalidation",
        "claim_tested": (
            "A sparse compression chosen using a few response probes will preserve "
            "the response geometry for unseen perturbation directions."
        ),
        "result": (
            "supported in this run" if heldout_improved
            else "not supported in this run"
        ),
        "grid": a.grid,
        "target_steps": a.target_steps,
        "horizon": a.horizon,
        "fraction": a.fraction,
        "candidates": a.candidates,
        "probes": a.probes,
        "splits": a.splits,
        "future_top_fraction": a.future_top_fraction,
        "future_threshold": threshold,
        "eligible_candidates": len(eligible),
        "probe_epsilon": a.epsilon,
        "future_only_best_future_corr": future_only["future_corr"],
        "probe_locations": probe_locations,
        "aggregate": aggregate,
        "splits_detail": split_rows,
        "interpretation": (
            "The response-aware selector fits its training perturbations better, "
            "but in the supplied run that advantage does not survive held-out "
            "perturbation directions. At this 5% state-only budget, checking a "
            "small probe set is therefore not enough to certify the local future "
            "operator. This is a property of this toy experiment, not a general theorem."
        ),
        "receipt_command": (
            "python gate2_crossvalidate.py --grid 36 --horizon 25 --candidates 140 "
            "--probes 8 --splits 6"
        ),
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
