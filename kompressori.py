#!/usr/bin/env python3
"""
Kompressori Gate 0: future compression is not response-geometry compression.

A deliberately small PhiWorld-like nonlinear field experiment.

We take one evolved state s=(phi, phi_old), retain only a spatial subset, zero
all hidden cells, and ask two different questions:

  1. FUTURE FIDELITY
     Does the compressed state produce the same hidden future trajectory?

  2. RESPONSE FIDELITY
     Around that compressed state, does the same tiny future perturbation
     produce the same hidden response as it does around the full state?

The second score is a finite-difference tangent/response test. It is the toy
version of the distinction:

    preserving today's output != preserving tomorrow's response geometry.

This is not a Navier-Stokes/Euler solver and not evidence for holography or a
brain mechanism. It is a falsifiable dynamical-compression probe inspired by
those questions.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Params:
    dt: float = 0.08
    damping: float = 0.001
    base_c_sq: float = 1.0
    tension_factor: float = 5.0
    potential_lin: float = 1.0
    potential_cub: float = 0.2
    biharmonic_gamma: float = 0.02


def lap(x: np.ndarray) -> np.ndarray:
    """Periodic five-point Laplacian."""
    return (
        np.roll(x, 1, 0)
        + np.roll(x, -1, 0)
        + np.roll(x, 1, 1)
        + np.roll(x, -1, 1)
        - 4 * x
    )


def accel(phi: np.ndarray, p: Params) -> np.ndarray:
    L = lap(phi)
    B = lap(L)
    c2 = p.base_c_sq / (1 + p.tension_factor * phi * phi + 1e-9)
    return (
        c2 * L
        + p.potential_lin * phi
        - p.potential_cub * phi**3
        - p.biharmonic_gamma * B
    )


def step(phi: np.ndarray, old: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    vel = phi - old
    new = phi + (1 - p.damping * p.dt) * vel + p.dt**2 * accel(phi, p)
    return new, phi


def initial_gaussian(n: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:n, :n]
    c = n // 2
    r = n / 15
    phi = 2 * np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * r * r))
    return phi.astype(float), phi.astype(float).copy()


def evolve_state(n: int, p: Params, steps: int) -> tuple[np.ndarray, np.ndarray]:
    phi, old = initial_gaussian(n)
    for _ in range(steps):
        phi, old = step(phi, old, p)
    return phi, old


def rollout(phi: np.ndarray, old: np.ndarray, p: Params, horizon: int) -> np.ndarray:
    hist = [phi.copy()]
    for _ in range(horizon):
        phi, old = step(phi, old, p)
        hist.append(phi.copy())
    return np.asarray(hist)


def exact_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    flat = score.ravel()
    k = max(1, min(flat.size - 1, int(round(fraction * flat.size))))
    idx = np.argpartition(flat, k - 1)[:k]
    m = np.zeros(flat.size, dtype=bool)
    m[idx] = True
    return m.reshape(score.shape)


def make_mask(
    kind: str,
    target: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = target.shape[0]
    yy, xx = np.mgrid[:n, :n]
    c = (n - 1) / 2

    if kind == "random":
        k = max(1, min(n * n - 1, int(round(fraction * n * n))))
        idx = rng.choice(n * n, k, replace=False)
        m = np.zeros(n * n, dtype=bool)
        m[idx] = True
        return m.reshape(n, n)

    if kind == "center":
        return exact_mask((xx - c) ** 2 + (yy - c) ** 2, fraction)

    if kind == "ring":
        r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
        score = np.abs(r - 0.27 * n) + rng.uniform(0, 1e-6, r.shape)
        return exact_mask(score, fraction)

    if kind == "edge":
        # Important: the field itself has PERIODIC boundaries. This is merely
        # a square edge-shaped observation mask, not a physical boundary cue.
        score = np.minimum.reduce([xx, yy, n - 1 - xx, n - 1 - yy]).astype(float)
        score += rng.uniform(0, 1e-6, score.shape)
        return exact_mask(score, fraction)

    if kind == "energy":
        # Keep the largest-|phi| cells. Useful as a simple non-random baseline.
        score = -np.abs(target) + rng.uniform(0, 1e-9, target.shape)
        return exact_mask(score, fraction)

    raise ValueError(f"unknown mask kind: {kind}")


def compress(phi: np.ndarray, old: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros_like(phi)
    out_old = np.zeros_like(old)
    out[mask] = phi[mask]
    out_old[mask] = old[mask]
    return out, out_old


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(float)
    b = b.ravel().astype(float)
    a -= a.mean()
    b -= b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 1e-15 else 0.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(float)
    b = b.ravel().astype(float)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 1e-15 else 0.0


def future_metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    hidden = ~mask
    a = pred[1:, hidden]
    b = truth[1:, hidden]
    mse = float(np.mean((a - b) ** 2))
    energy = float(np.mean(b * b)) + 1e-15
    return {
        "future_corr": _corr(a, b),
        "future_cosine": _cosine(a, b),
        "future_nrmse": float(np.sqrt(mse / energy)),
    }


def response_metrics(
    compressed_delta: np.ndarray,
    full_delta: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    hidden = ~mask
    a = compressed_delta[1:, hidden]
    b = full_delta[1:, hidden]
    nb = float(np.linalg.norm(b.ravel())) + 1e-15
    return {
        "response_corr": _corr(a, b),
        "response_cosine": _cosine(a, b),
        "response_gain_ratio": float(np.linalg.norm(a.ravel()) / nb),
        "response_rel_error": float(np.linalg.norm((a - b).ravel()) / nb),
    }


def torus_packet(
    n: int,
    rng: np.random.Generator,
    sigma: float | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Unit-RMS Gaussian displacement packet on the periodic domain."""
    if sigma is None:
        sigma = n / 18
    yy, xx = np.mgrid[:n, :n]
    cy = float(rng.uniform(0, n))
    cx = float(rng.uniform(0, n))
    dx0 = np.abs(xx - cx)
    dy0 = np.abs(yy - cy)
    dx = np.minimum(dx0, n - dx0)
    dy = np.minimum(dy0, n - dy0)
    g = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
    g /= np.sqrt(np.mean(g * g))
    return g, (cx, cy)


def run_gate0(
    grid: int,
    target_steps: int,
    horizon: int,
    fractions: list[float],
    mask_kinds: list[str],
    seeds: int,
    epsilon: float,
    p: Params,
) -> tuple[list[dict], dict]:
    target, old = evolve_state(grid, p, target_steps)
    reference = rollout(target.copy(), old.copy(), p, horizon)
    rows: list[dict] = []
    linearity_checks = []

    for seed in range(seeds):
        probe_rng = np.random.default_rng(9000 + seed)
        delta0, (probe_x, probe_y) = torus_packet(grid, probe_rng)

        full_plus = rollout(target + epsilon * delta0, old + epsilon * delta0, p, horizon)
        full_delta = full_plus - reference

        # Check that epsilon is small enough for the finite-difference response
        # to be meaningfully tangent-like.
        half_plus = rollout(
            target + 0.5 * epsilon * delta0,
            old + 0.5 * epsilon * delta0,
            p,
            horizon,
        )
        half_delta = half_plus - reference
        linearity_checks.append(
            {
                "seed": seed,
                "probe_x": probe_x,
                "probe_y": probe_y,
                "relative_error_full_vs_2x_half": float(
                    np.linalg.norm((full_delta - 2 * half_delta).ravel())
                    / (np.linalg.norm(full_delta.ravel()) + 1e-15)
                ),
                "cosine_full_vs_2x_half": _cosine(full_delta, 2 * half_delta),
            }
        )

        for kind_i, kind in enumerate(mask_kinds):
            for fraction in fractions:
                mask_rng = np.random.default_rng(
                    100000 * seed + 1000 * kind_i + int(round(10000 * fraction))
                )
                mask = make_mask(kind, target, fraction, mask_rng)
                cphi, cold = compress(target, old, mask)

                compressed_future = rollout(cphi, cold, p, horizon)
                compressed_plus = rollout(
                    cphi + epsilon * delta0,
                    cold + epsilon * delta0,
                    p,
                    horizon,
                )
                compressed_delta = compressed_plus - compressed_future

                row = {
                    "seed": seed,
                    "mask_kind": kind,
                    "fraction": float(mask.mean()),
                    "observed_cells": int(mask.sum()),
                    "hidden_cells": int((~mask).sum()),
                    "probe_x": probe_x,
                    "probe_y": probe_y,
                }
                row.update(future_metrics(compressed_future, reference, mask))
                row.update(response_metrics(compressed_delta, full_delta, mask))
                rows.append(row)

    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for kind in mask_kinds:
        grouped[kind] = {}
        for fraction in fractions:
            wanted = [
                r
                for r in rows
                if r["mask_kind"] == kind
                and abs(r["fraction"] - fraction) < 1.0 / (grid * grid)
            ]
            grouped[kind][f"{100*fraction:g}%"] = {
                key: float(np.mean([r[key] for r in wanted]))
                for key in [
                    "future_corr",
                    "future_nrmse",
                    "response_corr",
                    "response_gain_ratio",
                    "response_rel_error",
                ]
            }

    receipt = {
        "gate": "G0_future_vs_response_geometry",
        "claim_tested": (
            "A spatial subset that preserves the apparent future may still fail "
            "to preserve the field's response to the next perturbation."
        ),
        "params": asdict(p),
        "grid": grid,
        "target_steps": target_steps,
        "horizon": horizon,
        "fractions": fractions,
        "mask_kinds": mask_kinds,
        "seeds": seeds,
        "probe_epsilon": epsilon,
        "boundary_note": (
            "lap() uses np.roll, so the simulated domain is periodic. The 'edge' "
            "mask is an observation geometry, not a physical boundary condition."
        ),
        "linearity_checks": linearity_checks,
        "grouped_means": grouped,
    }
    return rows, receipt


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--target-steps", type=int, default=350)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--fractions", type=float, nargs="+", default=[.01, .02, .05, .10, .25, .50])
    ap.add_argument("--mask-kinds", nargs="+", default=["random", "center", "ring", "edge", "energy"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=.002)
    ap.add_argument("--outdir", type=Path, default=Path("results/gate0"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows, receipt = run_gate0(
        grid=args.grid,
        target_steps=args.target_steps,
        horizon=args.horizon,
        fractions=args.fractions,
        mask_kinds=args.mask_kinds,
        seeds=args.seeds,
        epsilon=args.epsilon,
        p=Params(),
    )
    write_csv(rows, args.outdir / "runs.csv")
    (args.outdir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    print("\nGate 0: future fidelity vs response-geometry fidelity")
    print("(means across seeds; response gain = 1 is ideal)\n")
    for kind, by_fraction in receipt["grouped_means"].items():
        print(kind)
        for fraction, m in by_fraction.items():
            print(
                f"  {fraction:>4s}  future={m['future_corr']:+.3f}  "
                f"response={m['response_corr']:+.3f}  gain={m['response_gain_ratio']:.3f}"
            )
    print(f"\nWrote {args.outdir/'runs.csv'}")
    print(f"Wrote {args.outdir/'receipt.json'}")


if __name__ == "__main__":
    main()
