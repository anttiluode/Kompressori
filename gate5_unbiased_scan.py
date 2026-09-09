#!/usr/bin/env python3
"""Kompressori Gate 5: deterministic, non-post-selected A->B transfer scan.

Gate 3 found one high-gain pair by search and Gate 4 characterized it. Gate 5
removes that selection step. A locations, B offsets, and delays are fixed before
any gain is measured.

Default protocol (40x40 field):
  * 30 deterministic A locations from a modular sequence,
  * B at radii 2, 4, 8, 12 in the four cardinal directions,
  * delays 12, 18, 24, 30, 36,
  * finite A epsilon=0.1 and tangent-like B epsilon=0.002.

The result answers a prevalence/locality question, not an Euler/Navier-Stokes
question: are local susceptibility windows common across this toy field?
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import kompressori as k
import gate3_transfer as g3
import gate4_kernel as g4


def deterministic_a_locations(n: int, count: int) -> list[tuple[int, int]]:
    """Outcome-independent sequence. Defaults give 30 distinct sites on n=40."""
    pts: list[tuple[int, int]] = []
    i = 0
    while len(pts) < count:
        pt = ((7 + 13 * i) % n, (3 + 17 * i) % n)
        if pt not in pts:
            pts.append(pt)
        i += 1
        if i > n * n * 2:
            raise RuntimeError("could not generate requested distinct A locations")
    return pts


def summarize(values) -> dict[str, float | int]:
    x = np.asarray(list(values), dtype=float)
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
        "p05": float(np.quantile(x, .05)),
        "p90": float(np.quantile(x, .90)),
        "p95": float(np.quantile(x, .95)),
        "fraction_gain_gt_1_1": float(np.mean(x > 1.1)),
        "fraction_gain_gt_1_2": float(np.mean(x > 1.2)),
        "fraction_gain_lt_0_9": float(np.mean(x < .9)),
    }


def scan(
    target: np.ndarray,
    old: np.ndarray,
    p: k.Params,
    a_locations: list[tuple[int, int]],
    delays: list[int],
    radii: list[int],
    epsilon_a: float,
    epsilon_b: float,
    horizon: int,
) -> list[dict]:
    n = target.shape[0]
    directions = [(1, 0, "+x"), (-1, 0, "-x"), (0, 1, "+y"), (0, -1, "-y")]
    rows: list[dict] = []

    for delay in delays:
        base_phi, base_old = g3.advance(target.copy(), old.copy(), p, delay)
        base_future = k.rollout(base_phi.copy(), base_old.copy(), p, horizon)
        base_b_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, float]] = {}

        for ax, ay in a_locations:
            packet_a = g4.packet_at(n, ax, ay)
            prep_phi, prep_old = g3.advance(
                target + epsilon_a * packet_a,
                old + epsilon_a * packet_a,
                p,
                delay,
            )
            prep_future = k.rollout(prep_phi.copy(), prep_old.copy(), p, horizon)
            dphi = prep_phi - base_phi
            dold = prep_old - base_old
            prep_rel = float(
                np.linalg.norm(dphi.ravel())
                / (np.linalg.norm(base_phi.ravel()) + 1e-15)
            )

            for radius in radii:
                for ux, uy, direction in directions:
                    bx = int((ax + ux * radius) % n)
                    by = int((ay + uy * radius) % n)
                    key = (bx, by)

                    if key not in base_b_cache:
                        packet_b = g4.packet_at(n, bx, by)
                        base_plus = k.rollout(
                            base_phi + epsilon_b * packet_b,
                            base_old + epsilon_b * packet_b,
                            p,
                            horizon,
                        )
                        delta_base = base_plus - base_future
                        base_b_cache[key] = (
                            packet_b,
                            delta_base,
                            float(np.linalg.norm(delta_base.ravel())) + 1e-15,
                        )

                    packet_b, delta_base, base_norm = base_b_cache[key]
                    prep_plus = k.rollout(
                        prep_phi + epsilon_b * packet_b,
                        prep_old + epsilon_b * packet_b,
                        p,
                        horizon,
                    )
                    delta_prep = prep_plus - prep_future

                    weight = packet_b * packet_b
                    weight /= float(weight.sum())
                    local_prep_rms = float(
                        np.sqrt(np.sum(weight * (dphi * dphi + dold * dold) / 2))
                    )

                    rows.append({
                        "a_x": ax,
                        "a_y": ay,
                        "b_x": bx,
                        "b_y": by,
                        "radius": radius,
                        "direction": direction,
                        "delay": delay,
                        "transfer_gain": float(
                            np.linalg.norm(delta_prep.ravel()) / base_norm
                        ),
                        "response_corr": g3.corr(delta_prep, delta_base),
                        "prepared_state_relative_change": prep_rel,
                        "local_preparation_rms_at_b": local_prep_rms,
                    })
    return rows


def tiny_control(
    target: np.ndarray,
    old: np.ndarray,
    p: k.Params,
    a_locations: list[tuple[int, int]],
    epsilon_a: float,
    epsilon_b: float,
    horizon: int,
    delay: int = 24,
) -> dict:
    rows = scan(
        target, old, p, a_locations,
        delays=[delay], radii=[2, 4],
        epsilon_a=epsilon_a, epsilon_b=epsilon_b, horizon=horizon,
    )
    return summarize(r["transfer_gain"] for r in rows)


def write_svg(receipt: dict, path: Path) -> None:
    by_r = receipt["by_radius"]
    by_d = receipt["by_delay"]
    radii = [2, 4, 8, 12]
    delays = [12, 18, 24, 30, 36]
    W, H = 1000, 460
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#0f1218"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#dfe7f7}.muted{fill:#96a2bb}.axis{stroke:#4a556e;stroke-width:1}.grid{stroke:#252c39;stroke-width:1}.mean{fill:#f6c85f}.range{stroke:#6f9ceb;stroke-width:5;stroke-linecap:round}.line{stroke:#65d6ad;stroke-width:3;fill:none}.dot{fill:#65d6ad}</style>',
        '<text x="30" y="35" font-size="22" font-weight="700">Gate 5 — deterministic transfer scan</text>',
        f'<text x="30" y="58" font-size="13" class="muted">{receipt["a_locations"]["count"]} predeclared A locations × 4 radii × 4 directions × 5 delays = {receipt["events"]} events</text>',
    ]
    x0, y0, x1, y1 = 70, 110, 470, 390
    parts += [
        f'<text x="{x0}" y="90" font-size="16" font-weight="700">Gain collapses with A→B distance</text>',
        f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>',
        f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>',
    ]
    def y_gain(v: float) -> float:
        return y1 - (v - .5) / 1.5 * (y1 - y0)
    for value in [.5, 1.0, 1.5, 2.0]:
        yy = y_gain(value)
        parts += [
            f'<line class="grid" x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}"/>',
            f'<text class="muted" x="{x0-38}" y="{yy+4:.1f}" font-size="12">{value:.1f}</text>',
        ]
    for i, radius in enumerate(radii):
        s = by_r[str(radius)]
        xx = x0 + 60 + i * 95
        parts += [
            f'<line class="range" x1="{xx}" y1="{y_gain(s["p05"]):.1f}" x2="{xx}" y2="{y_gain(s["p95"]):.1f}"/>',
            f'<circle class="mean" cx="{xx}" cy="{y_gain(s["mean"]):.1f}" r="6"/>',
            f'<text x="{xx-8}" y="{y1+24}" font-size="13">{radius}</text>',
            f'<text class="muted" x="{xx-28}" y="{y1+43}" font-size="11">{100*s["fraction_gain_gt_1_1"]:.1f}% &gt;1.1</text>',
        ]
    parts += [
        f'<text class="muted" x="{x0+145}" y="{H-18}" font-size="12">A→B center distance (cells)</text>',
        f'<text class="muted" x="{x0+10}" y="{y0+12}" font-size="11">blue = 5–95%; yellow = mean</text>',
    ]

    x0, y0, x1, y1 = 570, 110, 950, 390
    parts += [
        f'<text x="{x0}" y="90" font-size="16" font-weight="700">Amplification is a time window</text>',
        f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>',
        f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>',
    ]
    def y_frac(v: float) -> float:
        return y1 - v / .3 * (y1 - y0)
    for value in [0, .1, .2, .3]:
        yy = y_frac(value)
        parts += [
            f'<line class="grid" x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}"/>',
            f'<text class="muted" x="{x0-40}" y="{yy+4:.1f}" font-size="12">{int(value*100)}%</text>',
        ]
    points = []
    for i, delay in enumerate(delays):
        frac = by_d[str(delay)]["fraction_gain_gt_1_1"]
        xx = x0 + 35 + i * 80
        yy = y_frac(frac)
        points.append((xx, yy))
        parts += [
            f'<circle class="dot" cx="{xx}" cy="{yy:.1f}" r="5"/>',
            f'<text x="{xx-8}" y="{y1+24}" font-size="13">{delay}</text>',
        ]
    d = " ".join(("M" if i == 0 else "L") + f" {x} {y:.1f}" for i, (x, y) in enumerate(points))
    parts += [
        f'<path class="line" d="{d}"/>',
        f'<text class="muted" x="{x0+110}" y="{H-18}" font-size="12">delay before B (steps)</text>',
        f'<text x="{x0+10}" y="{y0+18}" font-size="12">fraction of all events with gain &gt; 1.1</text>',
        f'<text class="muted" x="{x0+10}" y="{y0+40}" font-size="11">tiny-A control at delay 24: max gain {receipt["tiny_a_control_at_delay_24_radii_2_4"]["max"]:.4f}×</text>',
        '</svg>',
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--target-steps", type=int, default=350)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--a-count", type=int, default=30)
    ap.add_argument("--epsilon-a", type=float, default=.1)
    ap.add_argument("--epsilon-b", type=float, default=.002)
    ap.add_argument("--tiny-a", type=float, default=.002)
    ap.add_argument("--outdir", type=Path, default=Path("results/gate5"))
    a = ap.parse_args()

    delays = [12, 18, 24, 30, 36]
    radii = [2, 4, 8, 12]
    p = k.Params()
    target, old = k.evolve_state(a.grid, p, a.target_steps)
    a_locations = deterministic_a_locations(a.grid, a.a_count)

    rows = scan(
        target, old, p, a_locations, delays, radii,
        a.epsilon_a, a.epsilon_b, a.horizon,
    )

    gains = np.asarray([r["transfer_gain"] for r in rows])
    local = np.asarray([r["local_preparation_rms_at_b"] for r in rows])
    by_radius = {
        str(radius): summarize(
            r["transfer_gain"] for r in rows if r["radius"] == radius
        )
        for radius in radii
    }
    by_delay = {
        str(delay): summarize(
            r["transfer_gain"] for r in rows if r["delay"] == delay
        )
        for delay in delays
    }
    near = [r["transfer_gain"] for r in rows if r["radius"] <= 4]
    far = [r["transfer_gain"] for r in rows if r["radius"] >= 8]
    best = max(rows, key=lambda r: r["transfer_gain"])
    worst = min(rows, key=lambda r: r["transfer_gain"])

    control = tiny_control(
        target, old, p, a_locations,
        epsilon_a=a.tiny_a,
        epsilon_b=a.epsilon_b,
        horizon=a.horizon,
        delay=24,
    )

    receipt = {
        "gate": "G5_deterministic_transfer_scan",
        "claim_tested": (
            "Gate 3's local A->B susceptibility is not merely an artifact "
            "of post-selecting pair 49."
        ),
        "result": (
            "supported for a common near-field nonlinear effect; not supported "
            "as a travelling or long-range cascade"
        ),
        "grid": a.grid,
        "target_steps": a.target_steps,
        "horizon": a.horizon,
        "epsilon_a": a.epsilon_a,
        "epsilon_b": a.epsilon_b,
        "a_locations": {
            "count": len(a_locations),
            "rule": f"x=(7+13*i)%{a.grid}, y=(3+17*i)%{a.grid} for i=0..{len(a_locations)-1}",
            "coordinates": [{"x": x, "y": y} for x, y in a_locations],
        },
        "delays": delays,
        "radii": radii,
        "directions": ["+x", "-x", "+y", "-y"],
        "events": len(rows),
        "overall": summarize(gains),
        "near_field_r_le_4": summarize(near),
        "far_field_r_ge_8": summarize(far),
        "by_radius": by_radius,
        "by_delay": by_delay,
        "best_event": best,
        "worst_event": worst,
        "tiny_a_control_at_delay_24_radii_2_4": control,
        "gain_vs_local_preparation_rms_pearson": float(np.corrcoef(gains, local)[0, 1]),
        "interpretation": (
            "Without selecting A/B pairs by outcome, finite A commonly changes the "
            "gain of nearby B probes. The effect decays sharply with distance: in the "
            "default run no radius-8 or radius-12 event exceeds gain 1.1. Tiny A at "
            "the same near-field locations stays approximately linear. The scan "
            "therefore validates a local nonlinear susceptibility effect in this toy "
            "field, while arguing against the more seductive story of a travelling "
            "or long-range amplification cascade."
        ),
        "receipt_command": "python gate5_unbiased_scan.py",
    }

    a.outdir.mkdir(parents=True, exist_ok=True)
    with (a.outdir / "events.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (a.outdir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    write_svg(receipt, a.outdir / "summary.svg")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
