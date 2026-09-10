"""Animated heatmap of house profit against house edge and comp rate, decade by decade.

Every cell of a grid (house edge on the x axis, comp rate on the y axis)
is its own casino: the same 8,192 gamblers, the same seed, different
rules, run for 120 years. One frame per simulated decade shows profit per
gambler per year in that decade, so you can watch the grid dim as gamblers
go broke and walk away. A second panel shows each cell's profit relative
to its own first decade: how much of the farm is left.

Writes outputs/landscape.gif, outputs/landscape_final.png and
outputs/landscape.csv.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
import io
import time
from dataclasses import replace
from multiprocessing import get_context

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from casino import HouseRules
from engine import SimConfig, run

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
EDGES = np.linspace(0.01, 0.28, 10)
COMP_RATES = np.linspace(0.0, 0.6, 8)
GAMBLERS = 8192
YEARS = 120
SEED = 11


def simulate_cell(args: tuple[float, float, HouseRules]) -> tuple[np.ndarray, np.ndarray]:
    """Profit per gambler-year and active share, per decade, for one rule set."""
    edge, comp_rate, base = args
    rules = replace(base, house_edge=edge, comp_rate=comp_rate)
    cfg = SimConfig(n_gamblers=GAMBLERS, n_periods=YEARS * 12, seed=SEED)
    res = run(rules, cfg)
    decades = YEARS // 10
    profit = res.profit_by_period.reshape(decades, 120).sum(axis=1) / GAMBLERS / 10.0
    active = res.active_by_period.reshape(decades, 120)[:, -1] / GAMBLERS
    return profit, active


def simulate_grid(base: HouseRules, workers: int) -> tuple[np.ndarray, np.ndarray]:
    tasks = [(float(e), float(c), base) for c in COMP_RATES for e in EDGES]
    t0 = time.perf_counter()
    with get_context("fork").Pool(workers) as pool:
        cells = pool.map(simulate_cell, tasks)
    print(f"{len(tasks)} casinos x {GAMBLERS} gamblers x {YEARS} years in {time.perf_counter() - t0:.0f}s", flush=True)
    decades = YEARS // 10
    profit = np.array([c[0] for c in cells]).reshape(len(COMP_RATES), len(EDGES), decades)
    active = np.array([c[1] for c in cells]).reshape(len(COMP_RATES), len(EDGES), decades)
    return profit, active


def render_frame(profit: np.ndarray, active: np.ndarray, decade: int, vmax: float) -> Image.Image:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=110)
    extent = (EDGES[0] * 100 - 0.7, EDGES[-1] * 100 + 0.7, COMP_RATES[0] * 100 - 4.3, COMP_RATES[-1] * 100 + 4.3)
    im1 = ax1.imshow(profit[:, :, decade], origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=vmax, extent=extent)
    best = np.unravel_index(np.argmax(profit[:, :, decade]), profit[:, :, decade].shape)
    ax1.plot(EDGES[best[1]] * 100, COMP_RATES[best[0]] * 100, marker="*", ms=14, color="#5ef2a0", mec="black")
    ax1.set_title(f"house profit per gambler per year, years {decade * 10 + 1}-{decade * 10 + 10}")
    ax1.set_xlabel("house edge (%)")
    ax1.set_ylabel("comps (% of theoretical loss returned)")
    fig.colorbar(im1, ax=ax1, label="$ per gambler per year")
    remaining = profit[:, :, decade] / np.maximum(profit[:, :, 0], 1e-9)
    im2 = ax2.imshow(remaining, origin="lower", aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0, extent=extent)
    ax2.set_title(f"profit this decade as a share of decade 1 (active share {active[:, :, decade].mean():.0%})")
    ax2.set_xlabel("house edge (%)")
    fig.colorbar(im2, ax=ax2, label="share of first-decade profit")
    fig.suptitle(f"Decade {decade + 1} of {profit.shape[2]}: the star marks the most profitable casino this decade", fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_outputs(profit: np.ndarray, active: np.ndarray) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    decades = profit.shape[2]
    vmax = float(profit.max())
    frames = [render_frame(profit, active, d, vmax) for d in range(decades)]
    frames[0].save(
        os.path.join(OUT_DIR, "landscape.gif"),
        save_all=True,
        append_images=frames[1:],
        duration=[900] * (decades - 1) + [2500],
        loop=0,
        optimize=True,
    )
    frames[-1].save(os.path.join(OUT_DIR, "landscape_final.png"))
    with open(os.path.join(OUT_DIR, "landscape.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["house_edge", "comp_rate", "decade", "profit_per_gambler_year", "active_share"])
        for i, c in enumerate(COMP_RATES):
            for j, e in enumerate(EDGES):
                for d in range(decades):
                    w.writerow([f"{e:.4f}", f"{c:.4f}", d + 1, f"{profit[i, j, d]:.3f}", f"{active[i, j, d]:.5f}"])


def load_saved() -> tuple[np.ndarray, np.ndarray]:
    """Read landscape.csv back into (comp, edge, decade) arrays."""
    with open(os.path.join(OUT_DIR, "landscape.csv")) as f:
        rows = list(csv.DictReader(f))
    decades = max(int(r["decade"]) for r in rows)
    profit = np.zeros((len(COMP_RATES), len(EDGES), decades))
    active = np.zeros_like(profit)
    for r in rows:
        i = int(np.argmin(np.abs(COMP_RATES - float(r["comp_rate"]))))
        j = int(np.argmin(np.abs(EDGES - float(r["house_edge"]))))
        profit[i, j, int(r["decade"]) - 1] = float(r["profit_per_gambler_year"])
        active[i, j, int(r["decade"]) - 1] = float(r["active_share"])
    return profit, active


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rerender", action="store_true", help="redraw from landscape.csv without simulating")
    args = parser.parse_args()
    if args.rerender:
        profit, active = load_saved()
    else:
        profit, active = simulate_grid(HouseRules(game="slots"), args.workers)
    save_outputs(profit, active)
    for d in (0, profit.shape[2] - 1):
        best = np.unravel_index(np.argmax(profit[:, :, d]), profit[:, :, d].shape)
        print(f"decade {d + 1}: best edge {EDGES[best[1]]:.3f}, comp {COMP_RATES[best[0]]:.2f}, "
              f"profit {profit[best][d]:.1f}/gambler/yr; grid mean active share {active[:, :, d].mean():.3f}")


if __name__ == "__main__":
    main()
