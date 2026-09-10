"""Sweep the house edge with everything else fixed, 16,384 gamblers for 100 years.

Writes outputs/edge_sweep.csv with profit per gambler per year, the yearly
bankruptcy rate, the share of gamblers still active at the end, and busts
per gambler by archetype, for each edge. This is the plain version of the
question the optimiser answers: does more edge ever stop paying?
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
from dataclasses import replace
from multiprocessing import get_context

from casino import HouseRules
from engine import SimConfig, run
from gamblers import archetype_names

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
EDGES = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
GAMBLERS = 16_384
YEARS = 100
SEED = 5


def simulate(edge: float) -> dict:
    rules = replace(HouseRules(game="slots", max_bet=30.0, comp_rate=0.0, volatility=2.0), house_edge=edge)
    res = run(rules, SimConfig(n_gamblers=GAMBLERS, n_periods=YEARS * 12, seed=SEED))
    row = {
        "house_edge": edge,
        "profit_per_gambler_year": res.profit_per_gambler_year,
        "profit_first_decade": res.profit_by_period[:120].sum() / GAMBLERS / 10.0,
        "profit_last_decade": res.profit_by_period[-120:].sum() / GAMBLERS / 10.0,
        "ruin_rate_per_year": res.ruin_rate,
        "ruin_rate_first_decade": res.ruin_frac_by_year[:10].mean(),
        "ruin_rate_last_decade": res.ruin_frac_by_year[-10:].mean(),
        "active_share_end": res.active_by_period[-1] / GAMBLERS,
    }
    for name, busts, n in zip(archetype_names(), res.busts_by_arch, res.n_by_arch):
        row[f"busts_per_{name}"] = busts / n
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    with get_context("fork").Pool(args.workers) as pool:
        rows = pool.map(simulate, EDGES)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "edge_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    for row in rows:
        print(f"edge {row['house_edge']:.2f}: profit/yr {row['profit_per_gambler_year']:7.1f} "
              f"(decade 1 {row['profit_first_decade']:7.1f}, decade 10 {row['profit_last_decade']:7.1f})  "
              f"ruin/yr {row['ruin_rate_per_year']:.4f} (first decade {row['ruin_rate_first_decade']:.4f}, "
              f"last {row['ruin_rate_last_decade']:.4f})  active at end {row['active_share_end']:.3f}")


if __name__ == "__main__":
    main()
