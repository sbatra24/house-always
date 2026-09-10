"""The big one: 1,000,000 gamblers for 10,000 months, as twelve independent casinos.

Twelve cohorts of 83,333 or 83,334 gamblers each run for 10,000 months
(833.3 years) with their own seed. Twelve casinos times 833.3 years is
10,000 casino-years; one million gamblers times 10,000 months is 10^10
gambler-months, or 833 million gambler-years. Each cohort is one process
task; two run at a time.

By default the rules are the ruin-constrained optimum found by optimize.py
(outputs/optimum_rules.json, cap 5% bankrupted per year), falling back to
HouseRules() when that file is missing.

Writes outputs/full_run_raw.npz (the merged Results arrays, saved before
any reporting runs), outputs/full_run.json, outputs/full_run_timeseries.csv
and outputs/full_run.png. `--summarise-only` rebuilds the reports from the
saved arrays.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from multiprocessing import get_context

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from casino import HouseRules
from engine import Results, SimConfig, run
from gamblers import archetype_names

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
TOTAL_GAMBLERS = 1_000_000
COHORTS = 12
MONTHS = 10_000
BASE_SEED = 100
OPTIMUM_LABEL = "ruin<=0.05"


def load_rules() -> tuple[HouseRules, str]:
    path = os.path.join(OUT_DIR, "optimum_rules.json")
    base = HouseRules(game="slots")
    if not os.path.exists(path):
        return base, "defaults (optimum_rules.json not found)"
    with open(path) as f:
        winner = json.load(f)["winners"].get(OPTIMUM_LABEL)
    if winner is None:
        return base, f"defaults (no feasible {OPTIMUM_LABEL} optimum)"
    return replace(base, **winner["rules"]), f"optimizer winner {OPTIMUM_LABEL}"


def cohort_sizes() -> list[int]:
    base, extra = divmod(TOTAL_GAMBLERS, COHORTS)
    return [base + (1 if i < extra else 0) for i in range(COHORTS)]


def run_cohort(args: tuple[int, int, HouseRules, int]) -> Results:
    index, size, rules, months = args
    cfg = SimConfig(n_gamblers=size, n_periods=months, seed=BASE_SEED + index)
    res = run(rules, cfg)
    print(f"cohort {index + 1:2d}/{COHORTS}: {size} gamblers, {res.wall_time:.0f}s, "
          f"profit/gambler/yr {res.profit_per_gambler_year:.1f}, ruin {res.ruin_rate:.4f}", flush=True)
    return res


def summarise(res: Results, rules: HouseRules, source: str, wall: float, months: int) -> dict:
    names = archetype_names()
    years = months / 12.0
    per = res.n_by_arch
    medians = res.median_months_to_ruin()
    by_arch = {
        name: {
            "gamblers": int(per[i]),
            "lifetime_50yr_profit_per_gambler": float(res.lifetime_profit_by_arch[i] / per[i]),
            "full_horizon_profit_per_gambler": float(res.profit_by_arch[i] / per[i]),
            "visits_per_gambler_per_year": float(res.visits_by_arch[i] / per[i] / years),
            "busts_per_gambler": float(res.busts_by_arch[i] / per[i]),
            "share_ever_ruined": float(1.0 - res.first_ruin_hist[i, -1] / per[i]),
            "median_months_to_first_ruin": None if np.isinf(medians[i]) else int(medians[i]),
        }
        for i, name in enumerate(names)
    }
    n_decades = months // 120
    decades = res.profit_by_period[: n_decades * 120].reshape(n_decades, 120).sum(axis=1) / res.n_gamblers / 10.0
    active = res.active_by_period[: n_decades * 120].reshape(n_decades, 120)[:, -1] / res.n_gamblers
    return {
        "rules": asdict(rules),
        "rules_source": source,
        "gamblers": res.n_gamblers,
        "months": months,
        "cohorts": COHORTS,
        "casino_years": COHORTS * years,
        "gambler_months": res.gambler_periods,
        "gambler_years": res.gambler_periods / 12.0,
        "wall_time_seconds": wall,
        "gambler_months_per_second": res.gambler_periods / wall,
        "gamblers_per_second": res.n_gamblers / wall,
        "house_profit_total": res.total_profit,
        "house_profit_per_gambler_year": res.profit_per_gambler_year,
        "handle_total": float(res.handle_by_period.sum()),
        "comps_total": float(res.comps_by_period.sum()),
        "hold_on_handle": float(res.total_profit / res.handle_by_period.sum()),
        "ruin_rate_per_year": res.ruin_rate,
        "ruin_events_total": float(res.ruin_by_period.sum()),
        "share_ever_ruined": float(1.0 - res.first_ruin_hist[:, -1].sum() / res.n_gamblers),
        "active_share_end": float(res.active_by_period[-1] / res.n_gamblers),
        "conservation_error_dollars": res.conservation_error,
        "conservation_error_relative": res.conservation_error / res.budget_in,
        "profit_per_gambler_year_by_decade": [float(x) for x in decades],
        "active_share_by_decade": [float(x) for x in active],
        "by_archetype": by_arch,
    }


def save_timeseries(res: Results, path: str) -> None:
    years = res.n_periods // 12
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "profit_per_gambler", "visits_per_gambler", "ruin_events_per_gambler", "active_share_year_end"])
        for y in range(years):
            sl = slice(12 * y, 12 * y + 12)
            w.writerow([
                y + 1,
                f"{res.profit_by_period[sl].sum() / res.n_gamblers:.4f}",
                f"{res.visits_by_period[sl].sum() / res.n_gamblers:.5f}",
                f"{res.ruin_by_period[sl].sum() / res.n_gamblers:.6f}",
                f"{res.active_by_period[12 * y + 11] / res.n_gamblers:.6f}",
            ])


def plot(res: Results, summary: dict, path: str) -> None:
    years = res.n_periods // 12
    yearly_profit = res.profit_by_period[: years * 12].reshape(years, 12).sum(axis=1) / res.n_gamblers
    active = res.active_by_period[: years * 12].reshape(years, 12)[:, -1] / res.n_gamblers
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), dpi=120, sharex=True)
    ax1.plot(np.arange(1, years + 1), yearly_profit, color="#c0392b", lw=0.9)
    ax1.set_ylabel("house profit per gambler ($/year)")
    ax1.set_title(f"{res.n_gamblers:,} gamblers, {years} years, {summary['cohorts']} casinos: "
                  f"${summary['house_profit_per_gambler_year']:,.0f} per gambler per year on average")
    ax1.grid(alpha=0.3)
    ax2.plot(np.arange(1, years + 1), active, color="#2980b9", lw=0.9)
    ax2.set_ylabel("share still gambling")
    ax2.set_xlabel("year")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


RAW_PATH = os.path.join(OUT_DIR, "full_run_raw.npz")
RAW_ARRAYS = Results.SUMMED
RAW_SCALARS = Results.SUMMED_SCALARS + ("n_gamblers", "n_periods", "wall_time")


def save_raw(res: Results, rules: HouseRules, source: str, cpu: float) -> None:
    arrays = {name: getattr(res, name) for name in RAW_ARRAYS}
    arrays["ruin_frac_by_year"] = res.ruin_frac_by_year
    scalars = {name: float(getattr(res, name)) for name in RAW_SCALARS}
    scalars["cpu_seconds"] = cpu
    np.savez_compressed(RAW_PATH, rules=json.dumps(asdict(rules)), source=source, scalars=json.dumps(scalars), **arrays)


def load_raw() -> tuple[Results, HouseRules, str, float]:
    with np.load(RAW_PATH) as data:
        scalars = json.loads(str(data["scalars"]))
        res = Results.empty(int(scalars["n_periods"]))
        for name in RAW_ARRAYS:
            setattr(res, name, data[name])
        res.ruin_frac_by_year = data["ruin_frac_by_year"]
        for name in RAW_SCALARS:
            setattr(res, name, scalars[name])
        res.n_gamblers = int(res.n_gamblers)
        res.n_periods = int(res.n_periods)
        rules = HouseRules(**{k: v for k, v in json.loads(str(data["rules"])).items() if k != "blackjack"})
        return res, rules, str(data["source"]), scalars["cpu_seconds"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--months", type=int, default=MONTHS)
    parser.add_argument("--scale", type=float, default=1.0, help="fraction of the million gamblers (for quick checks)")
    parser.add_argument("--summarise-only", action="store_true", help="rebuild reports from outputs/full_run_raw.npz")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.summarise_only:
        total, rules, source, cpu = load_raw()
        wall = total.wall_time
    else:
        rules, source = load_rules()
        print(f"rules: {source}: {rules}", flush=True)
        sizes = [max(1, int(round(s * args.scale))) for s in cohort_sizes()]
        tasks = [(i, s, rules, args.months) for i, s in enumerate(sizes)]
        t0 = time.perf_counter()
        with get_context("fork").Pool(args.workers) as pool:
            parts = pool.map(run_cohort, tasks, chunksize=1)
        wall = time.perf_counter() - t0
        cpu = os.times().children_user + os.times().children_system
        total = Results.empty(args.months)
        for part in parts:
            total = total.merge(part)
        total.wall_time = wall
        save_raw(total, rules, source, cpu)
    months = total.n_periods
    summary = summarise(total, rules, source, wall, months)
    summary["cpu_seconds_all_workers"] = cpu
    summary["nanoseconds_cpu_per_gambler_month"] = 1e9 * cpu / total.gambler_periods
    with open(os.path.join(OUT_DIR, "full_run.json"), "w") as f:
        json.dump(summary, f, indent=2)
    save_timeseries(total, os.path.join(OUT_DIR, "full_run_timeseries.csv"))
    plot(total, summary, os.path.join(OUT_DIR, "full_run.png"))
    print(f"\n{total.n_gamblers:,} gamblers x {months:,} months in {wall:.0f}s wall, {cpu:.0f}s CPU across workers: "
          f"{summary['gambler_months_per_second'] / 1e6:.1f}M gambler-months/s, "
          f"{summary['gamblers_per_second']:,.0f} gamblers/s, {summary['nanoseconds_cpu_per_gambler_month']:.0f} ns CPU per gambler-month")
    print(f"house profit ${total.total_profit:,.0f} (${total.profit_per_gambler_year:,.1f} per gambler per year), "
          f"ruin rate {total.ruin_rate:.4f}/yr, conservation error ${total.conservation_error:.4f}")
    for name, stats in summary["by_archetype"].items():
        print(f"  {name:17s} 50yr loss ${stats['lifetime_50yr_profit_per_gambler']:9,.0f}  "
              f"median months to ruin {stats['median_months_to_first_ruin']}  "
              f"busts/gambler {stats['busts_per_gambler']:7.1f}  visits/yr {stats['visits_per_gambler_per_year']:.2f}")


if __name__ == "__main__":
    main()
