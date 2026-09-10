"""Follow one gambler month by month.

    python life.py --archetype CHASER --months 120 --seed 3

Runs the same engine as everything else with a population of one, prints a
diary of every month (bankroll, bet, what happened at the tables, comps,
how they feel about coming back) and saves a PNG of the bankroll path to
outputs/life_<ARCHETYPE>.png together with the diary text.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from casino import HouseRules
from engine import SimConfig, Simulation, StepRecord
from gamblers import Archetype

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


@dataclass
class Month:
    month: int
    start: float
    played: bool
    bet: float
    pnl: float
    outcome: str
    jackpots: int
    comps: float
    end: float
    p_return: float
    lapsed: bool
    bet_mult: float
    streak: int
    deficit: float


def describe(record: StepRecord, played: bool) -> str:
    if not played:
        return "stayed home"
    s = record.session
    if record.bust[0]:
        return "lost everything"
    if s.hit_limit[0]:
        return "hit the loss limit and left"
    if s.hit_target[0]:
        return "hit the win target and cashed out"
    return f"played the full session ({int(round(s.rounds_played[0]))} rounds)"


def live(archetype: Archetype, months: int, seed: int, rules: HouseRules) -> tuple[Simulation, list[Month]]:
    cfg = SimConfig(n_gamblers=1, n_periods=months, seed=seed, mix={archetype: 1.0})
    sim = Simulation(rules, cfg)
    diary: list[Month] = []
    for t in range(months):
        start = float(min(sim.pop.bankroll[0] + sim.pop.budget[0], sim.pop.cap[0]))
        record = sim.step()
        played = record.idx.size == 1
        pop = sim.pop
        diary.append(
            Month(
                month=t + 1,
                start=start,
                played=played,
                bet=float(record.bet[0]) if played else 0.0,
                pnl=float(record.session.pnl[0]) if played else 0.0,
                outcome=describe(record, played),
                jackpots=int(record.session.jackpots[0]) if played else 0,
                comps=float(record.free_play[0] + record.rebate[0]) if played else 0.0,
                end=float(pop.bankroll[0]),
                p_return=float(pop.p_return[0]),
                lapsed=bool(pop.lapsed[0]),
                bet_mult=float(pop.bet_mult[0]),
                streak=int(pop.streak[0]),
                deficit=float(pop.deficit[0]),
            )
        )
    return sim, diary


def mood(m: Month, archetype: Archetype) -> str:
    if m.lapsed and m.p_return == 0.0:
        return "has quit for good"
    if m.lapsed:
        return "has sworn off the casino; only a relapse brings them back"
    parts = []
    if archetype == Archetype.CHASER and m.streak < 0:
        parts.append(f"chasing, next bet x{m.bet_mult:.0f}")
    if archetype == Archetype.BREAK_EVEN and m.deficit > 0:
        parts.append(f"remembers being ${m.deficit:,.0f} down")
    if archetype == Archetype.HOT_HAND and m.streak > 0:
        parts.append(f"feels hot, {m.streak} wins running")
    if archetype == Archetype.GAMBLERS_FALLACY and m.streak < 0:
        parts.append(f"{-m.streak} losses running, a win feels due")
    if archetype == Archetype.LOSS_AVERSE and m.streak < 0:
        parts.append(f"stinging, bet shrunk to x{m.bet_mult:.2f}")
    if archetype == Archetype.HOUSE_MONEY and m.pnl > 0:
        parts.append("playing with the house's money next time")
    parts.append(f"{m.p_return:.0%} likely to be back next month")
    return "; ".join(parts)


def format_diary(archetype: Archetype, sim: Simulation, diary: list[Month], rules: HouseRules) -> str:
    pop = sim.pop
    lines = [
        f"One {archetype.name.replace('_', ' ').lower()} gambler at {rules.game} "
        f"(edge {rules.moments().edge:.1%}, comps {rules.comp_rate:.0%} of theoretical loss)",
        f"Monthly gambling budget ${pop.budget[0]:,.0f}, base bet ${pop.base_bet[0]:,.2f}, reserve cap ${pop.cap[0]:,.0f}",
        "",
    ]
    for m in diary:
        head = f"Month {m.month:3d}  bankroll ${m.start:8,.0f}"
        if not m.played:
            lines.append(f"{head}  {m.outcome:42s} -> ${m.end:8,.0f}   ({mood(m, archetype)})")
            continue
        jackpot = f" JACKPOT x{m.jackpots}" if m.jackpots else ""
        result = f"{'won' if m.pnl >= 0 else 'lost'} ${abs(m.pnl):,.0f}"
        comps = f", ${m.comps:,.0f} in comps" if m.comps >= 0.5 else ""
        lines.append(
            f"{head}  bet ${m.bet:6,.2f}: {m.outcome}, {result}{jackpot}{comps} -> ${m.end:8,.0f}   ({mood(m, archetype)})"
        )
    played = [m for m in diary if m.played]
    busts = sum(1 for m in diary if m.outcome == "lost everything")
    house = -sum(m.pnl for m in played) - sum(m.comps for m in played)
    lines += [
        "",
        f"{len(played)} visits in {len(diary)} months, went broke {busts} time(s), "
        f"house made ${house:,.0f} net of comps, final bankroll ${diary[-1].end:,.0f}",
    ]
    return "\n".join(lines)


def plot(archetype: Archetype, diary: list[Month], path: str) -> None:
    months = np.array([m.month for m in diary])
    end = np.array([m.end for m in diary])
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    ax.plot(months, end, color="#2c3e50", lw=1.4, label="bankroll at month end")
    visits = np.array([m.played for m in diary])
    ax.scatter(months[visits], end[visits], s=14, color="#2980b9", zorder=3, label="visited the casino")
    busts = np.array([m.outcome == "lost everything" for m in diary])
    if busts.any():
        ax.scatter(months[busts], np.zeros(busts.sum()), marker="x", s=60, color="#c0392b", zorder=4, label="went broke")
    lapsed = np.array([m.lapsed for m in diary])
    if lapsed.any():
        ax.fill_between(months, 0, end.max() * 1.05, where=lapsed, color="#bdc3c7", alpha=0.35, label="sworn off gambling")
    ax.set_xlabel("month")
    ax.set_ylabel("bankroll ($)")
    ax.set_title(f"One {archetype.name.replace('_', ' ').lower()} gambler, {len(diary)} months")
    ax.set_ylim(0, end.max() * 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--archetype", default="CHASER", choices=[a.name for a in Archetype])
    parser.add_argument("--months", type=int, default=120)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--game", default="slots")
    parser.add_argument("--edge", type=float, default=0.08)
    parser.add_argument("--comp-rate", type=float, default=0.2)
    args = parser.parse_args()
    archetype = Archetype[args.archetype]
    rules = HouseRules(game=args.game, house_edge=args.edge, comp_rate=args.comp_rate)
    sim, diary = live(archetype, args.months, args.seed, rules)
    text = format_diary(archetype, sim, diary, rules)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"life_{archetype.name}.txt"), "w") as f:
        f.write(text + "\n")
    plot(archetype, diary, os.path.join(OUT_DIR, f"life_{archetype.name}.png"))


if __name__ == "__main__":
    main()
