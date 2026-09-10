"""Run N gamblers for T months against one set of HouseRules.

Layout
------
Gamblers are processed in chunks (default 32,768) so the working set stays
in cache; each chunk runs all T months before the next chunk starts. Each
month the gamblers who show up are gathered into a compact sub-population,
the session and psychology code runs on that subset with no masks, and the
state is scattered back.

Random numbers come from one PCG64 generator per block of RNG_BLOCK
gamblers, all seeded from a single SeedSequence, so a gambler's fate depends
only on the seed and their index: chunk size and worker count cannot change
a result.

Accounting
----------
Every dollar is tracked. Budgets flow in, reserve overflow leaks out (money
the gambler chooses not to keep at risk), the tables take or pay, and comps
flow back. `Results.conservation_error` is the residual of that identity.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np

from casino import (
    MAX_JACKPOTS_PER_SESSION,
    MAX_SEGMENTS,
    HouseRules,
    RoundMoments,
    SessionOutcome,
    comps,
    poisson_count,
    resolve_segment,
    session_barriers_in_bets,
)
from gamblers import (
    Archetype,
    Population,
    bet_size,
    drift_return_probability,
    psychology_table,
    session_barriers,
    update_history,
    update_return_probability,
)

RNG_BLOCK = 32768
N_ARCH = len(Archetype)


@dataclass(frozen=True)
class SimConfig:
    """Size, horizon, seed and chunking of one run."""

    n_gamblers: int = 100_000
    n_periods: int = 1200
    seed: int = 0
    chunk_size: int = RNG_BLOCK
    lifetime_months: int = 600
    mix: dict[Archetype, float] | None = None

    def __post_init__(self) -> None:
        if self.chunk_size % RNG_BLOCK and self.chunk_size < self.n_gamblers:
            raise ValueError(f"chunk_size must be a multiple of {RNG_BLOCK} or cover the whole population")

    @property
    def n_blocks(self) -> int:
        return -(-self.n_gamblers // RNG_BLOCK)


class RngBank:
    """One generator per RNG_BLOCK gamblers.

    `random` fills a full chunk-sized array block by block. The `*_for`
    methods draw only for the rows flagged in a boolean mask, in index
    order, so the draw sequence of a block depends on nothing outside it.
    `counts` turns a mask into per-block draw sizes once, to be reused by
    several draws for the same rows.
    """

    def __init__(self, seed: int, first_block: int, n_blocks: int, total_blocks: int) -> None:
        children = np.random.SeedSequence(seed).spawn(total_blocks)
        self.gens = [np.random.Generator(np.random.PCG64(children[first_block + i])) for i in range(n_blocks)]

    def random(self, out: np.ndarray) -> np.ndarray:
        for i, gen in enumerate(self.gens):
            gen.random(out=out[i * RNG_BLOCK : (i + 1) * RNG_BLOCK])
        return out

    def counts(self, mask: np.ndarray) -> list[int]:
        return [int(mask[i * RNG_BLOCK : (i + 1) * RNG_BLOCK].sum()) for i in range(len(self.gens))]

    def _draw(self, method: str, counts: list[int]) -> np.ndarray:
        parts = [getattr(gen, method)(k) for gen, k in zip(self.gens, counts)]
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def random_for(self, counts: list[int]) -> np.ndarray:
        return self._draw("random", counts)

    def normal_for(self, counts: list[int]) -> np.ndarray:
        return self._draw("standard_normal", counts)

    def exponential_for(self, counts: list[int]) -> np.ndarray:
        return self._draw("standard_exponential", counts)


@dataclass
class StepRecord:
    """What happened this month to the gamblers who played (rows = `idx`)."""

    idx: np.ndarray
    bet: np.ndarray
    session: SessionOutcome
    bust: np.ndarray
    free_play: np.ndarray
    rebate: np.ndarray


@dataclass
class Results:
    """Aggregates from a run. Additive across chunks via `merge`."""

    n_gamblers: int
    n_periods: int
    profit_by_period: np.ndarray
    handle_by_period: np.ndarray
    comps_by_period: np.ndarray
    visits_by_period: np.ndarray
    ruin_by_period: np.ndarray
    active_by_period: np.ndarray
    ruin_frac_by_year: np.ndarray
    profit_by_arch_period: np.ndarray
    n_by_arch: np.ndarray
    profit_by_arch: np.ndarray
    lifetime_profit_by_arch: np.ndarray
    visits_by_arch: np.ndarray
    busts_by_arch: np.ndarray
    first_ruin_hist: np.ndarray
    budget_in: float = 0.0
    leak_out: float = 0.0
    bankroll_start: float = 0.0
    bankroll_end: float = 0.0
    wall_time: float = 0.0

    SUMMED = (
        "profit_by_period", "handle_by_period", "comps_by_period", "visits_by_period",
        "ruin_by_period", "active_by_period", "profit_by_arch_period", "n_by_arch",
        "profit_by_arch", "lifetime_profit_by_arch", "visits_by_arch", "busts_by_arch",
        "first_ruin_hist",
    )
    SUMMED_SCALARS = ("budget_in", "leak_out", "bankroll_start", "bankroll_end")

    @classmethod
    def empty(cls, n_periods: int) -> "Results":
        years = n_periods // 12
        return cls(
            n_gamblers=0,
            n_periods=n_periods,
            profit_by_period=np.zeros(n_periods),
            handle_by_period=np.zeros(n_periods),
            comps_by_period=np.zeros(n_periods),
            visits_by_period=np.zeros(n_periods),
            ruin_by_period=np.zeros(n_periods),
            active_by_period=np.zeros(n_periods),
            ruin_frac_by_year=np.zeros(years),
            profit_by_arch_period=np.zeros((n_periods, N_ARCH)),
            n_by_arch=np.zeros(N_ARCH),
            profit_by_arch=np.zeros(N_ARCH),
            lifetime_profit_by_arch=np.zeros(N_ARCH),
            visits_by_arch=np.zeros(N_ARCH),
            busts_by_arch=np.zeros(N_ARCH),
            first_ruin_hist=np.zeros((N_ARCH, n_periods + 1)),
        )

    def merge(self, other: "Results") -> "Results":
        """Sum two partial results; ruin fractions are re-weighted by size."""
        n = self.n_gamblers + other.n_gamblers
        merged = Results.empty(self.n_periods)
        for name in self.SUMMED:
            setattr(merged, name, getattr(self, name) + getattr(other, name))
        for name in self.SUMMED_SCALARS:
            setattr(merged, name, getattr(self, name) + getattr(other, name))
        if n:
            merged.ruin_frac_by_year = (
                self.ruin_frac_by_year * self.n_gamblers + other.ruin_frac_by_year * other.n_gamblers
            ) / n
        merged.n_gamblers = n
        merged.wall_time = max(self.wall_time, other.wall_time)
        return merged

    @property
    def total_profit(self) -> float:
        return float(self.profit_by_period.sum())

    @property
    def gambler_periods(self) -> int:
        return self.n_gamblers * self.n_periods

    @property
    def profit_per_gambler_year(self) -> float:
        return self.total_profit / self.n_gamblers / (self.n_periods / 12.0)

    @property
    def ruin_rate(self) -> float:
        """Mean over years of the fraction of gamblers who went broke that year."""
        return float(self.ruin_frac_by_year.mean()) if self.ruin_frac_by_year.size else float("nan")

    @property
    def conservation_error(self) -> float:
        """budget_in - leak_out - house profit - (bankroll_end - bankroll_start)."""
        return self.budget_in - self.leak_out - self.total_profit - (self.bankroll_end - self.bankroll_start)

    def median_months_to_ruin(self) -> np.ndarray:
        """Per-archetype median month (1-based) of first ruin; inf if most never go broke."""
        out = np.full(N_ARCH, np.inf)
        for a in range(N_ARCH):
            hist = self.first_ruin_hist[a]
            total = hist.sum()
            if total == 0:
                continue
            cdf = np.cumsum(hist[:-1])
            if cdf[-1] >= total / 2.0:
                out[a] = int(np.searchsorted(cdf, total / 2.0)) + 1
        return out


class Simulation:
    """State and stepping logic for one chunk of gamblers."""

    def __init__(
        self,
        rules: HouseRules,
        cfg: SimConfig,
        first_block: int = 0,
        n_blocks: int | None = None,
    ) -> None:
        self.rules = rules
        self.cfg = cfg
        self.moments: RoundMoments = rules.moments()
        self.table = psychology_table()
        total_blocks = cfg.n_blocks
        n_blocks = total_blocks - first_block if n_blocks is None else n_blocks
        self.bank = RngBank(cfg.seed, first_block, n_blocks, total_blocks)
        sizes = [min(RNG_BLOCK, cfg.n_gamblers - (first_block + i) * RNG_BLOCK) for i in range(n_blocks)]
        self.pop = Population.concatenate(
            [Population.create(size, gen, self.moments, rules, cfg.mix) for size, gen in zip(sizes, self.bank.gens)]
        )
        n = self.pop.n
        self.cum_profit = np.zeros(n)
        self.first_ruin = np.full(n, -1, dtype=np.int64)
        self.ruined_year = np.zeros(n, dtype=bool)
        self.t = 0
        self.results = Results.empty(cfg.n_periods)
        self.results.n_gamblers = n
        self.results.n_by_arch = np.bincount(self.pop.archetype, minlength=N_ARCH).astype(float)
        self.results.bankroll_start = float(self.pop.bankroll.sum())
        self._u = np.empty(n)

    def step(self) -> StepRecord:
        """Advance every gambler in the chunk by one month."""
        pop, rules, moments, t = self.pop, self.rules, self.moments, self.t

        pop.bankroll += pop.budget
        leak = np.maximum(pop.bankroll - pop.cap, 0.0)
        pop.bankroll -= leak

        u_visit = self.bank.random(self._u)
        played = (u_visit < pop.p_return) & (pop.bankroll >= rules.min_bet)
        drift_return_probability(pop)
        idx = np.flatnonzero(played)
        sub = pop.take(idx, self.table)
        u_quit = u_visit[idx] / sub.p_return

        bet = bet_size(sub, moments, rules)
        target, limit = session_barriers(sub)
        session = self._play_session(bet, sub.bankroll, target, limit, played, idx)

        sub.bankroll += session.pnl
        bust = sub.bankroll <= 1e-9
        np.maximum(sub.bankroll, 0.0, out=sub.bankroll)
        free_play, rebate = comps(rules, moments, bet, session.pnl, session.rounds_played)
        comp_total = free_play + rebate
        sub.bankroll += comp_total

        update_history(sub, session.pnl)
        update_return_probability(sub, session.pnl, comp_total, moments.hit_frequency, bust, u_quit)
        pop.scatter(idx, sub)

        take = -session.pnl - comp_total
        self.cum_profit[idx] += take
        busted_idx = idx[bust]
        fresh = busted_idx[self.first_ruin[busted_idx] < 0]
        self.first_ruin[fresh] = t
        self.ruined_year[busted_idx] = True
        self._record(t, sub.archetype, take, bet * session.rounds_played, bust, comp_total, leak)

        self.t += 1
        return StepRecord(idx, bet, session, bust, free_play, rebate)

    def _play_session(
        self,
        bet: np.ndarray,
        bankroll: np.ndarray,
        target: np.ndarray,
        limit: np.ndarray,
        played: np.ndarray,
        idx: np.ndarray,
    ) -> SessionOutcome:
        """Run one session for the gamblers in `idx`, segment by segment.

        Each segment ends at the next big prize (an exponential draw in
        rounds) or at the end of the session, whichever comes first; the
        prize lands when the segment ends. Gamblers who hit a barrier drop
        out of later segments. Random numbers are drawn only for gamblers
        still playing, through a chunk-level mask, so each RNG block stays
        self-contained.
        """
        m, bank = self.moments, self.bank
        n = bet.shape[0]
        up, down = session_barriers_in_bets(bet, bankroll, target, limit)
        position = np.zeros(n)
        hit_target = np.zeros(n, dtype=bool)
        hit_limit = np.zeros(n, dtype=bool)
        rounds = np.zeros(n)
        remaining = np.full(n, float(m.rounds))
        counts = bank.counts(played)
        sel: slice | np.ndarray = slice(None)
        for segment in range(MAX_SEGMENTS):
            if segment:
                alive = ~(hit_target | hit_limit) & (remaining > 0.0)
                if not alive.any():
                    break
                sel = np.flatnonzero(alive)
                alive_full = np.zeros_like(played)
                alive_full[idx[sel]] = True
                counts = bank.counts(alive_full)
            rem = remaining[sel]
            if m.jump_prob and segment < MAX_SEGMENTS - 1:
                tau_jump = bank.exponential_for(counts) / m.jump_prob
                tau = np.minimum(tau_jump, rem)
                jumped = tau_jump < rem
            else:
                tau, jumped = rem, False
            z = bank.normal_for(counts)
            u = bank.random_for(counts)
            pos, ht, hl, r = resolve_segment(m, bet[sel], position[sel], up[sel], down[sel], tau, z, u)
            still = ~(ht | hl)
            pos += (still & jumped) * (m.jump_size * bet[sel])
            ht |= still & (pos >= up[sel])
            position[sel], hit_target[sel], hit_limit[sel] = pos, ht, hl
            rounds[sel] += r
            remaining[sel] = rem - tau
        if m.jackpot_prob:
            jackpots = poisson_count(m.jackpot_prob * rounds, bank.random_for(bank.counts(played)), MAX_JACKPOTS_PER_SESSION)
        else:
            jackpots = np.zeros(n)
        pnl = position + jackpots * (m.jackpot_size * bet)
        return SessionOutcome(pnl, hit_target, hit_limit, rounds, jackpots)

    def _record(
        self, t: int, arch: np.ndarray, take: np.ndarray, handle: np.ndarray,
        bust: np.ndarray, comp_total: np.ndarray, leak: np.ndarray,
    ) -> None:
        res, pop, cfg = self.results, self.pop, self.cfg
        res.profit_by_period[t] = take.sum()
        res.handle_by_period[t] = handle.sum()
        res.comps_by_period[t] = comp_total.sum()
        res.visits_by_period[t] = arch.shape[0]
        res.ruin_by_period[t] = bust.sum()
        res.active_by_period[t] = pop.n - pop.lapsed.sum()
        res.profit_by_arch_period[t] = np.bincount(arch, weights=take, minlength=N_ARCH)
        res.visits_by_arch += np.bincount(arch, minlength=N_ARCH)
        res.busts_by_arch += np.bincount(arch[bust], minlength=N_ARCH)
        res.budget_in += pop.budget.sum()
        res.leak_out += leak.sum()
        if (t + 1) % 12 == 0:
            res.ruin_frac_by_year[t // 12] = self.ruined_year.mean()
            self.ruined_year[:] = False
        if t + 1 == cfg.lifetime_months:
            res.lifetime_profit_by_arch = np.bincount(pop.archetype, weights=self.cum_profit, minlength=N_ARCH)

    def finish(self) -> Results:
        """Close out per-gambler statistics after the last step."""
        res, pop = self.results, self.pop
        res.profit_by_arch = np.bincount(pop.archetype, weights=self.cum_profit, minlength=N_ARCH)
        res.bankroll_end = float(pop.bankroll.sum())
        never = self.cfg.n_periods
        months = np.where(self.first_ruin < 0, never, self.first_ruin)
        for a in range(N_ARCH):
            res.first_ruin_hist[a] = np.bincount(months[pop.archetype == a], minlength=never + 1)
        return res

    def run(self) -> Results:
        for _ in range(self.cfg.n_periods):
            self.step()
        return self.finish()


def _chunk_plan(cfg: SimConfig) -> list[tuple[int, int]]:
    """(first_block, n_blocks) for every chunk."""
    per_chunk = cfg.chunk_size // RNG_BLOCK
    return [(b, min(per_chunk, cfg.n_blocks - b)) for b in range(0, cfg.n_blocks, per_chunk)]


def _run_chunks(args: tuple[HouseRules, SimConfig, list[tuple[int, int]]]) -> Results:
    os.environ["OMP_NUM_THREADS"] = "1"
    rules, cfg, plan = args
    out = Results.empty(cfg.n_periods)
    for first_block, n_blocks in plan:
        out = out.merge(Simulation(rules, cfg, first_block, n_blocks).run())
    return out


def run(rules: HouseRules, cfg: SimConfig, workers: int = 1) -> Results:
    """Simulate the whole population, optionally across processes."""
    start = time.perf_counter()
    plan = _chunk_plan(cfg)
    if workers <= 1 or len(plan) == 1:
        result = _run_chunks((rules, cfg, plan))
    else:
        shares = [plan[i::workers] for i in range(workers)]
        with get_context("fork").Pool(workers) as pool:
            parts = pool.map(_run_chunks, [(rules, cfg, s) for s in shares if s])
        result = Results.empty(cfg.n_periods)
        for part in parts:
            result = result.merge(part)
    result.wall_time = time.perf_counter() - start
    return result
