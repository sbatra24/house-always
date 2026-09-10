"""Casino games as pure vectorised functions, and the HouseRules that tune them.

How a session is simulated
--------------------------
A session is `rounds` plays of one bet. Every game is decomposed into two
parts:

* a diffusion: the frequent outcomes (red or black, a blackjack hand, a slot
  spin that pays nothing or a small prize), summarised by their per-round
  drift and standard deviation. Over a stretch of rounds their sum is close
  to normal, so a segment's closing P&L is one standard normal draw;
* jumps: rare big prizes, arriving as a Poisson process. The time to the
  next one is an exponential draw, the diffusion runs for exactly that
  long, then the prize lands. When a session expects more than
  JUMP_FOLD_THRESHOLD big prizes they are no longer rare, and they are
  folded into the diffusion instead. Jackpots are a second, much rarer
  stream, drawn as a Poisson count over the rounds actually played.

Within a segment the question "did the diffusion touch the win target or
the loss limit on the way?" has a closed form: conditional on its endpoint,
a Brownian path touched a level with probability exp(-2 * level * (level -
endpoint) / variance), the Brownian bridge result used for barrier options,
which does not depend on the drift. When both barriers may have been
touched, which came first is settled by the gambler's ruin probability, so
a huge bet against a nearby target degrades to the exact textbook answer
rather than to a coin flip. The tests compare all of this against
spin-by-spin simulation.

The exact per-round samplers at the bottom exist so the tests can check
that the moments the engine relies on are the real ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RED_POCKETS = 18
EUROPEAN_POCKETS = 37
AMERICAN_POCKETS = 38
STRAIGHT_UP_PAYOUT = 35
BLACKJACK_STD = 1.15
SMALL_PRIZE = 1.5
MAX_SEGMENTS = 12
JUMP_FOLD_THRESHOLD = 5.0
MAX_JACKPOTS_PER_SESSION = 3
EXP_CLIP = 700.0


@dataclass(frozen=True)
class BlackjackRules:
    """A rule set whose house edge is the sum of published rule effects.

    The baseline (six decks, dealer stands on soft 17, double after split,
    3:2 blackjack, no surrender) is 0.41% for basic strategy. Adjustments
    are the standard Wizard of Odds rule-variation figures, not a fresh
    combinatorial analysis. Standard deviation per hand is about 1.15 bets.
    """

    decks: int = 6
    dealer_hits_soft_17: bool = True
    blackjack_pays: float = 1.5
    double_after_split: bool = True
    late_surrender: bool = False

    def house_edge(self) -> float:
        deck_effect = {1: -0.0048, 2: -0.0019, 4: -0.0006, 6: 0.0, 8: 0.0002}
        edge = 0.0041 + deck_effect[self.decks]
        edge += 0.0022 if self.dealer_hits_soft_17 else 0.0
        edge += 0.0139 if self.blackjack_pays < 1.5 else 0.0
        edge += 0.0 if self.double_after_split else 0.0014
        edge -= 0.0008 if self.late_surrender else 0.0
        return edge


@dataclass(frozen=True)
class HouseRules:
    """Everything the house controls. Small on purpose.

    game: "roulette_eu", "roulette_us", "roulette_straight_us", "blackjack",
          "slots" or "generic".
    house_edge / volatility: used by slots and generic. For slots the
        volatility is the per-spin standard deviation per unit bet of the
        pay table excluding the jackpot.
    hit_frequency: slots, fraction of spins that pay anything.
    jackpot_size / jackpot_share: slots, jackpot in multiples of the bet and
        the share of every dollar staked that comes back through jackpots.
    comp_rate: free play as a fraction of theoretical loss (edge * handle).
    rebate_rate: fraction of a losing session refunded.
    rounds_per_session: rounds in one visit; None picks a game default.
    """

    game: str = "slots"
    house_edge: float = 0.08
    volatility: float = 6.0
    hit_frequency: float = 0.30
    jackpot_size: float = 1000.0
    jackpot_share: float = 0.02
    min_bet: float = 1.0
    max_bet: float = 100.0
    comp_rate: float = 0.20
    rebate_rate: float = 0.0
    rounds_per_session: int | None = None
    blackjack: BlackjackRules = field(default_factory=BlackjackRules)

    def moments(self) -> "RoundMoments":
        return round_moments(self)


@dataclass(frozen=True)
class RoundMoments:
    """Per-round statistics per unit bet, plus the session length.

    drift / std describe the diffusion part (the stake is always inside it,
    so drift is negative). jump_size / jump_prob describe big prizes paid
    gross on top of the diffusion; jackpot_size / jackpot_prob likewise.
    """

    drift: float
    std: float
    hit_frequency: float
    rounds: int
    jump_size: float = 0.0
    jump_prob: float = 0.0
    jackpot_size: float = 0.0
    jackpot_prob: float = 0.0

    @property
    def edge(self) -> float:
        """Expected loss per unit bet, all components included."""
        return -(self.drift + self.jump_size * self.jump_prob + self.jackpot_size * self.jackpot_prob)

    @property
    def session_std(self) -> float:
        """Standard deviation of a whole session per unit bet, every component."""
        jump_var = self.jump_prob * self.jump_size**2 + self.jackpot_prob * self.jackpot_size**2
        return np.sqrt(self.rounds * (self.std**2 + jump_var))

    @property
    def jumps_per_session(self) -> float:
        return self.jump_prob * self.rounds


DEFAULT_ROUNDS = {
    "roulette_eu": 60,
    "roulette_us": 60,
    "roulette_straight_us": 60,
    "blackjack": 100,
    "slots": 300,
    "generic": 100,
}


def even_money_moments(p_win: float) -> tuple[float, float]:
    """Drift and standard deviation of a +1/-1 bet that wins with p_win."""
    drift = 2.0 * p_win - 1.0
    return drift, np.sqrt(1.0 - drift * drift)


def round_moments(rules: HouseRules) -> RoundMoments:
    rounds = rules.rounds_per_session or DEFAULT_ROUNDS[rules.game]
    if rules.game == "roulette_eu":
        drift, std = even_money_moments(RED_POCKETS / EUROPEAN_POCKETS)
        return RoundMoments(drift, std, RED_POCKETS / EUROPEAN_POCKETS, rounds)
    if rules.game == "roulette_us":
        drift, std = even_money_moments(RED_POCKETS / AMERICAN_POCKETS)
        return RoundMoments(drift, std, RED_POCKETS / AMERICAN_POCKETS, rounds)
    if rules.game == "roulette_straight_us":
        p = 1.0 / AMERICAN_POCKETS
        drift = (STRAIGHT_UP_PAYOUT + 1) * p - 1.0
        std = (STRAIGHT_UP_PAYOUT + 1) * np.sqrt(p * (1.0 - p))
        return RoundMoments(drift, std, p, rounds)
    if rules.game == "blackjack":
        return RoundMoments(-rules.blackjack.house_edge(), BLACKJACK_STD, 0.43, rounds)
    if rules.game == "slots":
        prizes, probs = slots_pay_table(rules)
        small, big = prizes[1], prizes[2]
        p_small, p_big = probs[1], probs[2]
        jackpot_prob = rules.jackpot_share / rules.jackpot_size
        if p_big * rounds > JUMP_FOLD_THRESHOLD:
            # Frequent big prizes: the pay table was solved to have exactly this variance.
            drift = -(rules.house_edge + rules.jackpot_share)
            return RoundMoments(
                drift, rules.volatility, rules.hit_frequency, rounds,
                jackpot_size=rules.jackpot_size, jackpot_prob=jackpot_prob,
            )
        drift = -1.0 + small * p_small
        std = small * np.sqrt(p_small * (1.0 - p_small))
        return RoundMoments(
            drift=drift,
            std=std,
            hit_frequency=rules.hit_frequency,
            rounds=rounds,
            jump_size=big,
            jump_prob=p_big,
            jackpot_size=rules.jackpot_size,
            jackpot_prob=jackpot_prob,
        )
    if rules.game == "generic":
        return RoundMoments(-rules.house_edge, rules.volatility, 0.45, rounds)
    raise ValueError(f"unknown game {rules.game!r}")


def slots_pay_table(rules: HouseRules) -> tuple[np.ndarray, np.ndarray]:
    """A three-prize pay table matching the slot's return, variance and hit rate.

    Prizes (0, SMALL_PRIZE, big) with probabilities (1 - h, h - q, q). The
    small prize is fixed at 1.5 bets, the kind of payout that feels like a
    win and barely is. The big prize's size and rarity are solved so that
    the per-spin return excluding the jackpot is 1 - edge - jackpot_share
    and its standard deviation is the configured volatility:
        s*(h - q) + B*q = m,   s^2*(h - q) + B^2*q = m^2 + vol^2.
    """
    h = rules.hit_frequency
    s = SMALL_PRIZE
    mean = 1.0 - rules.house_edge - rules.jackpot_share
    second = rules.volatility**2 + mean**2
    if mean <= s * h or second <= s * s * h:
        raise ValueError("hit frequency too high for this return and volatility")
    big = (second - s * s * h) / (mean - s * h) - s
    q = (mean - s * h) / (big - s)
    if q >= h:
        raise ValueError("volatility too low for this hit frequency")
    return np.array([0.0, s, big]), np.array([1.0 - h, h - q, q])


def barrier_crossing(level: np.ndarray, endpoint: np.ndarray, inv_variance: np.ndarray) -> np.ndarray:
    """P(path touched +level | it ended at endpoint), for level > 0.

    exp(-2 * level * (level - endpoint) / variance), which is 1 when the
    endpoint is already beyond the level. To test a lower barrier pass
    `-endpoint` and the positive distance to it. Takes 1 / variance so the
    caller divides once for both barriers.
    """
    exponent = (-2.0 * level) * (level - endpoint) * inv_variance
    np.minimum(exponent, 0.0, out=exponent)
    return np.exp(exponent)


def ruin_probability(moments: RoundMoments, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    """P(the diffusion reaches +up before -down), distances in bets.

    Gambler's ruin for a diffusion with per-round drift mu < 0 and
    variance std^2: with theta = 2 * mu / std^2,
        P = (1 - exp(theta * down)) / (exp(-theta * up) - exp(theta * down)).
    For an even-money game exp(theta) equals q/p to five decimals, so this
    reproduces the textbook discrete answer when the barriers are whole
    bets. A fair game is handled by a vanishing theta, which gives the
    classical down / (up + down). Distant barriers underflow to the right
    limits; the one exponent that could overflow is capped.
    """
    theta = 2.0 * min(moments.drift, -1e-12) / (moments.std * moments.std)
    e_down = np.exp(theta * down)
    e_up = np.exp(np.minimum(-theta * up, EXP_CLIP))
    return (1.0 - e_down) / (e_up - e_down)


def expected_duration(moments: RoundMoments, up: np.ndarray, down: np.ndarray, p_up: np.ndarray) -> np.ndarray:
    """Wald's identity: E[rounds until a barrier] = E[P&L at the barrier] / drift."""
    if moments.drift > -1e-9:
        return up * down
    return (down - (up + down) * p_up) / (-moments.drift)


def poisson_count(lam: float | np.ndarray, u: np.ndarray, max_count: int) -> np.ndarray:
    """Poisson(lam) by inversion from uniforms, truncated at max_count.

    `lam` may be a scalar or one rate per gambler. For the jump rates in
    this project the truncation drops probability mass below 1e-5. The
    binomial it stands in for costs forty times more per gambler here.
    """
    count = np.zeros_like(u)
    term = np.exp(-lam) * np.ones_like(u)
    cdf = term.copy()
    for k in range(1, max_count + 1):
        count += u >= cdf
        term *= lam / k
        cdf += term
    return count


@dataclass
class SessionOutcome:
    """Result of a full session for a set of gamblers who played."""

    pnl: np.ndarray
    hit_target: np.ndarray
    hit_limit: np.ndarray
    rounds_played: np.ndarray
    jackpots: np.ndarray


def session_barriers_in_bets(
    bet: np.ndarray, bankroll: np.ndarray, target: np.ndarray, limit: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Round the target up to whole bets and cap the loss barrier at the bankroll.

    A win that overshoots a small target still ends the session, and nobody
    can lose more than they brought.
    """
    up = np.ceil(target / bet) * bet
    down = np.minimum(np.ceil(limit / bet) * bet, bankroll)
    return up, down


def resolve_segment(
    moments: RoundMoments,
    bet: np.ndarray,
    position: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    tau: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the diffusion for `tau` rounds from `position` (dollars since session start).

    Returns (new_position, hit_target, hit_limit, rounds_played). The
    endpoint comes from `z`; `u` samples the three-way barrier outcome
    (target from the bottom of the unit interval, limit from the top). A
    gambler who stops is credited the expected number of rounds to the
    barrier, from Wald's identity, capped at the segment length.
    """
    dist_up = up - position
    dist_down = down + position
    up_bets = dist_up / bet
    down_bets = dist_down / bet
    sd = bet * (moments.std * np.sqrt(tau))
    endpoint = bet * (moments.drift * tau) + sd * z
    inv_variance = 1.0 / (sd * sd)
    p_up = barrier_crossing(dist_up, endpoint, inv_variance)
    p_down = barrier_crossing(dist_down, -endpoint, inv_variance)
    p_first = ruin_probability(moments, up_bets, down_bets)
    both = p_up * p_down * p_first
    hit_target = u < p_up - p_up * p_down + both
    hit_limit = u >= 1.0 - p_down + both
    stopped = hit_target | hit_limit
    duration = expected_duration(moments, up_bets, down_bets, p_first)
    rounds = tau - stopped * np.maximum(tau - duration, 0.0)
    new_position = position + ~stopped * endpoint + hit_target * dist_up - hit_limit * dist_down
    return new_position, hit_target, hit_limit, rounds


def comps(
    rules: HouseRules, moments: RoundMoments, bet: np.ndarray, pnl: np.ndarray, rounds_played: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Free play and loss rebates owed for a session.

    Free play is comp_rate times theoretical loss (edge times the handle
    actually played). Rebates return rebate_rate of any net loss.
    """
    free_play = rules.comp_rate * moments.edge * bet * rounds_played
    rebate = rules.rebate_rate * np.maximum(-pnl, 0.0)
    return free_play, rebate


def spin_roulette(rng: np.random.Generator, n: int, pockets: int, straight_up: bool = False) -> np.ndarray:
    """Exact per-spin P&L per unit bet on red (or on a single number)."""
    spins = rng.integers(0, pockets, size=n)
    if straight_up:
        return np.where(spins == 0, STRAIGHT_UP_PAYOUT, -1).astype(float)
    return np.where(spins < RED_POCKETS, 1.0, -1.0)


def spin_slots(rng: np.random.Generator, n: int, rules: HouseRules) -> np.ndarray:
    """Exact per-spin P&L per unit bet, including the jackpot tail."""
    prizes, probs = slots_pay_table(rules)
    moments = round_moments(rules)
    base = rng.choice(prizes, size=n, p=probs) - 1.0
    jackpots = rng.random(n) < moments.jackpot_prob
    return base + jackpots * rules.jackpot_size
