"""Population model: every gambler is one row in a set of NumPy arrays.

Nothing in this module loops over gamblers. Each behavioural archetype is a
point in a shared parameter space (`Psychology`), so the same vectorised
formulas produce a loss-averse accountant and a Martingale chaser depending
only on which parameter row a gambler was dealt at birth.

Archetypes and the literature they lean on
------------------------------------------
STEADY
    Control archetype with no documented bias: fixed bet, fixed win target
    and loss limit, mild loss aversion. Everything else is measured against it.
LOSS_AVERSE
    Prospect theory with the Tversky & Kahneman (1992) estimates alpha = beta
    = 0.88 and lambda = 2.25. Losses loom larger than gains, so this gambler
    shrinks the bet after a losing session, keeps a tight loss limit, and is
    the most likely to stop coming back after going broke.
GAMBLERS_FALLACY
    Belief in the law of small numbers (Tversky & Kahneman 1971). Croson &
    Sundali (2005) found it in casino roulette records: after a run of
    losses a win feels "due", so the bet goes up; after a run of wins a
    loss feels due, so the bet goes down and the session ends sooner.
HOT_HAND
    Gilovich, Vallone & Tversky (1985). After wins the gambler believes they
    are on a streak, so the bet goes up and there is no win target to stop
    at. Croson & Sundali observed both this and the fallacy in the same room.
CHASER
    Lesieur's "The Chase" (1984) and the Martingale: the bet doubles after
    every losing session, resets after a win, and there is no loss limit
    short of the whole bankroll. Losses barely dent the urge to return.
NEAR_MISS
    Reid (1986) on near misses and Dixon et al. (2010) on "losses disguised
    as wins": the reinforcing events are the individual paying spins, not
    the net result. Return probability is driven by the game's hit
    frequency and is nearly insensitive to the net loss.
BREAK_EVEN
    Thaler & Johnson (1990): after prior losses, gambles that offer a chance
    to get back to even become attractive. The bet is scaled so that one
    good session (one standard deviation) wipes out the remembered deficit,
    and the gambler leaves as soon as they are slightly ahead.
HOUSE_MONEY
    Thaler & Johnson (1990) again: after a gain, the winnings are coded as
    the house's money and risked more freely. The bet and the loss limit
    both grow with the last session's gain.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum

import numpy as np

from casino import HouseRules, RoundMoments

PT_ALPHA = 0.88
PT_BETA = 0.88
PT_LAMBDA = 2.25

BET_MULT_FLOOR = 0.2
BET_MULT_CEILING = 32.0
RETURN_DECAY = 0.2
START_BANKROLL_BUDGETS = 2.0
CAP_BUDGETS = 6.0
MEDIAN_BUDGET = 250.0
BUDGET_SIGMA = 0.6
MEDIAN_HANDLE_RATIO = 5.0
HANDLE_RATIO_SIGMA = 0.5


class Archetype(IntEnum):
    STEADY = 0
    LOSS_AVERSE = 1
    GAMBLERS_FALLACY = 2
    HOT_HAND = 3
    CHASER = 4
    NEAR_MISS = 5
    BREAK_EVEN = 6
    HOUSE_MONEY = 7


@dataclass(frozen=True)
class Psychology:
    """Behavioural parameters for one archetype.

    loss_escalation / win_escalation
        Multiplicative change of the bet after each consecutive losing or
        winning session. 1.0 doubles, -0.3 shrinks by 30%, 0 does nothing.
    house_money_share
        Extra bet after a gain: this many session standard deviations of
        extra stake per dollar of last session's gain, scaled by 1/sd. Also
        extends the loss limit by the same share of the gain.
    break_even_weight
        Bet needed so that one standard deviation of session luck erases
        the remembered deficit, times this weight.
    loss_aversion
        Prospect theory lambda used to value the last session.
    target_frac / limit_frac
        Win target and loss limit as fractions of the session-start bankroll.
    return_base
        Monthly probability of visiting when nothing remarkable happened.
    value_slope
        How strongly the prospect theory value of the last session moves the
        return probability (in logit units per value unit).
    hit_weight
        Logit bonus per unit of game hit frequency (near-miss reinforcement).
    comp_weight
        Logit bonus per monthly budget received in comps and rebates.
    quit_after_ruin
        Probability of lapsing (leaving the casino) after going broke.
    quit_forever
        Share of those lapses that are permanent: the gambler never returns.
    relapse_hazard
        Monthly probability that a lapsed (not permanently quit) gambler
        comes back.
    """

    loss_escalation: float
    win_escalation: float
    house_money_share: float
    break_even_weight: float
    loss_aversion: float
    target_frac: float
    limit_frac: float
    return_base: float
    value_slope: float
    hit_weight: float
    comp_weight: float
    quit_after_ruin: float
    quit_forever: float
    relapse_hazard: float


PSYCHOLOGY: dict[Archetype, Psychology] = {
    Archetype.STEADY: Psychology(0.0, 0.0, 0.0, 0.0, 1.5, 0.30, 0.30, 0.40, 0.5, 0.0, 0.3, 0.40, 0.5, 0.02),
    Archetype.LOSS_AVERSE: Psychology(-0.3, 0.0, 0.0, 0.0, PT_LAMBDA, 0.20, 0.20, 0.40, 1.0, 0.0, 0.3, 0.70, 0.6, 0.01),
    Archetype.GAMBLERS_FALLACY: Psychology(0.4, -0.25, 0.0, 0.0, 1.25, 0.40, 0.60, 0.45, 0.4, 0.0, 0.3, 0.30, 0.3, 0.03),
    Archetype.HOT_HAND: Psychology(0.0, 0.4, 0.0, 0.0, 1.25, 1.00, 0.40, 0.45, 0.4, 0.0, 0.3, 0.35, 0.3, 0.03),
    Archetype.CHASER: Psychology(1.0, 0.0, 0.0, 0.0, 0.8, 0.25, 1.00, 0.55, 0.2, 0.0, 0.3, 0.15, 0.1, 0.05),
    Archetype.NEAR_MISS: Psychology(0.0, 0.0, 0.0, 0.0, 1.0, 0.50, 0.60, 0.35, 0.2, 1.5, 0.4, 0.30, 0.3, 0.04),
    Archetype.BREAK_EVEN: Psychology(0.0, 0.0, 0.0, 1.0, 1.25, 0.10, 1.00, 0.45, 0.3, 0.0, 0.3, 0.30, 0.3, 0.03),
    Archetype.HOUSE_MONEY: Psychology(0.0, 0.0, 1.0, 0.0, 1.25, 0.50, 0.30, 0.45, 0.4, 0.0, 0.3, 0.35, 0.3, 0.03),
}

DEFAULT_MIX: dict[Archetype, float] = {a: 1.0 / len(Archetype) for a in Archetype}

STATE_FIELDS = ("bankroll", "bet_mult", "streak", "deficit", "last_pnl", "p_return", "lapsed")
FIXED_FIELDS = ("archetype", "budget", "cap", "base_bet", "return_base")
PSYCH_FIELDS = tuple(f.name for f in fields(Psychology)) + ("return_logit",)


def psychology_table() -> np.ndarray:
    """Behavioural parameters as a (len(PSYCH_FIELDS), len(Archetype)) array.

    Row order follows PSYCH_FIELDS; the last row is logit(return_base) so
    the hot loop never takes a logarithm. One fancy-index call along the
    archetype axis gathers every parameter for a set of gamblers at once.
    """
    rows = [[getattr(PSYCHOLOGY[a], name) for a in Archetype] for name in PSYCH_FIELDS[:-1]]
    table = np.array(rows)
    return np.vstack([table, logit(table[PSYCH_FIELDS.index("return_base")])])


def logit(p: np.ndarray | float) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


def prospect_value(x: np.ndarray, loss_aversion: np.ndarray | float) -> np.ndarray:
    """Tversky & Kahneman (1992) value function, alpha = beta = 0.88.

    `x` is measured in monthly budgets. Gains return x**alpha; losses return
    -lambda * |x|**beta. Written without np.where: one power, one sign, one
    blend, because this runs once per gambler per session.
    """
    magnitude = np.power(np.abs(x), PT_ALPHA)
    is_loss = x < 0
    weight = 1.0 + (loss_aversion - 1.0) * is_loss
    return np.sign(x) * magnitude * weight


@dataclass
class Population:
    """Column store for a set of gamblers. Every array has shape (n,).

    `psych` holds the behavioural parameters gathered per gambler from the
    archetype table; it is filled lazily by `take`, which is the only path
    through which the session code sees a population.
    """

    archetype: np.ndarray
    budget: np.ndarray
    cap: np.ndarray
    base_bet: np.ndarray
    return_base: np.ndarray
    bankroll: np.ndarray
    bet_mult: np.ndarray
    streak: np.ndarray
    deficit: np.ndarray
    last_pnl: np.ndarray
    p_return: np.ndarray
    lapsed: np.ndarray
    psych: dict[str, np.ndarray]

    @property
    def n(self) -> int:
        return self.archetype.shape[0]

    @classmethod
    def create(
        cls,
        n: int,
        rng: np.random.Generator,
        moments: RoundMoments,
        rules: HouseRules,
        mix: dict[Archetype, float] | None = None,
    ) -> "Population":
        """Deal archetypes and draw heterogeneous budgets and bet sizes.

        Monthly gambling budgets are lognormal with median 250 and sigma 0.6.
        The base bet is set so that a full session's handle is a lognormal
        multiple (median 5) of the monthly budget, then clipped to the table
        limits. The starting bankroll is two budgets and the reserve cap six.
        """
        weights = np.array([(mix or DEFAULT_MIX).get(a, 0.0) for a in Archetype], dtype=float)
        weights /= weights.sum()
        archetype = rng.choice(len(Archetype), size=n, p=weights).astype(np.int8)
        budget = MEDIAN_BUDGET * np.exp(BUDGET_SIGMA * rng.standard_normal(n))
        handle_ratio = MEDIAN_HANDLE_RATIO * np.exp(HANDLE_RATIO_SIGMA * rng.standard_normal(n))
        base_bet = np.clip(budget * handle_ratio / moments.rounds, rules.min_bet, rules.max_bet)
        return_base = psychology_table()[PSYCH_FIELDS.index("return_base")][archetype]
        return cls(
            archetype=archetype,
            budget=budget,
            cap=CAP_BUDGETS * budget,
            base_bet=base_bet,
            return_base=return_base,
            bankroll=START_BANKROLL_BUDGETS * budget,
            bet_mult=np.ones(n),
            streak=np.zeros(n),
            deficit=np.zeros(n),
            last_pnl=np.zeros(n),
            p_return=return_base.copy(),
            lapsed=np.zeros(n, dtype=bool),
            psych={},
        )

    def take(self, idx: np.ndarray, table: np.ndarray) -> "Population":
        """Gather the rows in `idx` into a compact population with psychology attached."""
        cols = {name: getattr(self, name)[idx] for name in FIXED_FIELDS + STATE_FIELDS}
        gathered = table[:, cols["archetype"].astype(np.intp)]
        psych = dict(zip(PSYCH_FIELDS, gathered))
        return Population(psych=psych, **cols)

    def scatter(self, idx: np.ndarray, sub: "Population") -> None:
        """Write the mutable state of `sub` back into rows `idx`."""
        for name in STATE_FIELDS:
            getattr(self, name)[idx] = getattr(sub, name)

    @staticmethod
    def concatenate(parts: list["Population"]) -> "Population":
        if len(parts) == 1:
            return parts[0]
        cols = {
            name: np.concatenate([getattr(p, name) for p in parts]) for name in FIXED_FIELDS + STATE_FIELDS
        }
        return Population(psych={}, **cols)


def bet_size(pop: Population, moments: RoundMoments, rules: HouseRules) -> np.ndarray:
    """Bet for the coming session as a function of the gambler's recent history.

    base_bet * bet_mult carries streak escalation (chasing, fallacy, hot
    hand, loss-averse shrinkage). House-money gamblers add a slice of last
    session's gain; break-even gamblers bet enough that one standard
    deviation of luck would erase their remembered deficit. The result is
    clipped to the table limits and to what the gambler actually has.
    """
    sd_unit = moments.session_std
    bet = pop.base_bet * pop.bet_mult
    bet += pop.psych["house_money_share"] * np.maximum(pop.last_pnl, 0.0) / sd_unit
    np.maximum(bet, pop.psych["break_even_weight"] * pop.deficit / sd_unit, out=bet)
    np.clip(bet, rules.min_bet, rules.max_bet, out=bet)
    np.minimum(bet, pop.bankroll, out=bet)
    return bet


def session_barriers(pop: Population) -> tuple[np.ndarray, np.ndarray]:
    """Dollar win target and dollar loss limit for this session.

    The loss limit never exceeds the bankroll; a limit equal to the bankroll
    means the gambler is prepared to go broke. House-money gamblers extend
    their limit by last session's gain, because that money feels free.
    """
    target = pop.psych["target_frac"] * pop.bankroll
    limit = pop.psych["limit_frac"] * pop.bankroll
    limit += pop.psych["house_money_share"] * np.maximum(pop.last_pnl, 0.0)
    np.minimum(limit, pop.bankroll, out=limit)
    return target, limit


def update_history(pop: Population, pnl: np.ndarray) -> None:
    """Streak, bet multiplier and deficit updates after a session.

    A session that continues the current streak multiplies the bet
    multiplier by (1 + escalation); a session that breaks it resets the
    multiplier to (1 + escalation) for the new direction. The deficit is
    the running loss since the gambler was last at or above even.
    """
    won = pnl > 0
    lost = pnl < 0
    continued = (won & (pop.streak > 0)) | (lost & (pop.streak < 0))
    escalation = pop.psych["win_escalation"] * won + pop.psych["loss_escalation"] * lost
    base = 1.0 + continued * (pop.bet_mult - 1.0)
    pop.bet_mult = np.clip(base * (1.0 + escalation), BET_MULT_FLOOR, BET_MULT_CEILING)
    pop.streak = continued * pop.streak + won - lost
    pop.deficit = np.maximum(pop.deficit - pnl, 0.0)
    pop.last_pnl = pnl


def update_return_probability(
    pop: Population,
    pnl: np.ndarray,
    comps: np.ndarray,
    hit_frequency: float,
    bust: np.ndarray,
    u_quit: np.ndarray,
) -> None:
    """Decide how likely each gambler who just played is to come back.

    logit(p) = logit(base) + slope * v(pnl / budget) + hit_weight *
    hit_frequency + comp_weight * comps / budget. Busted players lapse with
    probability quit_after_ruin; a share quit_forever of those lapses set
    the return probability to zero for good, the rest to the relapse
    hazard. Anyone who played is by definition no longer lapsed. `u_quit`
    is one uniform per player; both decisions are read off it in sequence.
    """
    psych = pop.psych
    z = psych["return_logit"] + psych["value_slope"] * prospect_value(pnl / pop.budget, psych["loss_aversion"])
    z += psych["hit_weight"] * hit_frequency
    z += psych["comp_weight"] * comps / pop.budget
    p = 1.0 / (1.0 + np.exp(-z))
    quits = bust & (u_quit < psych["quit_after_ruin"])
    forever = quits & (u_quit < psych["quit_after_ruin"] * psych["quit_forever"])
    pop.lapsed = quits
    pop.p_return = p + quits * (psych["relapse_hazard"] - p)
    pop.p_return *= ~forever


def drift_return_probability(pop: Population) -> None:
    """Pull every non-lapsed gambler's return probability toward their base rate.

    Applied to the whole population before the month's players overwrite
    their own entry, so it only takes effect for those who stayed home.
    """
    pop.p_return += RETURN_DECAY * (pop.return_base - pop.p_return) * ~pop.lapsed


def archetype_names() -> list[str]:
    return [a.name for a in Archetype]
