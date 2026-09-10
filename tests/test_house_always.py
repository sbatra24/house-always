"""pytest suite: game edges, conservation of money, reproducibility, behaviour, chunking."""
from __future__ import annotations

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from casino import (
    BlackjackRules,
    HouseRules,
    poisson_count,
    round_moments,
    ruin_probability,
    slots_pay_table,
    spin_roulette,
    spin_slots,
)
from engine import RNG_BLOCK, Results, SimConfig, Simulation, run
from gamblers import Archetype, Population, bet_size, prospect_value, psychology_table, update_history
from optimize import GaussianProcess, pareto_front


# ---------------------------------------------------------------- games


def test_european_roulette_red_edge_is_one_in_37():
    rng = np.random.default_rng(0)
    n = 2_000_000
    pnl = spin_roulette(rng, n, 37)
    expected = -1.0 / 37
    se = pnl.std() / np.sqrt(n)
    assert abs(pnl.mean() - expected) < 4 * se
    m = round_moments(HouseRules(game="roulette_eu"))
    assert m.edge == pytest.approx(1.0 / 37)
    assert m.std == pytest.approx(pnl.std(), rel=1e-3)


def test_american_roulette_edges_match_wheel():
    assert round_moments(HouseRules(game="roulette_us")).edge == pytest.approx(2.0 / 38)
    assert round_moments(HouseRules(game="roulette_straight_us")).edge == pytest.approx(2.0 / 38)
    rng = np.random.default_rng(1)
    pnl = spin_roulette(rng, 2_000_000, 38, straight_up=True)
    assert abs(pnl.mean() + 2.0 / 38) < 4 * pnl.std() / np.sqrt(pnl.size)


def test_slot_pay_table_matches_configured_moments():
    rules = HouseRules(game="slots", house_edge=0.08, volatility=6.0, hit_frequency=0.3, jackpot_share=0.02)
    prizes, probs = slots_pay_table(rules)
    assert probs.sum() == pytest.approx(1.0)
    assert (prizes >= 0).all() and (probs > 0).all()
    mean = prizes @ probs
    var = (prizes**2) @ probs - mean**2
    assert mean == pytest.approx(1.0 - 0.08 - 0.02)
    assert np.sqrt(var) == pytest.approx(6.0)
    assert probs[1:].sum() == pytest.approx(0.3)
    rng = np.random.default_rng(2)
    pnl = spin_slots(rng, 4_000_000, rules)
    m = round_moments(rules)
    assert m.edge == pytest.approx(0.08)
    assert abs(pnl.mean() + 0.08) < 4 * pnl.std() / np.sqrt(pnl.size)


def test_blackjack_rule_effects():
    base = BlackjackRules(dealer_hits_soft_17=False)
    assert base.house_edge() == pytest.approx(0.0041)
    assert BlackjackRules(dealer_hits_soft_17=False, blackjack_pays=1.2).house_edge() - base.house_edge() == pytest.approx(0.0139)
    assert BlackjackRules(dealer_hits_soft_17=False, decks=1).house_edge() < base.house_edge()
    assert BlackjackRules(dealer_hits_soft_17=False, late_surrender=True).house_edge() < base.house_edge()


def test_full_session_mean_matches_edge_for_every_game():
    """Barriers pushed out of reach: the session P&L per unit handle must equal -edge."""
    for game in ("roulette_eu", "roulette_us", "blackjack", "slots", "generic"):
        rules = HouseRules(game=game, house_edge=0.05, volatility=4.0)
        sim = Simulation(rules, SimConfig(n_gamblers=RNG_BLOCK, n_periods=1, seed=5))
        n = RNG_BLOCK
        bet = np.full(n, 10.0)
        huge = np.full(n, 1e12)
        out = sim._play_session(bet, huge, huge, huge, np.ones(n, dtype=bool), np.arange(n))
        m = rules.moments()
        handle = bet * m.rounds
        assert np.allclose(out.rounds_played, m.rounds)
        se = out.pnl.std() / np.sqrt(n)
        assert abs(out.pnl.mean() + m.edge * handle.mean()) < 4 * se, game


def test_ruin_probability_matches_textbook_gamblers_ruin():
    m = round_moments(HouseRules(game="roulette_us"))
    p, q = 18 / 38, 20 / 38
    for a, b in ((1, 3), (5, 20), (2, 8)):
        exact = (1 - (q / p) ** b) / (1 - (q / p) ** (a + b))
        assert ruin_probability(m, np.array([a]), np.array([b]))[0] == pytest.approx(exact, rel=2e-3)


def test_barrier_resolution_matches_spin_by_spin_simulation():
    """A roulette session with a win target and loss limit, exact paths versus the engine."""
    rules = HouseRules(game="roulette_us")
    rng = np.random.default_rng(9)
    bet, bankroll, target, limit = 10.0, 1000.0, 100.0, 1000.0
    n_paths, rounds = 20_000, rules.moments().rounds
    steps = np.where(rng.random((n_paths, rounds)) < 18 / 38, bet, -bet)
    path = np.cumsum(steps, axis=1)
    hit_up = (path >= target).any(axis=1)
    exact_p_up = hit_up.mean()
    n = RNG_BLOCK
    sim = Simulation(rules, SimConfig(n_gamblers=n, n_periods=1, seed=5))
    out = sim._play_session(
        np.full(n, bet), np.full(n, bankroll), np.full(n, target), np.full(n, limit), np.ones(n, dtype=bool), np.arange(n)
    )
    se = np.sqrt(exact_p_up * (1 - exact_p_up) * (1 / n_paths + 1 / n))
    assert abs(out.hit_target.mean() - exact_p_up) < 4 * se
    assert out.pnl[out.hit_target].min() >= target


def test_poisson_count_has_the_right_mean():
    rng = np.random.default_rng(3)
    u = rng.random(1_000_000)
    for lam in (0.01, 0.6, 2.0):
        counts = poisson_count(lam, u, 12)
        assert abs(counts.mean() - lam) < 4 * np.sqrt(lam / u.size)


# ---------------------------------------------------------------- engine


def small_cfg(**kw) -> SimConfig:
    base = dict(n_gamblers=8192, n_periods=60, seed=42)
    base.update(kw)
    return SimConfig(**base)


def test_money_is_conserved():
    for game in ("slots", "roulette_us"):
        res = run(HouseRules(game=game, rebate_rate=0.1), small_cfg())
        assert abs(res.conservation_error) < 1e-6 * res.budget_in
        assert res.total_profit > 0


def test_same_seed_same_result_different_seed_different_result():
    a = run(HouseRules(), small_cfg())
    b = run(HouseRules(), small_cfg())
    c = run(HouseRules(), small_cfg(seed=43))
    assert np.array_equal(a.profit_by_period, b.profit_by_period)
    assert np.array_equal(a.first_ruin_hist, b.first_ruin_hist)
    assert not np.array_equal(a.profit_by_period, c.profit_by_period)


def test_chunked_and_unchunked_runs_agree():
    cfg_one = SimConfig(n_gamblers=2 * RNG_BLOCK, n_periods=24, seed=7, chunk_size=2 * RNG_BLOCK)
    cfg_two = SimConfig(n_gamblers=2 * RNG_BLOCK, n_periods=24, seed=7, chunk_size=RNG_BLOCK)
    one = run(HouseRules(), cfg_one)
    two = run(HouseRules(), cfg_two)
    assert np.array_equal(one.first_ruin_hist, two.first_ruin_hist)
    assert np.array_equal(one.visits_by_period, two.visits_by_period)
    assert np.allclose(one.profit_by_period, two.profit_by_period, rtol=1e-10, atol=1e-6)
    assert np.allclose(one.profit_by_arch, two.profit_by_arch, rtol=1e-10, atol=1e-6)


def test_parallel_workers_match_serial():
    cfg = SimConfig(n_gamblers=2 * RNG_BLOCK, n_periods=24, seed=8)
    serial = run(HouseRules(), cfg, workers=1)
    parallel = run(HouseRules(), cfg, workers=2)
    assert np.array_equal(serial.first_ruin_hist, parallel.first_ruin_hist)
    assert np.allclose(serial.profit_by_period, parallel.profit_by_period, rtol=1e-10, atol=1e-6)


def test_results_merge_is_additive():
    a = run(HouseRules(), SimConfig(n_gamblers=4096, n_periods=24, seed=1))
    b = run(HouseRules(), SimConfig(n_gamblers=4096, n_periods=24, seed=2))
    m = Results.empty(24).merge(a).merge(b)
    assert m.n_gamblers == 8192
    assert m.total_profit == pytest.approx(a.total_profit + b.total_profit)
    assert m.ruin_rate == pytest.approx((a.ruin_rate + b.ruin_rate) / 2)


def test_higher_edge_takes_more_per_visit_and_comps_cost_money():
    low = run(HouseRules(house_edge=0.03), small_cfg())
    high = run(HouseRules(house_edge=0.12), small_cfg())
    assert high.total_profit / high.visits_by_period.sum() > low.total_profit / low.visits_by_period.sum()
    none = run(HouseRules(comp_rate=0.0), small_cfg())
    generous = run(HouseRules(comp_rate=0.5), small_cfg())
    assert generous.comps_by_period.sum() > none.comps_by_period.sum() == 0


# ---------------------------------------------------------------- behaviour


def population_of(archetype: Archetype, n: int = 4) -> tuple[Population, HouseRules]:
    rules = HouseRules(game="roulette_us", min_bet=1.0, max_bet=10_000.0)
    pop = Population.create(n, np.random.default_rng(0), rules.moments(), rules, mix={archetype: 1.0})
    pop.base_bet[:] = 10.0
    pop.bankroll[:] = 5_000.0
    return pop.take(np.arange(n), psychology_table()), rules


def after_sessions(archetype: Archetype, outcomes: list[float]) -> tuple[Population, HouseRules, np.ndarray]:
    pop, rules = population_of(archetype)
    for pnl in outcomes:
        update_history(pop, np.full(pop.n, pnl))
    return pop, rules, bet_size(pop, rules.moments(), rules)


def test_prospect_theory_value_function():
    assert prospect_value(np.array([1.0]), 2.25)[0] == pytest.approx(1.0)
    assert prospect_value(np.array([-1.0]), 2.25)[0] == pytest.approx(-2.25)
    gains = prospect_value(np.array([1.0, 2.0]), 2.25)
    assert gains[1] < 2 * gains[0]  # diminishing sensitivity
    assert -prospect_value(np.array([-0.5]), 2.25)[0] > prospect_value(np.array([0.5]), 2.25)[0]


def test_loss_averse_bets_less_after_losses_than_house_money_does():
    _, _, loss_averse = after_sessions(Archetype.LOSS_AVERSE, [-100.0, -100.0])
    _, _, house_money = after_sessions(Archetype.HOUSE_MONEY, [-100.0, -100.0])
    _, _, steady = after_sessions(Archetype.STEADY, [-100.0, -100.0])
    assert loss_averse[0] < house_money[0]
    assert loss_averse[0] < steady[0] == pytest.approx(10.0)
    assert house_money[0] == pytest.approx(10.0)


def test_house_money_bets_more_after_a_gain():
    _, _, after_win = after_sessions(Archetype.HOUSE_MONEY, [300.0])
    _, _, after_loss = after_sessions(Archetype.HOUSE_MONEY, [-300.0])
    assert after_win[0] > 10.0 and after_loss[0] == pytest.approx(10.0)


def test_chaser_doubles_after_each_loss_and_resets_on_a_win():
    _, _, two_losses = after_sessions(Archetype.CHASER, [-50.0, -50.0])
    _, _, then_win = after_sessions(Archetype.CHASER, [-50.0, -50.0, 50.0])
    assert two_losses[0] == pytest.approx(40.0)
    assert then_win[0] == pytest.approx(10.0)


def test_hot_hand_and_fallacy_react_to_streaks_in_opposite_directions():
    _, _, hot = after_sessions(Archetype.HOT_HAND, [50.0, 50.0])
    _, _, fallacy_wins = after_sessions(Archetype.GAMBLERS_FALLACY, [50.0, 50.0])
    _, _, fallacy_losses = after_sessions(Archetype.GAMBLERS_FALLACY, [-50.0, -50.0])
    assert hot[0] > 10.0
    assert fallacy_wins[0] < 10.0 < fallacy_losses[0]


def test_break_even_bet_scales_with_remembered_deficit():
    pop, rules, small = after_sessions(Archetype.BREAK_EVEN, [-200.0])
    _, _, large = after_sessions(Archetype.BREAK_EVEN, [-200.0, -2000.0])
    assert pop.deficit[0] == pytest.approx(200.0)
    assert large[0] > small[0] >= 10.0
    _, _, recovered = after_sessions(Archetype.BREAK_EVEN, [-200.0, 250.0])
    assert recovered[0] == pytest.approx(10.0)


def test_bets_never_exceed_bankroll_or_table_limits():
    pop, rules, bets = after_sessions(Archetype.CHASER, [-1.0] * 10)
    assert (bets <= np.minimum(rules.max_bet, pop.bankroll)).all()
    pop.bankroll[:] = 3.0
    assert (bet_size(pop, rules.moments(), rules) == 3.0).all()


def test_archetype_ruin_ordering_in_a_real_run():
    res = run(HouseRules(), SimConfig(n_gamblers=16384, n_periods=120, seed=21))
    busts = res.busts_by_arch / res.n_by_arch
    assert busts[Archetype.CHASER] > busts[Archetype.STEADY]
    assert busts[Archetype.LOSS_AVERSE] <= busts[Archetype.STEADY]


# ---------------------------------------------------------------- optimiser


def test_gaussian_process_interpolates_a_smooth_function():
    rng = np.random.default_rng(0)
    x = rng.random((40, 2))
    y = np.sin(3 * x[:, 0]) + x[:, 1] ** 2
    gp = GaussianProcess(x, y, rng)
    xs = rng.random((200, 2))
    mean, std = gp.predict(xs)
    truth = np.sin(3 * xs[:, 0]) + xs[:, 1] ** 2
    assert np.sqrt(np.mean((mean - truth) ** 2)) < 0.1
    assert (std > 0).all()
    fit_mean, fit_std = gp.predict(x)
    assert np.abs(fit_mean - y).max() < 0.05
    assert fit_std.mean() < std.mean()


def test_pareto_front_keeps_only_non_dominated_points():
    profits = np.array([10.0, 20.0, 15.0, 5.0, 20.0])
    ruins = np.array([0.1, 0.3, 0.2, 0.05, 0.25])
    front = set(pareto_front(profits, ruins).tolist())
    assert front == {4, 2, 0, 3}


def test_slot_session_with_barriers_matches_spin_by_spin_simulation():
    """Skewed slot pay table, tight target and limit: exact paths versus the segment engine."""
    rules = HouseRules(game="slots", jackpot_share=0.0)
    prizes, probs = slots_pay_table(rules)
    rng = np.random.default_rng(12)
    bet, bankroll, target, limit = 10.0, 1000.0, 300.0, 300.0
    n_paths, rounds = 20_000, rules.moments().rounds
    per_spin = rng.choice(prizes, size=(n_paths, rounds), p=probs) - 1.0
    path = np.cumsum(per_spin * bet, axis=1)
    first_up = np.where((path >= target).any(axis=1), (path >= target).argmax(axis=1), rounds)
    first_down = np.where((path <= -limit).any(axis=1), (path <= -limit).argmax(axis=1), rounds)
    exact_p_up = np.mean((first_up < first_down) & (first_up < rounds))
    exact_p_down = np.mean((first_down < first_up) & (first_down < rounds))
    n = RNG_BLOCK
    sim = Simulation(rules, SimConfig(n_gamblers=n, n_periods=1, seed=5))
    out = sim._play_session(
        np.full(n, bet), np.full(n, bankroll), np.full(n, target), np.full(n, limit), np.ones(n, dtype=bool), np.arange(n)
    )
    for exact, got in ((exact_p_up, out.hit_target.mean()), (exact_p_down, out.hit_limit.mean())):
        se = np.sqrt(exact * (1 - exact) * (1 / n_paths + 1 / n))
        assert abs(got - exact) < 4 * se + 0.01
