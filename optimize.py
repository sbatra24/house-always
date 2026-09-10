"""Search HouseRules for the most profitable casino that bankrupts people slowest.

Objective: house profit per gambler per year, over a 30-year horizon with
16,384 gamblers, subject to the fraction of gamblers who go broke in a
typical year staying below a cap. Five levers: house edge, maximum bet,
comp rate, jackpot size and slot volatility.

Method: Bayesian optimisation with a Gaussian process written here in
NumPy (Matern 5/2 kernel with per-dimension lengthscales, hyperparameters
fit by marginal likelihood) and the constrained expected improvement of
Gardner et al. (2014): EI on profit times the GP probability that the ruin
constraint holds. One shared Latin hypercube design seeds separate BO runs
for each ruin cap; every evaluation feeds the Pareto front.

Run `python optimize.py` to reproduce outputs/pareto.csv, outputs/pareto.png,
outputs/optimizer_evals.csv and outputs/optimum_rules.json.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
import json
import time
from dataclasses import dataclass, replace
from multiprocessing import get_context

import numpy as np

from casino import HouseRules
from engine import SimConfig, run

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


@dataclass(frozen=True)
class Dimension:
    name: str
    low: float
    high: float
    log: bool = False

    def to_unit(self, x: float) -> float:
        if self.log:
            return (np.log(x) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))
        return (x - self.low) / (self.high - self.low)

    def from_unit(self, u: float) -> float:
        if self.log:
            return float(np.exp(np.log(self.low) + u * (np.log(self.high) - np.log(self.low))))
        return float(self.low + u * (self.high - self.low))


SPACE = (
    Dimension("house_edge", 0.01, 0.30),
    Dimension("max_bet", 5.0, 200.0, log=True),
    Dimension("comp_rate", 0.0, 0.6),
    Dimension("jackpot_size", 100.0, 20000.0, log=True),
    Dimension("volatility", 2.0, 12.0),
)

RUIN_CAPS = (0.02, 0.05, 0.10)
EVAL_GAMBLERS = 16_384
EVAL_MONTHS = 360
EVAL_SEED = 2026


def rules_from_unit(u: np.ndarray, base: HouseRules) -> HouseRules:
    values = {d.name: d.from_unit(float(v)) for d, v in zip(SPACE, u)}
    return replace(base, **values)


@dataclass
class Evaluation:
    unit: np.ndarray
    rules: HouseRules
    profit: float
    ruin: float
    seconds: float


def evaluate(args: tuple[np.ndarray, HouseRules]) -> Evaluation:
    u, base = args
    rules = rules_from_unit(u, base)
    cfg = SimConfig(n_gamblers=EVAL_GAMBLERS, n_periods=EVAL_MONTHS, seed=EVAL_SEED)
    res = run(rules, cfg)
    return Evaluation(u.copy(), rules, res.profit_per_gambler_year, res.ruin_rate, res.wall_time)


def evaluate_many(units: list[np.ndarray], base: HouseRules, workers: int) -> list[Evaluation]:
    tasks = [(u, base) for u in units]
    if workers <= 1 or len(tasks) == 1:
        return [evaluate(t) for t in tasks]
    with get_context("fork").Pool(min(workers, len(tasks))) as pool:
        return pool.map(evaluate, tasks)


def latin_hypercube(n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    cells = (np.arange(n)[:, None] + rng.random((n, dims))) / n
    for d in range(dims):
        cells[:, d] = cells[rng.permutation(n), d]
    return cells


class GaussianProcess:
    """Zero-mean GP with a Matern 5/2 ARD kernel and Gaussian noise.

    Targets are standardised internally. Hyperparameters (log lengthscales,
    log signal variance, log noise variance) are fit by maximising the log
    marginal likelihood with random restarts followed by coordinate-wise
    hill climbing, which is plenty for the few dozen points BO sees.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> None:
        self.x = x
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) or 1.0
        self.y = (y - self.y_mean) / self.y_std
        self.rng = rng
        self.theta = self._fit()
        self._factor(self.theta)

    @staticmethod
    def kernel(a: np.ndarray, b: np.ndarray, theta: np.ndarray) -> np.ndarray:
        lengthscales = np.exp(theta[:-2])
        signal = np.exp(theta[-2])
        diff = (a[:, None, :] - b[None, :, :]) / lengthscales
        r = np.sqrt(np.sum(diff * diff, axis=-1))
        s5 = np.sqrt(5.0) * r
        return signal * (1.0 + s5 + s5 * s5 / 3.0) * np.exp(-s5)

    def _log_marginal(self, theta: np.ndarray) -> float:
        k = self.kernel(self.x, self.x, theta) + np.exp(theta[-1]) * np.eye(len(self.x))
        try:
            chol = np.linalg.cholesky(k)
        except np.linalg.LinAlgError:
            return -np.inf
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, self.y))
        return float(-0.5 * self.y @ alpha - np.log(np.diag(chol)).sum() - 0.5 * len(self.x) * np.log(2 * np.pi))

    def _fit(self) -> np.ndarray:
        dims = self.x.shape[1]
        lows = np.array([np.log(0.05)] * dims + [np.log(0.1), np.log(1e-4)])
        highs = np.array([np.log(3.0)] * dims + [np.log(5.0), np.log(0.5)])
        candidates = lows + self.rng.random((60, dims + 2)) * (highs - lows)
        scores = np.array([self._log_marginal(c) for c in candidates])
        best = candidates[int(np.argmax(scores))]
        best_score = scores.max()
        step = 0.5
        for _ in range(6):
            improved = False
            for d in range(dims + 2):
                for sign in (-1.0, 1.0):
                    trial = best.copy()
                    trial[d] = np.clip(trial[d] + sign * step, lows[d], highs[d])
                    score = self._log_marginal(trial)
                    if score > best_score:
                        best, best_score, improved = trial, score, True
            if not improved:
                step *= 0.5
        return best

    def _factor(self, theta: np.ndarray) -> None:
        k = self.kernel(self.x, self.x, theta) + np.exp(theta[-1]) * np.eye(len(self.x))
        self.chol = np.linalg.cholesky(k)
        self.alpha = np.linalg.solve(self.chol.T, np.linalg.solve(self.chol, self.y))

    def predict(self, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation in original units."""
        ks = self.kernel(xs, self.x, self.theta)
        mean = ks @ self.alpha
        v = np.linalg.solve(self.chol, ks.T)
        var = np.exp(self.theta[-2]) - np.sum(v * v, axis=0)
        std = np.sqrt(np.maximum(var, 1e-12))
        return mean * self.y_std + self.y_mean, std * self.y_std


def normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26, accurate to 1.5e-7, enough for acquisition."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-x * x))


def normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def expected_improvement(mean: np.ndarray, std: np.ndarray, best: float) -> np.ndarray:
    z = (mean - best) / std
    return (mean - best) * normal_cdf(z) + std * normal_pdf(z)


def constrained_ei(
    gp_profit: GaussianProcess, gp_ruin: GaussianProcess, xs: np.ndarray, cap: float | None, best: float
) -> np.ndarray:
    mean, std = gp_profit.predict(xs)
    ei = expected_improvement(mean, std, best)
    if cap is None:
        return ei
    r_mean, r_std = gp_ruin.predict(xs)
    return ei * normal_cdf((cap - r_mean) / r_std)


def propose(
    gp_profit: GaussianProcess, gp_ruin: GaussianProcess, cap: float | None, evals: list[Evaluation], rng: np.random.Generator
) -> np.ndarray:
    """Maximise constrained EI over random candidates, then refine locally."""
    x = np.array([e.unit for e in evals])
    mean, _ = gp_profit.predict(x)
    if cap is None:
        feasible = np.ones(len(evals), dtype=bool)
    else:
        r_mean, r_std = gp_ruin.predict(x)
        feasible = normal_cdf((cap - r_mean) / r_std) > 0.5
    best = float(mean[feasible].max()) if feasible.any() else float(mean.min())
    candidates = rng.random((4000, len(SPACE)))
    scores = constrained_ei(gp_profit, gp_ruin, candidates, cap, best)
    top = candidates[np.argsort(scores)[-8:]]
    for _ in range(40):
        trial = np.clip(top + 0.05 * rng.standard_normal(top.shape), 0.0, 1.0)
        trial_scores = constrained_ei(gp_profit, gp_ruin, trial, cap, best)
        old_scores = constrained_ei(gp_profit, gp_ruin, top, cap, best)
        better = trial_scores > old_scores
        top[better] = trial[better]
    final = constrained_ei(gp_profit, gp_ruin, top, cap, best)
    return top[int(np.argmax(final))]


def pareto_front(profits: np.ndarray, ruins: np.ndarray) -> np.ndarray:
    """Indices of points not dominated in (higher profit, lower ruin)."""
    order = np.lexsort((ruins, -profits))
    keep = []
    best_ruin = np.inf
    for i in order:
        if ruins[i] < best_ruin:
            keep.append(i)
            best_ruin = ruins[i]
    return np.array(keep)


def optimise(base: HouseRules, n_init: int, n_rounds: int, workers: int, seed: int) -> tuple[list[Evaluation], dict]:
    """Shared initial design, then one proposal per ruin cap in every round."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    evals = evaluate_many(list(latin_hypercube(n_init, len(SPACE), rng)), base, workers)
    for i, e in enumerate(evals):
        print(f"init {i + 1:2d}/{n_init}  profit {e.profit:8.1f}  ruin {e.ruin:.4f}  ({e.seconds:.1f}s)", flush=True)
    caps = (None, *RUIN_CAPS)
    for rnd in range(n_rounds):
        x = np.array([e.unit for e in evals])
        gp_profit = GaussianProcess(x, np.array([e.profit for e in evals]), rng)
        gp_ruin = GaussianProcess(x, np.array([e.ruin for e in evals]), rng)
        proposals = [propose(gp_profit, gp_ruin, cap, evals, rng) for cap in caps]
        new = evaluate_many(proposals, base, workers)
        for cap, e in zip(caps, new):
            label = "unconstrained" if cap is None else f"ruin<={cap:.2f}"
            print(f"round {rnd + 1:2d} {label:14s} profit {e.profit:8.1f}  ruin {e.ruin:.4f}  edge {e.rules.house_edge:.3f}  "
                  f"max_bet {e.rules.max_bet:6.1f}  comp {e.rules.comp_rate:.2f}  jackpot {e.rules.jackpot_size:7.0f}  "
                  f"vol {e.rules.volatility:5.2f}  ({e.seconds:.1f}s)", flush=True)
        evals.extend(new)
    winners: dict = {}
    for cap in caps:
        label = "unconstrained" if cap is None else f"ruin<={cap:.2f}"
        feasible = [e for e in evals if cap is None or e.ruin <= cap]
        winners[label] = max(feasible, key=lambda e: e.profit) if feasible else None
    print(f"optimisation wall time {time.perf_counter() - t0:.0f}s over {len(evals)} evaluations")
    return evals, winners


def rules_dict(rules: HouseRules) -> dict:
    return {d.name: getattr(rules, d.name) for d in SPACE}


def save_outputs(evals: list[Evaluation], winners: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "optimizer_evals.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([d.name for d in SPACE] + ["profit_per_gambler_year", "ruin_rate"])
        for e in evals:
            w.writerow([f"{getattr(e.rules, d.name):.6g}" for d in SPACE] + [f"{e.profit:.4f}", f"{e.ruin:.6f}"])
    profits = np.array([e.profit for e in evals])
    ruins = np.array([e.ruin for e in evals])
    front = pareto_front(profits, ruins)
    front = front[np.argsort(ruins[front])]
    with open(os.path.join(OUT_DIR, "pareto.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ruin_rate", "profit_per_gambler_year"] + [d.name for d in SPACE])
        for i in front:
            w.writerow([f"{ruins[i]:.6f}", f"{profits[i]:.4f}"] + [f"{getattr(evals[i].rules, d.name):.6g}" for d in SPACE])
    summary = {
        "evaluations": len(evals),
        "horizon_months": EVAL_MONTHS,
        "gamblers_per_evaluation": EVAL_GAMBLERS,
        "winners": {
            label: None if e is None else {"rules": rules_dict(e.rules), "profit_per_gambler_year": e.profit, "ruin_rate": e.ruin}
            for label, e in winners.items()
        },
    }
    with open(os.path.join(OUT_DIR, "optimum_rules.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_pareto(profits, ruins, front, winners)


def plot_pareto(profits: np.ndarray, ruins: np.ndarray, front: np.ndarray, winners: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=130)
    ax.scatter(ruins * 100, profits, s=18, color="#9aa5b1", label="every rule set tried")
    ax.plot(ruins[front] * 100, profits[front], "-o", color="#c0392b", ms=5, lw=1.6, label="Pareto front")
    labels: dict[tuple[float, float], list[str]] = {}
    for label, e in winners.items():
        if e is not None:
            labels.setdefault((e.ruin, e.profit), []).append(label)
    for (ruin, profit), names in labels.items():
        ax.annotate(
            " = ".join(names), (ruin * 100, profit), textcoords="offset points", xytext=(6, -12), fontsize=8, color="#2c3e50"
        )
    ax.set_xlabel("gamblers bankrupted per year (%)")
    ax.set_ylabel("house profit per gambler per year ($)")
    ax.set_title("Farm or slaughter: profit against ruin rate, 30-year horizon")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pareto.png"))
    plt.close(fig)


def load_saved() -> tuple[list[Evaluation], dict]:
    """Rebuild evaluations and winners from the CSV and JSON written earlier."""
    base = HouseRules(game="slots")
    evals: list[Evaluation] = []
    with open(os.path.join(OUT_DIR, "optimizer_evals.csv")) as f:
        for row in csv.DictReader(f):
            rules = replace(base, **{d.name: float(row[d.name]) for d in SPACE})
            unit = np.array([d.to_unit(getattr(rules, d.name)) for d in SPACE])
            evals.append(Evaluation(unit, rules, float(row["profit_per_gambler_year"]), float(row["ruin_rate"]), 0.0))
    with open(os.path.join(OUT_DIR, "optimum_rules.json")) as f:
        saved = json.load(f)["winners"]
    winners = {}
    for label, w in saved.items():
        if w is None:
            winners[label] = None
            continue
        winners[label] = min(evals, key=lambda e: abs(e.profit - w["profit_per_gambler_year"]) + abs(e.ruin - w["ruin_rate"]))
    return evals, winners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--init", type=int, default=24, help="Latin hypercube points")
    parser.add_argument("--rounds", type=int, default=12, help="BO rounds (one proposal per ruin cap each)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replot", action="store_true", help="redraw outputs from the saved CSV without simulating")
    args = parser.parse_args()
    if args.replot:
        evals, winners = load_saved()
    else:
        evals, winners = optimise(HouseRules(game="slots"), args.init, args.rounds, args.workers, args.seed)
    save_outputs(evals, winners)
    for label, e in winners.items():
        if e is not None:
            print(f"{label:14s} profit {e.profit:8.1f}/gambler/yr  ruin {e.ruin:.4f}  {rules_dict(e.rules)}")


if __name__ == "__main__":
    main()
