# house-always

I run a casino in this repo. It is a simulated one, with a million simulated gamblers, open for ten thousand simulated years, and at the end I ask the computer which house rules take the most money from those gamblers while sending them broke as slowly as possible. The whole thing is NumPy. There is no loop over gamblers anywhere.

It grew out of [nthchance.com](https://nthchance.com), a browser game I built where you play the house instead of the player, and I kept wondering what the actual answer was: farm the customers or slaughter them?

## What is in here

```
gamblers.py    population model, eight behavioural archetypes as rows in arrays
casino.py      games, sessions, comps, and the HouseRules dataclass
engine.py      chunked, seeded, parallel simulation loop with full money accounting
optimize.py    Bayesian optimisation (own Gaussian process) over house rules, Pareto front
animate.py     animated heatmap of profit against edge and comp rate, one frame per decade
life.py        one gambler, month by month, as a diary and a chart
sweep.py       plain sweep over house edge, 100 years, everything else fixed
run_full.py    the 1,000,000 x 10,000 run
tests/         pytest suite (25 tests)
outputs/       every number and picture quoted below, produced by run.sh
run.sh         reproduces everything
```

```
pip install -r requirements.txt
python -m pytest tests -q
python life.py --archetype CHASER --months 120
python optimize.py
python animate.py
python sweep.py
python run_full.py
./run.sh
```

## The headline numbers

Everything below comes from files in `outputs/`. The optimiser ran 72 rule sets, each a 30-year casino with 16,384 gamblers. The full run is twelve independent casinos of 83,333 or 83,334 gamblers each, every one open for 10,000 months (833 years). Twelve casinos times 833 years is 10,000 casino-years; a million gamblers times 10,000 months is 10^10 gambler-months, or 833 million gambler-years. Those are the definitions, so the "10,000 years" claim is honest and so is the "million gamblers" claim, and the two are different things.

**The profit-maximising edge is the biggest edge I allowed.** I let the optimiser pick any house edge from 1% to 30% and it went to 30% every time, constrained or not. Real slots sit between 2% and 15%, so I widened the range on purpose to find where more edge stops paying. Inside this model it never does. The gamblers respond to what happened to them last month, and by the month after that they have mostly forgotten. That is the boring result and I am reporting it as boring.

**The interesting lever is the maximum bet.** The unconstrained optimum caps bets at $10.99 per spin, when the search range went up to $200. A casino that lets a chaser put $100 on a spin makes less money over 30 years than one that stops them at $11, because the chaser goes broke, walks away from the casino 15% of the time they do, and one in ten of those walks is permanent. The Pareto front in `outputs/pareto.png` is almost entirely a curve in maximum bet: to lower the ruin rate from 6.8% to 2% of gamblers per year, the house first lowers the bet cap to the $5 floor, and only then starts giving up edge.

| optimum | house edge | max bet | comps | jackpot | volatility | profit per gambler per year | gamblers bankrupted per year |
|---|---|---|---|---|---|---|---|
| unconstrained | 30% | $10.99 | 0% | 100x | 2.0 | $1,462 | 6.8% |
| ruin at most 5% per year | 30% | $6.53 | 0.7% | 242x | 2.0 | $1,361 | 5.0% |
| ruin at most 2% per year | 19% | $5.00 | 0% | 100x | 2.0 | $941 | 1.95% |

**Comps never pay for themselves.** Free play priced at a share of theoretical loss makes gamblers a little more likely to come back in this model, and the money it costs is always larger than the money it brings back. The optimiser drove the comp rate to zero in every run. I want to be clear that the comp sensitivity in `gamblers.py` (0.3 to 0.4 logit units of return probability per monthly budget received in comps) is my guess and nothing in the literature I read pins it down. Change it and this result changes.

**The surprise: a greedier casino can report a lower bankruptcy rate.** `outputs/edge_sweep.csv` holds a plain sweep over house edge with everything else fixed (max bet $30, no comps, volatility 2, 16,384 gamblers, 100 years). At a 30% edge the casino bankrupts 13.0% of its gamblers per year in the first decade and 0.65% per year in the tenth. At a 10% edge the figures are 9.8% and 2.0%. Averaged over the century, the 30% casino "only" bankrupts 3.8% of gamblers per year against 4.6% for the 10% casino, because it burned through the chasers and the break-even bettors early and the survivors are the careful archetypes. A ruin rate averaged over a long horizon rewards slaughter, which is the opposite of what I built the constraint to do. Profit per gambler per year still rises with edge at every horizon in that file: $157 at 2%, $716 at 10%, $1,260 at 30%, and the first decade pays 14% more than the tenth at a 2% edge and about a third more at 10% and above.

**Low volatility wins.** Every optimum sits at the bottom of the volatility range. Big, rare prizes let gamblers hit their win targets and leave, and they also send the escalators broke faster. The winning slot pays 1.5x on 28% of spins, 15x on 1.7% of spins, and a 242x jackpot once in 12,000 spins, at a 30% edge; it takes more money over 30 years than any flashier machine I tried.

### The full run

I ran the ruin-constrained optimum (30% edge, $6.53 maximum bet, 0.7% comps, 242x jackpot, volatility 2) for the full million gamblers and 10,000 months. Numbers from `outputs/full_run.json`:

One million gamblers times 10,000 months is 10,000,000,000 gambler-months. They ran in 656 seconds of wall time on two Xeon cores at 2.8 GHz with 7 GB of memory, 1,294 CPU-seconds across both workers. That is 15.2 million gambler-months per second, 129 nanoseconds of CPU per gambler-month, or 1,524 complete 833-year gambling lives per second.

The house kept $965.3 billion on $3.36 trillion of handle, a 28.75% hold, and paid $7.4 billion in comps. That is $1,158 per gambler per year, against a median gambling budget of $3,000 per year. Money is conserved to $0.15 across the whole run, a relative error of 5e-14.

There were 7.6 million ruin events. 23.9% of gamblers went broke at least once in 833 years, and 79.3% were still gambling at the end. The yearly bankruptcy rate averaged 0.39% over 833 years. The same rules scored 5.0% over 30 years in the optimiser. Same casino, same gamblers; the fragile ones simply do not last.

Profit per gambler per year fell from $1,415 in the first decade to $1,213 in the tenth and $1,117 in the eighty-third, and the share of gamblers still active went from 90% to 83% to 79%. Most of the decline happens in the first century and the curve keeps sloping down for the remaining seven.

Per archetype, from the same run:

| archetype | house profit per gambler, first 50 years | ever went broke | median months to first ruin | ruin events per gambler in 833 years | visits per year |
|---|---|---|---|---|---|
| GAMBLERS_FALLACY | $89,789 | 0.0% | never | 0.00 | 3.71 |
| CHASER | $85,309 | 88.3% | 10 | 52.6 | 1.53 |
| NEAR_MISS | $82,602 | 0.0% | never | 0.00 | 4.61 |
| HOT_HAND | $71,973 | 0.0% | never | 0.00 | 4.11 |
| HOUSE_MONEY | $68,726 | 20.3% | never (fewer than half) | 0.75 | 4.00 |
| BREAK_EVEN | $61,650 | 82.9% | 33 | 7.6 | 1.26 |
| STEADY | $54,484 | 0.0% | never | 0.00 | 3.32 |
| LOSS_AVERSE | $17,516 | 0.0% | never | 0.00 | 3.11 |

**The archetype that loses the most money over a lifetime is the gambler's fallacy.** I expected the chaser. The chaser goes broke ten months in, on the median, and 52 times over the horizon, and every time they go broke they visit less for a while and sometimes never again, so the house sees them 1.5 times a year. The fallacy gambler never goes broke at all. They raise their bet after every loss, keep a 60% loss limit that stops them short of ruin, and show up 3.7 times a year for 833 years. Slow and steady extraction beats the spectacular blow-up by a few thousand dollars per lifetime, and by a factor of two over the full horizon ($1.50 million against $0.72 million). The loss-averse prospect-theory gambler is the worst customer the house has, at a fifth of the fallacy gambler's losses, because they shrink their bet after every losing session and leave the moment they are 20% down.

The median months-to-ruin figure is "never" for six of eight archetypes because fewer than half of them ever go broke; their win targets and loss limits keep them away from zero, and a monthly budget refills what the tables take. Only the two archetypes without a loss limit, the chaser and the break-even bettor, go broke as a matter of routine.

### The animation

`outputs/landscape.gif` is twelve frames, one per decade, of house profit per gambler per year across a 10 x 8 grid of house edge (1% to 28%) and comp rate (0% to 60%). Each cell is its own casino with 8,192 gamblers and 120 years. The best cell earns $1,041 per gambler per year in the first decade and $764 in the twelfth, a 27% decline, as the gamblers who chase and the gamblers who bet to break even go broke and a share of them quit for good. The share of gamblers still active across the grid falls from 81% to 68%. The star never moves: the most profitable casino is the highest edge with no comps in every decade.

### The diaries

`outputs/life_CHASER.txt` follows one Martingale-style chaser for ten years at an 8% slot with 20% comps: 23 visits, broke 14 times, the house netted $2,437 from a $257 monthly budget. In month 57 they come back from a lapse, put $100 on a spin (the doubling multiplier survived the lapse), hit a big prize and leave with $8,137. Most of it goes home under the reserve cap, and the rest is chased into the ground by month 63. `outputs/life_LOSS_AVERSE.txt` is the same seed for a prospect-theory gambler: 35 visits, never broke, the house netted $2,007. `outputs/life_BREAK_EVEN.txt`: 23 visits, broke 13 times, $4,909 to the house, sworn off gambling by month 120. Each comes with a PNG of the bankroll path.

## How it works

### Gamblers

Every gambler is one row across a set of arrays: budget, bankroll, base bet, an archetype id, and seven state variables (bankroll, bet multiplier, streak, remembered deficit, last result, return probability, lapsed flag). Each month the gambler receives a gambling budget (lognormal, median $250), keeps at most six budgets in reserve, and visits the casino with their current return probability. The eight archetypes are rows of one parameter table, so the same vectorised formulas produce all of them:

| archetype | what it does | where the idea comes from |
|---|---|---|
| STEADY | fixed bet, 30% win target, 30% loss limit | control |
| LOSS_AVERSE | shrinks the bet 30% after every losing session, tight limits, values outcomes with alpha = beta = 0.88 and lambda = 2.25 | Kahneman & Tversky 1979, Tversky & Kahneman 1992 |
| GAMBLERS_FALLACY | bets more after losses (a win is due), less after wins | Tversky & Kahneman 1971, Croson & Sundali 2005 |
| HOT_HAND | bets more after wins, no win target | Gilovich, Vallone & Tversky 1985 |
| CHASER | doubles after every losing session, no loss limit, keeps coming back | Lesieur 1984, the Martingale |
| NEAR_MISS | return probability driven by how often the machine pays anything rather than by the net result | Reid 1986, Dixon et al. 2010 on losses disguised as wins |
| BREAK_EVEN | bets enough that one lucky session erases the remembered deficit, leaves as soon as slightly ahead | Thaler & Johnson 1990 |
| HOUSE_MONEY | bets and risks more after a gain, coded as the house's money | Thaler & Johnson 1990 |

Every archetype values last month's result with the prospect theory value function (alpha = beta = 0.88, and lambda ranging from 0.8 for the chaser to 2.25 for the loss-averse gambler) and that value moves next month's return probability on the logit scale. A gambler who goes broke lapses with an archetype-specific probability; some lapses are permanent, the rest end with a monthly relapse hazard.

### Sessions

A session is a few hundred rounds at one bet. Simulating each spin for 10^10 gambler-months was out of the question (a binomial draw alone costs 170 ns per gambler on this machine), so a session is resolved in closed form. Every game is split into a diffusion (the frequent outcomes: red or black, a blackjack hand, a slot spin that pays nothing or 1.5x) and jumps (rare big prizes, a Poisson process). The diffusion's closing P&L over a segment is one normal draw. Whether the path touched the gambler's win target or loss limit on the way is the Brownian bridge barrier probability, exp(-2 L (L - X) / variance), which does not depend on drift. When both barriers could have been touched, which came first is decided by the gambler's ruin probability for a drifting diffusion, so a gambler betting everything on one spin gets the textbook answer. Segments end at the next big prize (an exponential draw), the prize lands, and the next segment starts. Jackpots are a second Poisson stream over the rounds actually played. Comps are charged on rounds actually played, using Wald's identity for the expected time to a barrier.

The tests compare the session means and the barrier logic against spin-by-spin simulation. For roulette the approximation is within Monte Carlo error in every regime I checked; for slots the same holds as long as one bet is a small fraction of the bankroll.

### Engine

Gamblers are processed in chunks of 32,768 so the working set fits in cache. Each block of 32,768 gamblers has its own PCG64 generator spawned from one seed, so results do not depend on chunk size or on how many worker processes run (both are tested). Each month the gamblers who show up are gathered into a compact sub-population, the session and psychology code runs on that subset with no masks, and the state is scattered back. Every dollar is tracked: budgets in, reserve overflow out, table results, comps back. The residual of that identity is `Results.conservation_error` and it is zero to floating-point precision on every run.

### Optimiser

`optimize.py` is Bayesian optimisation with a Gaussian process written in NumPy: Matern 5/2 kernel with a lengthscale per dimension, hyperparameters fit by marginal likelihood, and the constrained expected improvement of Gardner et al. (2014), which multiplies expected improvement in profit by the GP's probability that the ruin constraint holds. One Latin hypercube of 24 rule sets seeds the runs; each of 12 rounds then proposes one rule set per ruin cap (none, 2%, 5%, 10%) and evaluates them two at a time. Every evaluation feeds the Pareto front.

## Limitations

These are stylised agents. Nobody in this simulation is a person; each one is a dozen parameters I chose by reading papers and making judgement calls, and a real gambler is not a fixed archetype for 833 years. The parameters I am least sure of are the ones that decide the results: how fast a bad month fades from a gambler's return probability (20% per month toward baseline), how much comps matter (a guess), and what share of people who go broke never come back (10% to 60% by archetype). The gamblers never learn what the edge is; they only feel their own outcomes. The session model is an approximation and it overstates the house take when a single bet is more than about a tenth of the bankroll, which mostly affects chasers in the month they go broke. Slot volatility is one number, and the pay table behind it has three prizes plus a jackpot. Blackjack is a house edge with a standard deviation, built from published rule effects. Money outside the gambling budget does not exist.

What the code does do is run the same stylised world at a scale where the numbers are stable, conserve every dollar, reproduce bit for bit from a seed, and put the whole pipeline in one script. Read the results as the consequences of the assumptions in `gamblers.py`, which are all in one table you can edit.

## Licence

MIT, see `LICENSE`.
