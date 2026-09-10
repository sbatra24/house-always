#!/usr/bin/env bash
# Reproduce every file in outputs/ from scratch. Total time on a free
# 2-core machine is roughly 35 minutes; the full run is the long pole.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0

echo "== tests"
python3 -m pytest tests -q

echo "== optimizer (Bayesian optimisation over house rules)"
python3 optimize.py --init 24 --rounds 12 --workers 2

echo "== animated landscape (house edge x comp rate, per decade)"
python3 animate.py --workers 2

echo "== plain edge sweep, 100 years"
python3 sweep.py --workers 2

echo "== one life, per archetype"
python3 life.py --archetype CHASER --months 120 --seed 3 > /dev/null
python3 life.py --archetype LOSS_AVERSE --months 120 --seed 3 > /dev/null
python3 life.py --archetype BREAK_EVEN --months 120 --seed 3 > /dev/null

echo "== full run: 1,000,000 gamblers x 10,000 months, 12 casinos"
python3 run_full.py --workers 2

echo "done; see outputs/"
