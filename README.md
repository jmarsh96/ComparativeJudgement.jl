# ComparativeJudgement

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jmarsh96.github.io/ComparativeJudgement.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/)
[![Build Status](https://github.com/jmarsh96/ComparativeJudgement.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jmarsh96/ComparativeJudgement.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jmarsh96/ComparativeJudgement.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jmarsh96/ComparativeJudgement.jl)

A Julia package for fitting comparative judgement models to pairwise
comparison data: items are placed on a latent strength scale from the
outcomes of head-to-head comparisons, rather than from absolute scores.

The package currently implements the Bradley–Terry model with three
workflows:

- **Maximum likelihood** (`MLE`) — fast point estimates of the latent
  strengths, good for ranking items.
- **Bayesian** (`Bayesian`) — a Pólya-Gamma augmented Gibbs sampler giving
  full posterior distributions, so every strength and win probability comes
  with uncertainty.
- **Anchored** (`BradleyTerryAnchored`) — a joint Bayesian model in which
  known measurements `y = a + b·λ + ε` for a subset of items calibrate the
  latent scale, so measurements can be predicted (with credible intervals)
  for items that were never measured.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/jmarsh96/ComparativeJudgement.jl")
```

## Quick start

```julia
using ComparativeJudgement

wins = [0 12 15;
        8  0 13;
        5  7  0]                      # wins[i, j]: times item i beat item j
data = PairwiseData(wins, ["A", "B", "C"])

fitted = fit(BradleyTerry(), MLE(), data)
strengths(fitted)                     # estimated latent strengths
probability(fitted, "A", "C")         # P(A beats C)
```

## Documentation

The [documentation](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/)
includes a [worked tutorial](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/bradley_terry/)
fitting all three models to simulated data — including an anchored example
that measures half the items and predicts the measurements of the other
half — and a full [API reference](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/api/).
