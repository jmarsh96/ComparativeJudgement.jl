# ComparativeJudgement

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jmarsh96.github.io/ComparativeJudgement.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/)
[![Build Status](https://github.com/jmarsh96/ComparativeJudgement.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jmarsh96/ComparativeJudgement.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jmarsh96/ComparativeJudgement.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jmarsh96/ComparativeJudgement.jl)

A Julia package for fitting comparative judgement models to pairwise
comparison data: items are placed on a latent strength scale from the
outcomes of head-to-head comparisons, rather than from absolute scores.

The package currently implements the Bradley–Terry model with four
workflows:

- **Maximum likelihood** (`MLE`) — fast point estimates of the latent
  strengths, good for ranking items.
- **Bayesian** (`Bayesian`) — a Pólya-Gamma augmented Gibbs sampler giving
  full posterior distributions, so every strength and win probability comes
  with uncertainty.
- **Anchored** (`BradleyTerryAnchored`) — a joint Bayesian model in which
  known measurements `y = a + b·λ + ε` for a subset of items calibrate the
  latent scale, so measurements can be predicted (with credible intervals)
  for items that were never measured. Anchors may apply to a single item or to
  the *average* over a group of items (e.g. the mean mark of a batch of scripts).
- **Covariates** (`BradleyTerryCovariates`) — strengths modelled as a linear
  function of item covariates, `λ_i = zᵢᵀβ`. Fit by MLE or Bayesian inference,
  with model selection via stepwise AIC/BIC (`StepwiseMLE`) or shrinkage /
  spike-and-slab priors (`HorseshoePrior`, `SpikeSlabPrior`).

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

With item covariates, model the strengths as `λᵢ = zᵢᵀβ`:

```julia
Z = [0.0 1.0;                         # one row of covariates per item
     1.0 0.0;
     2.0 1.0]
cd = CovariateData(data, Z, [:size, :age])

cfit = fit(BradleyTerryCovariates(), MLE(), cd)                 # ML coefficients
coefficients(cfit)                                              # point estimates β
coefficient_intervals(cfit; level=0.95)                        # confidence intervals

fit(BradleyTerryCovariates(), StepwiseMLE(criterion=:BIC), cd)  # stepwise selection
fit(BradleyTerryCovariates(), Bayesian(), cd, HorseshoePrior()) # shrinkage
fit(BradleyTerryCovariates(), Bayesian(), cd, SpikeSlabPrior()) # selection + PIPs
```

## Documentation

The [documentation](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/)
includes a [worked tutorial](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/bradley_terry/)
fitting the MLE and Bayesian models to simulated data, an
[anchored-model tutorial](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/anchored_bt/)
that measures half the items and predicts the measurements of the other half (and
covers group-averaged anchors), a
[covariate-model tutorial](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/covariate_bt/)
covering MLE, stepwise selection and shrinkage / spike-and-slab priors, and a
full [API reference](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/api/).
