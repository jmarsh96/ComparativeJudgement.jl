# ComparativeJudgement

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jmarsh96.github.io/ComparativeJudgement.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/)
[![Build Status](https://github.com/jmarsh96/ComparativeJudgement.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jmarsh96/ComparativeJudgement.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jmarsh96/ComparativeJudgement.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jmarsh96/ComparativeJudgement.jl)

A Julia package for fitting comparative judgement models to pairwise
comparison data: items are placed on a latent strength scale from the
outcomes of head-to-head comparisons, rather than from absolute scores.

## Models

Two pairwise-comparison models are implemented, differing only in the link that
maps the strength difference to a win probability:

| Model | `P(i beats j)` | Constructor |
| --- | --- | --- |
| **Bradley–Terry** | `logistic(λᵢ − λⱼ)` (logit link) | `BradleyTerry()` |
| **Thurstone Case V** | `Φ(λᵢ − λⱼ)` (probit link) | `ThurstoneCaseV()` |

Each model comes in three variants:

- **Plain** — one free latent strength per item.
- **Anchored** (`BradleyTerryAnchored`, `ThurstoneCaseVAnchored`) — known
  measurements `y = a + b·λ + ε` for a subset of items calibrate the latent
  scale, so measurements can be predicted (with intervals) for items that were
  never measured. Anchors may apply to a single item or to the *average* over a
  group of items (e.g. the mean mark of a batch of scripts).
- **Covariates** (`BradleyTerryCovariates`, `ThurstoneCaseVCovariates`) —
  strengths modelled as a linear function of item covariates, `λᵢ = zᵢᵀβ`.

Bradley–Terry has two further structural extensions:

- **Rater heterogeneity** (`BradleyTerryRaterHeterogeneity`) — a mixture in which
  rater `r` follows Bradley–Terry with reliability `q_r` and otherwise guesses,
  `P(r judges i beats j) = q_r·logistic(λᵢ − λⱼ) + (1 − q_r)/2`, so unreliable
  assessors are down-weighted and the `q_r` flag them. Takes per-rater
  comparisons in a `RaterData`.
- **Intransitivity** (`BradleyTerryIntransitive`) — a skew-symmetric per-pair
  term `γᵢⱼ = −γⱼᵢ` added to the predictor,
  `logit P(i beats j) = (λᵢ − λⱼ) + γᵢⱼ`, measuring how far the judgements depart
  from a single transitive scale.

## Inference methods

Every model and variant can be fitted by either method:

- **Maximum likelihood** (`MLE`) — fast point estimates of the latent
  strengths, good for ranking items.
- **Bayesian** (`Bayesian`) — Gibbs sampling giving full posterior
  distributions, so every strength and win probability comes with uncertainty.
  Bradley–Terry uses Pólya-Gamma augmentation; Thurstone uses Albert–Chib
  truncated-normal augmentation.

For covariate models, the covariates can additionally be selected by greedy
AIC/BIC stepwise search (`StepwiseMLE`) or by Bayesian shrinkage / variable
selection priors (`HorseshoePrior`, `SpikeSlabPrior`, with posterior inclusion
probabilities), alongside the default `NormalPrior`.

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

Swap in the probit Thurstone Case V model by changing the model argument — the
data, accessors and inference methods are identical:

```julia
fit(ThurstoneCaseV(), MLE(), data)            # probit point estimates
fit(ThurstoneCaseV(), Bayesian(), data)       # probit posterior draws
```

With item covariates, model the strengths as `λᵢ = zᵢᵀβ`:

```julia
Z = [0.0 1.0;                         # one row of covariates per item
     1.0 0.0;
     2.0 1.0]
cd = CovariateData(data, Z, [:size, :age])

cfit = fit(BradleyTerryCovariates(), MLE(), cd)                 # ML coefficients
coef(cfit)                                                      # point estimates β (StatsAPI)
coefnames(cfit)                                                 # covariate names
confint(cfit; level=0.95)                                      # confidence intervals

fit(BradleyTerryCovariates(), StepwiseMLE(criterion=:BIC), cd)  # stepwise selection
fit(BradleyTerryCovariates(), Bayesian(), cd, HorseshoePrior()) # shrinkage
fit(BradleyTerryCovariates(), Bayesian(), cd, SpikeSlabPrior()) # selection + PIPs
```

For rater heterogeneity, supply per-rater comparisons (winner, loser and rater
labels, one entry per judgement) and read off the estimated reliabilities:

```julia
rd = RaterData(["A", "C", "B"], ["B", "A", "C"], ["r1", "r1", "r2"])  # winner, loser, rater
rfit = fit(BradleyTerryRaterHeterogeneity(), MLE(), rd)
rater_reliabilities(rfit)                                       # q_r per rater
strengths(rfit)                                                # consensus scale

ifit = fit(BradleyTerryIntransitive(), MLE(), data)            # plain PairwiseData
intransitivity(ifit)                                           # skew-symmetric γ matrix
```

## Model checking

A fit is a [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl)
`StatisticalModel` that stores the data it was fit to, so it plugs into the wider
Julia statistics ecosystem and the diagnostics are model-only:

```julia
fit1 = fit(BradleyTerry(), MLE(), data)
aic(fit1); bic(fit1); loglikelihood(fit1); dof(fit1); nobs(fit1)   # StatsAPI
coef(fit1); coefnames(fit1); vcov(fit1); stderror(fit1); confint(fit1)

fit2 = fit(BradleyTerry(), Bayesian(), data)
waic(fit2); loo(fit2)                                  # Bayesian predictive criteria
ssr(fit1)                                              # scale separation reliability
split_half_reliability(BradleyTerry(), MLE(), data)    # estimate stability

# Compare and stress-test competing models
crossvalidate(BradleyTerry(), MLE(), data)             # k-fold predictive log loss
rank_correlation(fit1, fit2)                           # do conclusions survive a model swap?
compare(fit1, fit2; criterion=:aic)                    # information-criterion table
```

See the [Diagnostics](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/diagnostics/)
and [Comparison](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/comparison/)
pages for the full set (WAIC/PSIS-LOO, likelihood-ratio test, decision-level
agreement, …).

## Documentation

The [documentation](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/)
includes worked tutorials on simulated data for both models. For Bradley–Terry:
the [plain model](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/bradley_terry/)
(MLE and Bayesian), the
[anchored model](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/anchored_bt/)
(measure half the items and predict the rest, including group-averaged anchors),
the [covariate model](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/covariate_bt/)
(MLE, stepwise selection and shrinkage / spike-and-slab priors), the
[rater-heterogeneity model](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/rater_heterogeneity_bt/)
(recovering rater reliabilities) and the
[intransitive model](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/intransitivity_bt/)
(detecting departures from a single scale). The
[Thurstone Case V](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/thurstone_case_v/)
model has matching [anchored](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/thurstone_case_v_anchored/)
and [covariate](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/thurstone_case_v_covariates/)
tutorials. See the full
[API reference](https://jmarsh96.github.io/ComparativeJudgement.jl/dev/api/) for
the complete interface.
