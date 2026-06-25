```@meta
CurrentModule = ComparativeJudgement
```

# ComparativeJudgement

[ComparativeJudgement](https://github.com/jmarsh96/ComparativeJudgement.jl)
fits comparative judgement models to pairwise comparison data: items are
placed on a latent strength scale from the outcomes of head-to-head
comparisons.

The package implements two pairwise-comparison models — **Bradley–Terry**
(logistic link, ``P(i \text{ beats } j) = \operatorname{logistic}(\lambda_i -
\lambda_j)``) and **Thurstone Case V** (probit link, ``\Phi(\lambda_i -
\lambda_j)``) — each with the same set of workflows:

- **Maximum likelihood** — fast point estimates and rankings.
- **Bayesian** — Gibbs sampling for full posterior uncertainty (Pólya-Gamma
  augmentation for Bradley–Terry, Albert–Chib truncated-normal augmentation for
  Thurstone).
- **Anchored** — known measurements for a few items calibrate the latent scale,
  so measurements can be predicted for all items (MLE or Bayesian).
- **Covariate** — latent strengths are explained by item covariates,
  ``\lambda_i = z_i^\top\beta``, with MLE, stepwise selection, and Bayesian
  shrinkage priors.

Bradley–Terry additionally offers two structural extensions, each fittable by
MLE or Bayesian:

- **Rater heterogeneity** — a mixture in which rater ``r`` follows Bradley–Terry
  with reliability ``q_r`` and otherwise guesses,
  ``q_r\,\sigma(\lambda_i - \lambda_j) + (1 - q_r)/2``, so unreliable assessors
  are down-weighted and flagged.
- **Intransitivity** — a skew-symmetric per-pair term ``\gamma_{ij} = -\gamma_{ji}``
  added to the predictor, ``\operatorname{logit} P(i \succ j) = (\lambda_i -
  \lambda_j) + \gamma_{ij}``, measuring departures from a single transitive scale.

See the [Bradley–Terry](bradley_terry.md) and [Thurstone Case V](thurstone_case_v.md)
tutorials for worked examples on simulated data, including the
[rater-heterogeneity](rater_heterogeneity_bt.md) and
[intransitive](intransitivity_bt.md) models, and the [API reference](api.md)
for the full interface.

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
