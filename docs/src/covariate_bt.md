# Covariate Bradley–Terry models

Sometimes the items being compared carry **covariates** — measurable features —
and we want to explain their latent strengths with those features rather than
estimate one free strength per item. A [`Covariates`](@ref) model parameterises
each strength as a linear combination of item covariates,

```math
\lambda_i = z_i^\top \beta,
\qquad
P(i \text{ beats } j) = \frac{1}{1 + e^{-(z_i - z_j)^\top \beta}},
```

so the unknown is the coefficient vector ``\beta \in \mathbb{R}^p`` (length =
number of covariates) instead of the ``K`` item strengths. Because the comparison
log-odds depend only on the covariate *difference* ``z_i - z_j``, this is exactly
logistic regression on the difference design matrix — the same Pólya-Gamma
machinery used for the plain [Bradley–Terry models](bradley_terry.md) applies,
with the design matrix swapped from item indicators to covariate differences.

This is the "predictor" model of the R package `BradleyTerry2`.

!!! note "No intercept"
    An overall intercept (a covariate constant across items) cancels in the
    differences ``z_i - z_j`` and is not identifiable. [`CovariateData`](@ref)
    rejects constant columns.

ComparativeJudgement offers four ways to fit and select covariate models:

- **Maximum likelihood** ([`MLE`](@ref)) — point estimates of ``\beta``.
- **Stepwise selection** ([`StepwiseMLE`](@ref)) — greedy AIC/BIC variable
  selection around the MLE.
- **Bayesian** ([`Bayesian`](@ref)) — Gibbs sampling with a [`NormalPrior`](@ref),
  a [`HorseshoePrior`](@ref) for global-local shrinkage, or a
  [`SpikeSlabPrior`](@ref) for selection with posterior inclusion probabilities.

## Simulating covariate data

We simulate 60 items, each with five covariates. Only the first two drive the
strengths (``\beta = (1.5, -1.0)``); the other three are noise:

```@example cov
using ComparativeJudgement
using Random

rng = MersenneTwister(2025)

K = 60
β_true = [1.5, -1.0, 0.0, 0.0, 0.0]
p = length(β_true)

Z = randn(rng, K, p)
λ_true = Z * β_true
labels = ["item" * lpad(i, 2, '0') for i in 1:K]

logistic(x) = 1 / (1 + exp(-x))

wins = zeros(Int, K, K)
for i in 1:K, j in (i + 1):K
    for _ in 1:8                                  # 8 comparisons per pair
        if rand(rng) < logistic(λ_true[i] - λ_true[j])
            wins[i, j] += 1
        else
            wins[j, i] += 1
        end
    end
end

cd = CovariateData(PairwiseData(wins, labels), Z,
                   [:x1, :x2, :x3, :x4, :x5])
nothing # hide
```

[`CovariateData`](@ref) bundles the comparisons with the ``K \times p`` covariate
matrix `Z` (one row per item) and the covariate names. The names default to
`:x1, :x2, …` if omitted, and a `name => values` form is also accepted:
`CovariateData(data, :x1 => Z[:,1], :x2 => Z[:,2])`.

## Maximum likelihood

```@example cov
fitted = fit(BradleyTerryCovariates(), MLE(), cd)
coefficients(fitted)
```

[`coefficients`](@ref) returns ``\hat\beta`` keyed by covariate name — close to
the true ``(1.5, -1.0, 0, 0, 0)``. [`strengths`](@ref) recovers the latent
strengths ``\lambda = Z\beta`` (centred to sum to zero), and [`probability`](@ref)
gives fitted win probabilities by label or index:

```@example cov
λ̂ = strengths(fitted)
(probability(fitted, "item01", "item02"), loglikelihood(fitted))
```

## Stepwise selection

[`StepwiseMLE`](@ref) greedily adds/removes covariates to optimise an information
criterion (`:AIC` or `:BIC`), in direction `:forward`, `:backward`, or `:both`:

```@example cov
selected = fit(BradleyTerryCovariates(),
               StepwiseMLE(direction=:both, criterion=:BIC), cd)
coefficients(selected)
```

BIC keeps only the two real covariates. The search path is recorded in
`selected.result.trace` and the retained indices in `selected.result.selected`:

```@example cov
selected.result.selected
```

## Bayesian inference

A Bayesian fit returns posterior draws of ``\beta``. With the default
[`NormalPrior`](@ref):

```@example cov
method = Bayesian(n_samples=1500, n_burnin=500)
bayes = fit(BradleyTerryCovariates(), method, cd; rng=MersenneTwister(1))
coefficients(bayes)
```

Posterior summaries of the latent strengths come from [`posterior_mean`](@ref),
[`posterior_std`](@ref) and [`credible_interval`](@ref), exactly as for the plain
Bayesian fit:

```@example cov
(posterior_mean(bayes)[1], credible_interval(bayes, 1; prob=0.95))
```

### Horseshoe shrinkage

The [`HorseshoePrior`](@ref) is a global-local shrinkage prior: it pulls small
coefficients hard toward zero while leaving large ones almost untouched, without
a hard in/out decision. The null covariates collapse toward zero:

```@example cov
hs = fit(BradleyTerryCovariates(), method, cd, HorseshoePrior();
         rng=MersenneTwister(2))
coefficients(hs)
```

### Spike-and-slab selection

The [`SpikeSlabPrior`](@ref) mixes a wide "slab" with a narrow "spike" and gives
a per-covariate **posterior inclusion probability** via
[`inclusion_probabilities`](@ref) — a Bayesian analogue of stepwise selection:

```@example cov
ss = fit(BradleyTerryCovariates(), method, cd, SpikeSlabPrior();
         rng=MersenneTwister(3))
inclusion_probabilities(ss)
```

The two real covariates have inclusion probabilities near 1, the three noise
covariates near the prior level — recovering the true sparsity pattern.
