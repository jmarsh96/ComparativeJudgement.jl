# Model comparison

These functions contrast **competing** models fit to the same comparisons, and
ask the question that matters most in practice: would the conclusions of the
study change under a different but equally defensible model? (To assess a single
fit on its own, see [Model diagnostics](diagnostics.md).)

| Function | Question |
|----------|----------|
| [`lrtest`](@ref) | Is the larger nested (covariate) model justified? |
| [`crossvalidate`](@ref) | Which model predicts held-out comparisons best? |
| [`rank_correlation`](@ref) | Is the rank order stable across models? |
| [`top_k_agreement`](@ref), [`boundary_agreement`](@ref) | Do the operational decisions change? |
| [`compare`](@ref) | Tabulate a slate of models by one criterion |

## A worked dataset

```@example cmp
using ComparativeJudgement
using Random

rng = MersenneTwister(2025)

K = 24
λ_true = collect(range(2.0, -2.0, length=K))
labels = ["S" * lpad(i, 2, '0') for i in 1:K]
wins = zeros(Int, K, K)
for i in 1:K, j in (i + 1):K
    p = 1 / (1 + exp(-(λ_true[i] - λ_true[j])))
    for _ in 1:12
        rand(rng) < p ? (wins[i, j] += 1) : (wins[j, i] += 1)
    end
end
data = PairwiseData(wins, labels)
nothing # hide
```

## Likelihood-ratio test (nested covariate models)

The covariate models are the one nested family: dropping covariates from
``\lambda_i = z_i^\top\beta`` gives a sub-model, so [`lrtest`](@ref) applies. Here
two covariates drive the strengths and a third is noise; the test correctly finds
the noise covariate unjustified and the real one justified:

```@example cmp
Z = randn(rng, K, 3)
β_true = [1.8, -1.2, 0.0]                       # x3 is noise
λ = Z * β_true
w = zeros(Int, K, K)
for i in 1:K, j in (i + 1):K
    p = 1 / (1 + exp(-(λ[i] - λ[j])))
    for _ in 1:10
        rand(rng) < p ? (w[i, j] += 1) : (w[j, i] += 1)
    end
end
cdata = PairwiseData(w, labels)

full      = fit(BradleyTerryCovariates(), MLE(), CovariateData(cdata, Z, [:x1, :x2, :x3]))
drop_x3   = fit(BradleyTerryCovariates(), MLE(), CovariateData(cdata, Z[:, 1:2], [:x1, :x2]))
drop_x2   = fit(BradleyTerryCovariates(), MLE(), CovariateData(cdata, Z[:, 1:1], [:x1]))

(noise = lrtest(drop_x3, full), real = lrtest(drop_x2, drop_x3))
```

## Cross-validated predictive log loss

[`crossvalidate`](@ref) is the most direct and model-agnostic comparison: it
refits on each training fold and scores the held-out comparisons by the mean
logarithmic loss ([`log_loss`](@ref)). Lower is better, and it requires no
assumption that any candidate model is true. The covariate model built on the
*true* covariate predicts unseen comparisons better than one built on a random
covariate:

```@example cmp
cv_true = crossvalidate(BradleyTerryCovariates(), MLE(),
                        CovariateData(cdata, Z[:, 1:1], [:x1]); k=5, rng=rng)
cv_rand = crossvalidate(BradleyTerryCovariates(), MLE(),
                        CovariateData(cdata, randn(rng, K, 1), [:x1]); k=5, rng=rng)
(true_covariate = cv_true.mean_logloss, random_covariate = cv_rand.mean_logloss)
```

[`train_test_split`](@ref) and [`kfold`](@ref) are available directly if you want
to manage the partitioning yourself.

## Rank correlation and decision-level agreement

The technique that bears most directly on robustness is the simplest: fit the
competing models and compare the scales. [`rank_correlation`](@ref) measures the
stability of the full rank order. The Bradley–Terry and Thurstone models differ
only in their link function, so — as the literature predicts — they agree almost
perfectly:

```@example cmp
bt = fit(BradleyTerry(), MLE(), data)
th = fit(ThurstoneCaseV(), MLE(), data)
(spearman = rank_correlation(bt, th),
 kendall  = rank_correlation(bt, th; method=:kendall))
```

Rank correlation alone can mask disagreement at the boundaries that matter, so
also report agreement on the operational decisions — the top ``k`` items, or who
falls above a grade boundary:

```@example cmp
(top5 = top_k_agreement(bt, th, 5),
 boundary = boundary_agreement(bt, th, 0.0))
```

## Tabulating a slate of models

[`compare`](@ref) scores several fits on the same data by a single criterion
(`:aic`/`:bic` for maximum-likelihood fits, `:waic`/`:loo` for Bayesian fits) and
tabulates them by the gap ``\Delta`` from the best model:

```@example cmp
btb = fit(BradleyTerry(), Bayesian(n_samples=1500, n_burnin=500), data; rng=rng)
thb = fit(ThurstoneCaseV(), Bayesian(n_samples=1500, n_burnin=500), data; rng=rng)
compare(btb, thb; criterion=:loo, names=["Bradley–Terry", "Thurstone"])
```

A difference of a point or two between the link models is the expected, and
reportable, result: the conclusions are robust to the choice of link. The
instructive divergences arise instead from structural assumptions — ties,
intransitivity, rater heterogeneity, and adaptive selection.
