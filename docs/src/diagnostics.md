# Model diagnostics

These functions assess a **single** fitted model: how well it predicts, and how
reliably its scale is estimated. (To compare *competing* models, or to check
whether your conclusions survive a change of model, see
[Model comparison](comparison.md).)

| Function | Question | Fit type |
|----------|----------|----------|
| [`aic`](@ref), [`bic`](@ref) | Information criteria | `MLE` |
| [`waic`](@ref) | Bayesian out-of-sample accuracy | `Bayesian` |
| [`loo`](@ref) | PSIS leave-one-out CV | `Bayesian` |
| [`ssr`](@ref) | Scale separation reliability | any |
| [`split_half_reliability`](@ref) | Estimate stability | any |

## A worked dataset

We simulate 20 items with evenly spaced strengths and a dozen comparisons per
pair, then fit the Bradley–Terry model by maximum likelihood and by MCMC:

```@example diag
using ComparativeJudgement
using Random

rng = MersenneTwister(2024)

K = 20
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

mle = fit(BradleyTerry(), MLE(), data)
bayes = fit(BradleyTerry(), Bayesian(n_samples=2000, n_burnin=500), data; rng=rng)
nothing # hide
```

## Information criteria (AIC / BIC)

For maximum-likelihood fits, [`aic`](@ref) and [`bic`](@ref) trade goodness of
fit against the number of parameters [`nparams`](@ref); [`nobs`](@ref) is the
number of comparisons used in the BIC penalty. Lower is better.

```@example diag
(aic = aic(mle, data), bic = bic(mle, data),
 nparams = nparams(mle), nobs = nobs(data))
```

AIC estimates out-of-sample predictive performance, BIC approximates a Bayes
factor; for human-judgement data — where no candidate is exactly true — the
predictive emphasis of AIC is usually the more defensible. Both are defined for
maximum-likelihood fits only; for a [`Bayesian`](@ref) fit use [`waic`](@ref) or
[`loo`](@ref) below.

## WAIC and PSIS-LOO

For Bayesian fits, [`waic`](@ref) and [`loo`](@ref) estimate out-of-sample
predictive accuracy from the pointwise log-likelihood
([`pointwise_loglikelihood`](@ref)) without refitting. They report the expected
log pointwise predictive density (`elpd`, higher is better) and an information
criterion on the deviance scale (lower is better):

```@example diag
waic(bayes, data)
```

```@example diag
loo(bayes, data)
```

[`loo`](@ref) uses Pareto-smoothed importance sampling and additionally returns a
per-observation Pareto-``\hat k`` diagnostic: values above 0.7 flag comparisons
for which the LOO approximation is unreliable. The two criteria normally agree
closely, and both track the maximum-likelihood AIC on a well-behaved fit.

## Scale separation reliability

[`ssr`](@ref) is the house metric of the CJ literature: the proportion of the
observed variance in the estimated strengths attributable to true differences
between items rather than to estimation error. The standard errors come from the
posterior for a Bayesian fit, and from the observed information for a plain
maximum-likelihood fit (so the data is passed):

```@example diag
(ssr_mle = ssr(mle, data), ssr_bayes = ssr(bayes))
```

The two routes agree closely. Treat SSR as a descriptive summary rather than a
model-selection criterion: it is inflated by adaptive pair selection and is in
part an artefact of the design.

## Split-half reliability

[`split_half_reliability`](@ref) measures how stable the estimates are by
repeatedly splitting the comparisons into two random halves, refitting the *same*
model to each, and correlating the two strength vectors. It reports the mean
correlation over the splits and its Spearman–Brown step-up; the literature treats
≥ 0.7 as good:

```@example diag
split_half_reliability(BradleyTerry(), MLE(), data; n_splits=50, rng=rng)
```
