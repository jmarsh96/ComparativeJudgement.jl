# Bradley–Terry models

The Bradley–Terry model describes pairwise comparisons between items. Each
item ``i`` has a latent strength ``\lambda_i``, and the probability that it
wins a comparison against item ``j`` is

```math
P(i \text{ beats } j) = \frac{1}{1 + e^{-(\lambda_i - \lambda_j)}}.
```

This page fits the plain Bradley–Terry model two ways on one simulated dataset:

- **Maximum likelihood** ([`MLE`](@ref)) — fast point estimates of the
  strengths, good for ranking items.
- **Bayesian** ([`Bayesian`](@ref)) — a Pólya-Gamma augmented Gibbs sampler
  giving full posterior distributions, so you also get uncertainty for every
  strength and win probability.

Two extensions build on it, each with its own page: the
[anchored model](anchored_bt.md) calibrates the latent scale to real
measurements, and the [covariate model](covariate_bt.md) explains strengths with
item covariates.

## Simulating comparison data

We simulate 30 items with known, evenly spaced strengths and 600 comparisons
in total, each between a randomly chosen pair of items. This is the typical
comparative judgement regime: with 435 possible pairs, most pairs are
compared only once or twice and some never meet at all:

```@example bt
using ComparativeJudgement
using Random
using Plots

rng = MersenneTwister(42)

labels = ["S" * lpad(i, 2, '0') for i in 1:30]   # S01 … S30
n = length(labels)
λ_true = collect(range(-1.5, 1.5, length=n))     # S01 weakest … S30 strongest

logistic(x) = 1 / (1 + exp(-x))

n_comparisons = 600
wins = zeros(Int, n, n)
for _ in 1:n_comparisons
    i = rand(rng, 1:n)
    j = rand(rng, 1:n-1)
    j = j >= i ? j + 1 : j                       # distinct random pair
    if rand(rng) < logistic(λ_true[i] - λ_true[j])
        wins[i, j] += 1
    else
        wins[j, i] += 1
    end
end

data = PairwiseData(wins, labels)
nothing # hide
```

`wins[i, j]` counts how often item `i` beat item `j`; [`PairwiseData`](@ref)
pairs the matrix with the item labels, which can be any type (strings,
symbols, integers, …).

## Maximum likelihood

[`fit`](@ref) takes the model, the inference method, and the data:

```@example bt
fitted_mle = fit(BradleyTerry(), MLE(), data)
fitted_mle.converged
```

(`fit(BradleyTerry(), data)` is a shorthand for the same thing.)
[`strengths`](@ref) returns the estimated ``\hat\lambda``, centred to sum to
zero — directly comparable to `λ_true`:

```@example bt
λ̂ = strengths(fitted_mle)
scatter(λ_true, λ̂;
        xlabel="true strength λ", ylabel="MLE estimate λ̂",
        label="items", legend=:topleft)
plot!(identity, -2.6:0.1:2.6; linestyle=:dash, color=:black, label="perfect recovery")
```

The estimates scatter around the diagonal with no systematic distortion —
the spread is sampling noise from ~40 comparisons per item (the fit agrees
with R's `BradleyTerry2` to solver precision on this data).

Ranking the items is a `sortperm` away. With only 600 comparisons the
recovered order is close to the truth (`S30`, `S29`, …, `S01`) but adjacent
items — separated by just 0.10 on the latent scale — do swap:

```@example bt
labels[sortperm(λ̂, rev=true)]
```

[`probability`](@ref) gives fitted win probabilities, by label or by index,
and [`loglikelihood`](@ref) the log-likelihood at the estimate:

```@example bt
(probability(fitted_mle, "S30", "S01"), loglikelihood(fitted_mle))
```

## Bayesian inference

The Bayesian fit runs a Gibbs sampler (Pólya-Gamma augmentation makes every
update conjugate) and returns posterior draws instead of a point estimate.
[`Bayesian`](@ref) controls the run: `n_samples`, `n_burnin`, `thin`, and
`center` (each sweep re-centres ``\lambda`` to sum to zero, fixing the
location that the likelihood leaves free). The prior on ``\lambda`` is a
[`NormalPrior`](@ref), by default ``N(0, 10 I)``:

```@example bt
method = Bayesian(n_samples=2000, n_burnin=500)
fitted_bayes = fit(BradleyTerry(), method, data, NormalPrior(n); rng=rng)
nothing # hide
```

Posterior summaries come from [`posterior_mean`](@ref),
[`posterior_std`](@ref), and [`credible_interval`](@ref). Plotting the
posterior means with their 95% credible intervals against the truth makes
the sparsity visible: roughly 40 comparisons per item leave each strength
known only to within a few tenths, and nearly all intervals straddle the
true values:

```@example bt
post = posterior_mean(fitted_bayes)
ci = [credible_interval(fitted_bayes, k; prob=0.95) for k in 1:n]
lo, hi = first.(ci), last.(ci)

scatter(1:n, post;
        yerror=(post .- lo, hi .- post),
        xlabel="item", ylabel="latent strength λ",
        xticks=(1:5:n, labels[1:5:n]),
        label="posterior mean, 95% CI", legend=:topleft)
scatter!(1:n, λ_true; marker=:x, color=:red, label="truth")
```

```@example bt
credible_interval(fitted_bayes, 1; prob=0.95)   # 95% CI for item S01's strength
```

For Bayesian fits, [`probability`](@ref) returns the posterior mean win
probability, which accounts for the uncertainty in the strengths:

```@example bt
(probability(fitted_bayes, "S30", "S01"), probability(fitted_bayes, "S16", "S15"))
```
