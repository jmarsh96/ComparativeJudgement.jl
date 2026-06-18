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

The two signal covariates `x1` and `x2` have a clear (positive and negative)
linear relationship with the true strengths, while the noise covariate `x3`
shows none — this is the structure the model has to recover:

```@example cov
using Plots
using Statistics: mean, quantile

scatter(Z[:, 1], λ_true; label="x1 (β = 1.5)", markersize=3,
        xlabel="covariate value", ylabel="true strength λ = Zβ", legend=:topleft)
scatter!(Z[:, 2], λ_true; label="x2 (β = -1.0)", markersize=3)
scatter!(Z[:, 3], λ_true; label="x3 (β = 0, noise)", markersize=3)
```

Sorting the items by their true strength turns the raw comparison outcomes into a
clean gradient — the off-diagonal structure the fit exploits:

```@example cov
order = sortperm(λ_true)
N = wins .+ wins'                                   # comparisons per pair
winprop = [N[i, j] == 0 ? 0.5 : wins[i, j] / N[i, j] for i in 1:K, j in 1:K]
heatmap(winprop[order, order]; c=:RdBu, clims=(0, 1), aspect_ratio=1,
        xlabel="item (ascending strength)", ylabel="item (ascending strength)",
        title="P(row item beats column item)", size=(470, 420))
```

## Maximum likelihood

```@example cov
fitted = fit(BradleyTerryCovariates(), MLE(), cd)
coefficients(fitted)
```

[`coefficients`](@ref) returns ``\hat\beta`` keyed by covariate name — close to
the true ``(1.5, -1.0, 0, 0, 0)``. [`coefficient_std`](@ref) returns the standard
errors (from the inverse Fisher information) and [`coefficient_intervals`](@ref)
the Wald confidence intervals:

```@example cov
coefficient_intervals(fitted; level=0.95)
```

Plotting the estimates with their 95% confidence intervals separates the two
signal covariates from the three noise covariates, whose intervals straddle zero:

```@example cov
β̂   = collect(values(coefficients(fitted)))
lohi = collect(values(coefficient_intervals(fitted; level=0.95)))
lo, hi = first.(lohi), last.(lohi)
scatter(1:p, β̂; yerror=(β̂ .- lo, hi .- β̂), label="MLE, 95% CI",
        xticks=(1:p, string.(cd.names)), xlabel="covariate", ylabel="coefficient β",
        legend=:topright)
scatter!(1:p, β_true; marker=:x, markersize=7, color=:red, label="truth")
hline!([0]; color=:black, linestyle=:dash, label="")
```

[`strengths`](@ref) recovers the latent strengths ``\lambda = Z\hat\beta`` (centred
to sum to zero), and [`probability`](@ref) gives fitted win probabilities by label
or index:

```@example cov
λ̂ = strengths(fitted)
(probability(fitted, "item01", "item02"), loglikelihood(fitted))
```

The recovered strengths track the (centred) truth along the diagonal:

```@example cov
λ_true_c = λ_true .- mean(λ_true)
scatter(λ_true_c, λ̂; label="items", markersize=3, legend=:topleft,
        xlabel="true strength λ", ylabel="estimated strength λ̂")
plot!(identity, extrema(λ_true_c)...; linestyle=:dash, color=:black,
      label="perfect recovery")
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

Each accepted step lowers the criterion until no add/remove move helps, at which
point the search stops. Tracing the BIC over the path shows the descent and the
growing model size:

```@example cov
steps = [t.step for t in selected.result.trace]
ic    = [t.ic   for t in selected.result.trace]
nsel  = [length(t.selected) for t in selected.result.trace]
plot(steps, ic; marker=:circle, label="BIC", xlabel="step", ylabel="BIC",
     xticks=steps, legend=:topright)
annotate!([(steps[i], ic[i], text("  $(nsel[i]) cov", 8, :left, :bottom))
           for i in eachindex(steps)])
```

## Bayesian inference

A Bayesian fit returns posterior draws of ``\beta``. With the default
[`NormalPrior`](@ref):

```@example cov
method = Bayesian(n_samples=1500, n_burnin=500)
bayes = fit(BradleyTerryCovariates(), method, cd; rng=MersenneTwister(1))
coefficients(bayes)
```

The coefficient uncertainty comes from the same accessors as the MLE fit:
[`coefficient_std`](@ref) gives posterior standard deviations and
[`coefficient_intervals`](@ref) posterior credible intervals (the raw draws are
also available in `bayes.result.β_samples`, an `n_samples × p` matrix):

```@example cov
coefficient_intervals(bayes; level=0.95)
```

Their posterior means and 95% credible intervals recover the truth, with every
noise covariate's interval comfortably covering zero:

```@example cov
bm   = collect(values(coefficients(bayes)))
lohi = collect(values(coefficient_intervals(bayes; level=0.95)))
lo, hi = first.(lohi), last.(lohi)
scatter(1:p, bm; yerror=(bm .- lo, hi .- bm), label="posterior mean, 95% CI",
        xticks=(1:p, string.(cd.names)), xlabel="covariate", ylabel="coefficient β")
scatter!(1:p, β_true; marker=:x, markersize=7, color=:red, label="truth")
hline!([0]; color=:black, linestyle=:dash, label="")
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

Overlaying the horseshoe posterior means on the `NormalPrior` ones makes the
shrinkage explicit: the two signal coefficients are barely moved, while the three
noise coefficients are pulled hard toward zero:

```@example cov
mn = vec(mean(bayes.result.β_samples, dims=1))    # Normal prior
mh = vec(mean(hs.result.β_samples, dims=1))        # Horseshoe prior
scatter(1:p, mn; label="Normal prior", xticks=(1:p, string.(cd.names)),
        xlabel="covariate", ylabel="posterior mean β", legend=:topright)
scatter!(1:p, mh; marker=:diamond, label="Horseshoe prior")
scatter!(1:p, β_true; marker=:x, markersize=7, color=:red, label="truth")
hline!([0]; color=:black, linestyle=:dash, label="")
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

As a bar chart against the prior inclusion probability ``\pi_0 = 0.5`` (dashed),
the two real covariates sit near 1 and the three noise covariates near zero —
recovering the true sparsity pattern:

```@example cov
pip = collect(values(inclusion_probabilities(ss)))
bar(string.(cd.names), pip; legend=false, ylims=(0, 1),
    xlabel="covariate", ylabel="posterior inclusion probability",
    title="Spike-and-slab variable selection")
hline!([0.5]; color=:black, linestyle=:dash)
```

The two real covariates have inclusion probabilities near 1, the three noise
covariates near zero — recovering the true sparsity pattern.
