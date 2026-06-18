# Anchored covariate Bradley–Terry models

The [covariate model](covariate_bt.md) explains item strengths with covariates
(``\lambda_i = z_i^\top\beta``); the [anchored model](bradley_terry.md) calibrates
the latent scale to real measurements (``y = a + b\lambda + \varepsilon``). Their
combination — `Anchored{Covariates{BradleyTerry}}`, aliased
[`BradleyTerryCovariatesAnchored`](@ref) — does **both at once**:

```math
\lambda_i = z_i^\top\beta,
\qquad
\operatorname{logit} P(i \text{ beats } j) = (z_i - z_j)^\top\beta,
\qquad
y_k = a + b\,(z_k^\top\beta) + \varepsilon_k, \;\; \varepsilon_k \sim N(0,\sigma^2),\; k \in S.
```

Everything is fit jointly over ``(\beta, a, b, \sigma^2)``: the comparisons and the
anchors both inform the coefficients ``\beta``, the anchors pin down the calibration
``(a, b, \sigma^2)``. The payoff over the plain anchored model is that, because
strengths are a *function of covariates*, we can **predict a measurement for an item
that was never compared and never measured** — from its covariates alone,
``y^\* = a + b\,z_{\text{new}}^\top\beta``.

!!! note "Composition and identifiability"
    The model is literally the two wrappers composed:
    `Anchored(Covariates(BradleyTerry()))`, fit with an [`AnchoredData`](@ref) that
    wraps a [`CovariateData`](@ref). Because ``\beta`` is identified by the
    comparisons (constant covariate columns are rejected), ``\lambda = Z\beta`` has a
    determined location and is used **uncentred** — its absolute scale is exactly what
    ``(a, b)`` calibrate to.

The same four workflows as the covariate model are available — [`MLE`](@ref),
[`StepwiseMLE`](@ref), and [`Bayesian`](@ref) with a [`NormalPrior`](@ref),
[`HorseshoePrior`](@ref) or [`SpikeSlabPrior`](@ref) on ``\beta``.

## Simulating data

We simulate 60 items with five covariates (only the first two matter), draw the
comparisons, and **measure half the items** — holding out the other half to test
measurement prediction. The true calibration is ``a = 2``, ``b = 3``, ``\sigma = 0.3``:

```@example cova
using ComparativeJudgement
using Random
using Statistics: mean, quantile

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
    for _ in 1:8
        rand(rng) < logistic(λ_true[i] - λ_true[j]) ? (wins[i, j] += 1) : (wins[j, i] += 1)
    end
end

cd = CovariateData(PairwiseData(wins, labels), Z, [:x1, :x2, :x3, :x4, :x5])

a_true, b_true, σ_true = 2.0, 3.0, 0.3
measured = 1:2:K                                 # measure every other item
held_out = 2:2:K                                 # predict these
y_true = a_true .+ b_true .* λ_true              # noiseless measurement scale
y_obs  = y_true[measured] .+ σ_true .* randn(rng, length(measured))

acd = AnchoredData(cd, labels[measured], y_obs)
nothing # hide
```

[`AnchoredData`](@ref) wraps the [`CovariateData`](@ref) together with the measured
items' labels and values (a `Dict` works too). Only the measured covariates relate
linearly to the measurements; the held-out items will be predicted purely from their
covariates and comparison record.

## Maximum likelihood

```@example cova
fitted = fit(BradleyTerryCovariatesAnchored(), MLE(), acd)
coefficients(fitted)
```

[`coefficients`](@ref) recovers ``\beta \approx (1.5, -1.0, 0, 0, 0)``, and
[`calibration`](@ref) the joint estimates of ``(a, b, \sigma^2)``:

```@example cova
calibration(fitted)
```

[`coefficient_intervals`](@ref) gives Wald confidence intervals from the joint Fisher
information; the two signal covariates separate cleanly from the three null ones:

```@example cova
using Plots

β̂   = collect(values(coefficients(fitted)))
lohi = collect(values(coefficient_intervals(fitted; level=0.95)))
lo, hi = first.(lohi), last.(lohi)
scatter(1:p, β̂; yerror=(β̂ .- lo, hi .- β̂), label="MLE, 95% CI",
        xticks=(1:p, string.(cd.names)), xlabel="covariate", ylabel="coefficient β",
        legend=:topright)
scatter!(1:p, β_true; marker=:x, markersize=7, color=:red, label="truth")
hline!([0]; color=:black, linestyle=:dash, label="")
```

### Predicting held-out measurements

[`predict`](@ref) maps items onto the measurement scale via ``a + b\,\lambda``. With
no item argument it returns the predicted measurement for every item; we compare the
**never-measured** held-out items against their true values:

```@example cova
preds = predict(fitted)
rmse = sqrt(mean((preds[held_out] .- y_true[held_out]).^2))
round(rmse, digits=3)
```

The calibration line maps latent strengths to the measurement scale; the measured
items pin it down and the held-out items ride along it:

```@example cova
cal = calibration(fitted)
λ̂ = strengths(fitted)
scatter(λ̂[measured], y_obs; xlabel="estimated strength λ̂ = Zβ̂",
        ylabel="measurement y", label="measured items", legend=:topleft)
scatter!(λ̂[held_out], y_true[held_out]; marker=:x, color=:red,
         label="held-out truth")
plot!(x -> cal.a + cal.b * x, extrema(λ̂)...; color=:black, label="fitted line a + b·λ")
```

Predicted-versus-true for the held-out half tracks the diagonal — none of these items
was measured:

```@example cova
scatter(y_true[held_out], preds[held_out]; label="held-out items",
        xlabel="true measurement y", ylabel="predicted measurement ŷ", legend=:topleft)
plot!(identity, extrema(y_true[held_out])...; linestyle=:dash, color=:black,
      label="perfect prediction")
```

### Predicting a brand-new item

Because strengths are a function of covariates, we can predict the measurement of an
item that is **not in the dataset at all** — passing its covariate vector to
[`predict`](@ref). Its latent strength is ``z_{\text{new}}^\top\hat\beta`` and its
measurement ``a + b\,z_{\text{new}}^\top\hat\beta``:

```@example cova
z_new = [1.0, -0.5, 0.0, 0.0, 0.0]
(point = predict(fitted, z_new),
 interval = predict(fitted, z_new; prob=0.9))   # normal interval from σ̂²
```

## Bayesian inference

A Bayesian fit returns joint posterior draws of ``\beta``, ``(a, b)``, ``\sigma^2``
and the strengths. With the default [`NormalPrior`](@ref) on ``\beta``:

```@example cova
method = Bayesian(n_samples=1500, n_burnin=500)
bayes = fit(BradleyTerryCovariatesAnchored(), method, acd; rng=MersenneTwister(1))
coefficients(bayes)
```

[`coefficient_intervals`](@ref) now gives posterior credible intervals, and
[`calibration`](@ref) the posterior-mean calibration:

```@example cova
(coefficients = coefficient_intervals(bayes; level=0.95), calibration = calibration(bayes))
```

```@example cova
bm   = collect(values(coefficients(bayes)))
lohi = collect(values(coefficient_intervals(bayes; level=0.95)))
lo, hi = first.(lohi), last.(lohi)
scatter(1:p, bm; yerror=(bm .- lo, hi .- bm), label="posterior mean, 95% CI",
        xticks=(1:p, string.(cd.names)), xlabel="covariate", ylabel="coefficient β")
scatter!(1:p, β_true; marker=:x, markersize=7, color=:red, label="truth")
hline!([0]; color=:black, linestyle=:dash, label="")
```

For the held-out items, [`predict`](@ref) gives full posterior-predictive intervals
that propagate both the comparison and the calibration uncertainty:

```@example cova
λ_post = posterior_mean(bayes)
pred_ints = [predict(bayes, k; prob=0.9, rng=MersenneTwister(k)) for k in held_out]
plo, phi = first.(pred_ints), last.(pred_ints)
preds_b = predict(bayes)

scatter(λ_post[measured], y_obs; xlabel="posterior mean strength λ",
        ylabel="measurement y", label="measured items", legend=:topleft)
scatter!(λ_post[held_out], preds_b[held_out];
         yerror=(preds_b[held_out] .- plo, phi .- preds_b[held_out]),
         label="held-out predictions, 90% PI")
scatter!(λ_post[held_out], y_true[held_out]; marker=:x, color=:red, label="held-out truth")
```

The latent-strength accessors ([`posterior_mean`](@ref), [`credible_interval`](@ref))
and a posterior-predictive draw for an unseen covariate row work too:

```@example cova
(credible_interval(bayes, 1; prob=0.95),
 quantile(predict(bayes, z_new; rng=MersenneTwister(7)), [0.05, 0.95]))
```

### Shrinkage and selection

The covariate selection tools carry over unchanged. The [`HorseshoePrior`](@ref)
shrinks the null coefficients while leaving the signal almost untouched:

```@example cova
hs = fit(BradleyTerryCovariatesAnchored(), method, acd, HorseshoePrior();
         rng=MersenneTwister(2))
coefficients(hs)
```

The [`SpikeSlabPrior`](@ref) reports posterior inclusion probabilities — high for the
two real covariates, near zero for the rest:

```@example cova
ss = fit(BradleyTerryCovariatesAnchored(), method, acd, SpikeSlabPrior();
         rng=MersenneTwister(3))
pip = collect(values(inclusion_probabilities(ss)))
bar(string.(cd.names), pip; legend=false, ylims=(0, 1), xlabel="covariate",
    ylabel="posterior inclusion probability", title="Spike-and-slab selection")
hline!([0.5]; color=:black, linestyle=:dash)
```

And [`StepwiseMLE`](@ref) selects covariates by joint AIC/BIC, exactly as for the
covariate model — here BIC keeps only the two real covariates:

```@example cova
selected = fit(BradleyTerryCovariatesAnchored(),
               StepwiseMLE(direction=:both, criterion=:BIC), acd)
(selected = selected.result.selected, coefficients = coefficients(selected))
```

Priors can be controlled individually by passing a full
[`AnchoredCovariatePrior`](@ref) (a ``\beta`` prior plus the calibration and variance
priors); passing a bare ``\beta`` prior, as above, wraps it with weakly-informative
defaults.

