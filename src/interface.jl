"""
    fit(model, [method], data, [prior]; kwargs...)

Fit a comparative judgement `model` to `data` and return a
[`FittedComparativeModel`](@ref).

The inference `method` selects the estimation strategy ([`MLE`](@ref),
[`StepwiseMLE`](@ref) or [`Bayesian`](@ref)); when omitted, each model picks a
sensible default ([`MLE`](@ref) for [`BradleyTerry`](@ref) and
[`Covariates`](@ref) models, [`Bayesian`](@ref) for [`Anchored`](@ref) models).
Bayesian fits accept a prior and an `rng` keyword for reproducibility. The
anchored covariate model ([`BradleyTerryCovariatesAnchored`](@ref)) is fit with an
[`AnchoredData`](@ref) wrapping a [`CovariateData`](@ref) and supports all three
methods.
"""
function fit end

"""
    loglikelihood(fitted)

Log-likelihood of the data under the fitted model. For [`MLE`](@ref) fits this
is a scalar evaluated at the estimate; for [`Bayesian`](@ref) fits it is the
vector of log-likelihood values, one per retained posterior draw.
"""
function loglikelihood end

"""
    probability(fitted, i, j)
    probability(fitted, label_i, label_j)

Probability that item `i` beats item `j` under the fitted model. Items may be
given by index or by label. For Bayesian fits this is the posterior mean of
the win probability.
"""
function probability end

"""
    strengths(fitted)

Estimated latent strengths λ, one per item, in the order of `fitted.labels`.
For [`MLE`](@ref) fits these are the point estimates (centred to sum to zero);
for [`Bayesian`](@ref) fits this is the posterior mean, equivalent to
[`posterior_mean`](@ref).
"""
function strengths end

"""
    coefficients(fitted)

Estimated covariate coefficients β of a [`Covariates`](@ref) fit, keyed by
covariate name. Point estimates for [`MLE`](@ref)/[`StepwiseMLE`](@ref) fits
(selected covariates only); posterior means for [`Bayesian`](@ref) fits.
"""
function coefficients end

"""
    coefficient_std(fitted)

Uncertainty of the covariate coefficients β of a [`Covariates`](@ref) fit, keyed
by covariate name: standard errors (from the inverse Fisher information) for
[`MLE`](@ref)/[`StepwiseMLE`](@ref) fits, posterior standard deviations for
[`Bayesian`](@ref) fits.
"""
function coefficient_std end

"""
    coefficient_intervals(fitted; level=0.95)

Interval estimates for the covariate coefficients β of a [`Covariates`](@ref)
fit, as a named tuple of `(lo, hi)` keyed by covariate name. These are Wald
confidence intervals `β̂ ± z·SE` for [`MLE`](@ref)/[`StepwiseMLE`](@ref) fits and
posterior credible intervals for [`Bayesian`](@ref) fits.
"""
function coefficient_intervals end

"""
    inclusion_probabilities(fitted)

Posterior inclusion probabilities per covariate from a [`Bayesian`](@ref)
[`Covariates`](@ref) fit with a [`SpikeSlabPrior`](@ref), keyed by covariate
name.
"""
function inclusion_probabilities end

"""
    posterior_mean(fitted)

Posterior mean of the latent strengths λ of a [`Bayesian`](@ref) fit, one per
item, in the order of `fitted.labels`.
"""
function posterior_mean end

"""
    posterior_std(fitted)

Posterior standard deviation of the latent strengths λ of a
[`Bayesian`](@ref) fit, one per item, in the order of `fitted.labels`.
"""
function posterior_std end

"""
    credible_interval(fitted, k; prob=0.95)

Symmetric posterior credible interval `(lo, hi)` for the latent strength of
item `k` from a [`Bayesian`](@ref) fit.
"""
function credible_interval end

"""
    predict(fitted, k; prob=nothing, rng=Random.default_rng())
    predict(fitted, label; prob=nothing, rng=Random.default_rng())
    predict(fitted, z::AbstractVector; prob=nothing, rng=Random.default_rng())
    predict(fitted)

Posterior-predictive anchor measurements `y* = a + b·λ + ε` for an
[`Anchored`](@ref) fit, on the scale of the anchor values.

With an item index `k` or `label`, returns a vector of posterior-predictive
draws, or the symmetric `prob` credible interval `(lo, hi)` when `prob` is
given. With no item argument, returns the vector of posterior-predictive
means for all items.

For an anchored *covariate* fit ([`BradleyTerryCovariatesAnchored`](@ref)), a
covariate vector `z` predicts the measurement of an unseen item from its
covariates alone (`y* = a + b·zᵀβ`). [`MLE`](@ref) fits return the point
prediction, or a normal interval from `σ̂²` when `prob` is given.
"""
function predict end

"""
    calibration(fitted)

Posterior means of the calibration parameters of an [`Anchored`](@ref) fit,
as a named tuple `(a = ..., b = ..., σ² = ...)` for the anchor model
`y = a + b·λ + ε`, `ε ~ N(0, σ²)`.
"""
function calibration end
