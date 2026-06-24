# ─── Information criteria (AIC / BIC) ────────────────────────────────────────
#
# Computed from the *data* log-likelihood at the point estimate — the sum of the
# pointwise log densities — so the penalty terms of the ridge-penalised models
# (rater heterogeneity, intransitivity) do not enter, and anchored fits are
# scored on their comparisons alone. This matches `loglikelihood` for the
# unpenalised models. The grid helper `_ic` is shared with `StepwiseMLE`.

# Data log-likelihood at the point estimate (sum over observations).
_data_loglik(fitted, data) = sum(pointwise_loglikelihood(fitted, data))

const _MLEMethods = Union{MLE, StepwiseMLE}

"""
    deviance(fitted, data)

Residual deviance `-2·ℓ̂` of an [`MLE`](@ref)/[`StepwiseMLE`](@ref) fit, where
`ℓ̂` is the data log-likelihood at the point estimate.
"""
function deviance(fitted::FittedComparativeModel{M, I}, data) where {M, I <: _MLEMethods}
    return -2.0 * _data_loglik(fitted, data)
end

"""
    aic(fitted, data)

Akaike information criterion `-2·ℓ̂ + 2k` of an [`MLE`](@ref)/[`StepwiseMLE`](@ref)
fit (`k = nparams(fitted)`). Lower is better; an estimate of out-of-sample
predictive performance. Defined for maximum-likelihood fits only — use
[`waic`](@ref) or [`loo`](@ref) for [`Bayesian`](@ref) fits.
"""
function aic(fitted::FittedComparativeModel{M, I}, data) where {M, I <: _MLEMethods}
    return _ic(_data_loglik(fitted, data), nparams(fitted), nobs(data), :AIC)
end

"""
    bic(fitted, data)

Bayesian information criterion `-2·ℓ̂ + k·log n` of an
[`MLE`](@ref)/[`StepwiseMLE`](@ref) fit (`k = nparams(fitted)`, `n = nobs(data)`
comparisons). Lower is better. Defined for maximum-likelihood fits only — use
[`waic`](@ref) or [`loo`](@ref) for [`Bayesian`](@ref) fits.
"""
function bic(fitted::FittedComparativeModel{M, I}, data) where {M, I <: _MLEMethods}
    return _ic(_data_loglik(fitted, data), nparams(fitted), nobs(data), :BIC)
end

# Bayesian fits: redirect to the predictive criteria.
aic(::FittedComparativeModel{M, Bayesian}, args...) where {M} = throw(ArgumentError(
    "AIC is defined for maximum-likelihood fits; use `waic` or `loo` for a Bayesian fit."))
bic(::FittedComparativeModel{M, Bayesian}, args...) where {M} = throw(ArgumentError(
    "BIC is defined for maximum-likelihood fits; use `waic` or `loo` for a Bayesian fit."))
