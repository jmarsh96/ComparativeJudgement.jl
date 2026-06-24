# ─── Widely Applicable Information Criterion (WAIC) ───────────────────────────
#
# Estimates out-of-sample predictive accuracy from the pointwise log-likelihood
# of a Bayesian fit (Watanabe 2010; Vehtari, Gelman & Gabry 2017). Uses the
# `p_waic` (variance) form: the effective number of parameters is the sum over
# observations of the posterior variance of the log density.

"""
    WAICResult

Result of [`waic`](@ref): the expected log pointwise predictive density
`elpd_waic`, the effective number of parameters `p_waic`, `waic = -2·elpd_waic`
(lower is better), a standard error `se`, and the per-observation `pointwise`
elpd contributions.
"""
struct WAICResult
    elpd_waic::Float64
    p_waic::Float64
    waic::Float64
    se::Float64
    pointwise::Vector{Float64}
end

function Base.show(io::IO, r::WAICResult)
    println(io, "WAICResult")
    println(io, "  elpd_waic = ", round(r.elpd_waic, digits=2), " ± ", round(r.se, digits=2))
    println(io, "  p_waic    = ", round(r.p_waic, digits=2))
    print(io,   "  waic      = ", round(r.waic, digits=2))
end

"""
    waic(fitted, data)

Widely Applicable Information Criterion for a [`Bayesian`](@ref) `fitted` model
on its `data`. Returns a [`WAICResult`](@ref). Errors for [`MLE`](@ref) fits
(use [`aic`](@ref)/[`bic`](@ref)). Use [`waic`](@ref)/[`loo`](@ref) to compare
Bayesian models on the same data — lower `waic` is better.
"""
function waic(fitted::FittedComparativeModel, data)
    ll = pointwise_loglikelihood(fitted, data)
    ll isa AbstractMatrix || throw(ArgumentError(
        "WAIC requires a Bayesian fit with posterior draws; got a point estimate. " *
        "Use `aic`/`bic` for an MLE fit."))
    S, n = size(ll)
    elpd_i = Vector{Float64}(undef, n)
    p_i = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        col = view(ll, :, i)
        lpd = _logsumexp(col) - log(S)      # log pointwise predictive density
        p_i[i] = var(col)                   # p_waic contribution (sample variance)
        elpd_i[i] = lpd - p_i[i]
    end
    elpd = sum(elpd_i)
    se = sqrt(n * var(elpd_i))
    return WAICResult(elpd, sum(p_i), -2.0 * elpd, se, elpd_i)
end
