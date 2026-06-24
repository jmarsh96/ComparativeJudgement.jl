# ─── Likelihood-ratio test for nested covariate models ───────────────────────
#
# The covariate models are the only nested family here: dropping covariates from
# the linear predictor `λ_i = z_iᵀβ` gives a sub-model. Twice the difference in
# maximised log-likelihoods is compared to a χ² distribution with degrees of
# freedom equal to the number of dropped covariates. (The link models and the
# structural extensions are not nested, so they are compared with AIC/BIC or the
# predictive criteria instead.)

"""
    LRTResult

Result of [`lrtest`](@ref): the likelihood-ratio `statistic` `2(ℓ_full −
ℓ_restricted)`, its `df` (covariates dropped), the χ² `pvalue`, and the two
maximised log-likelihoods.
"""
struct LRTResult
    statistic::Float64
    df::Int
    pvalue::Float64
    loglik_restricted::Float64
    loglik_full::Float64
end

function Base.show(io::IO, r::LRTResult)
    println(io, "LRTResult (likelihood-ratio test)")
    println(io, "  statistic = ", round(r.statistic, digits=3), "  on ", r.df, " df")
    print(io,   "  p-value   = ", round(r.pvalue, digits=4))
end

"""
    lrtest(restricted, full)

Likelihood-ratio test of a `restricted` covariate [`MLE`](@ref) fit nested in a
`full` one (the restricted model's selected covariates must be a subset of the
full model's, both fit to the same items). Returns an [`LRTResult`](@ref); a
small p-value rejects the restricted model in favour of the full one.
"""
function lrtest(restricted::FittedComparativeModel{<:Covariates, <:Any, CovariateMLEResult},
                full::FittedComparativeModel{<:Covariates, <:Any, CovariateMLEResult})
    rsel = Set(restricted.result.selected)
    fsel = Set(full.result.selected)
    issubset(rsel, fsel) || throw(ArgumentError(
        "models are not nested: the restricted model's covariates must be a subset " *
        "of the full model's (restricted=$(sort(collect(rsel))), full=$(sort(collect(fsel))))"))
    restricted.labels == full.labels || throw(ArgumentError(
        "the two fits must be on the same items"))
    df = length(fsel) - length(rsel)
    df >= 1 || throw(ArgumentError(
        "the models have the same number of covariates; there is nothing to test"))
    ℓr = restricted.result.loglik
    ℓf = full.result.loglik
    stat = max(2.0 * (ℓf - ℓr), 0.0)
    return LRTResult(stat, df, _chisq_sf(stat, df), ℓr, ℓf)
end

lrtest(restricted::FittedComparativeModel, full::FittedComparativeModel) = throw(ArgumentError(
    "lrtest is defined for two nested covariate MLE fits; the other models are not " *
    "nested — compare them with `aic`/`bic`, `waic`/`loo`, or `crossvalidate`."))
