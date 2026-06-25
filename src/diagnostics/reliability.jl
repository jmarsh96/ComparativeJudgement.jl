# ─── Reliability-based measures: SSR and split-half reliability ──────────────
#
# Two single-model diagnostics of how reliably the scale is estimated. The scale
# separation reliability (SSR) is the house metric of the CJ literature; the
# split-half reliability resamples the comparisons, refits the *same* model to
# two random halves, and correlates the estimates. Both are single-model
# estimate-stability checks — contrast with the cross-model `rank_correlation`.

# SSR = (σ²λ − mean SE²) / σ²λ : the share of the observed strength variance not
# attributable to estimation error.
function _ssr(λ::AbstractVector, se::AbstractVector)
    σ²λ = var(λ)
    σ²λ == 0.0 && return NaN
    return (σ²λ - mean(abs2, se)) / σ²λ
end

# Per-item standard errors of the latent strengths, by fit type:
#   Bayesian        → posterior standard deviation,
#   covariate MLE   → delta-method SE from λ = Zβ and the coefficient covariance,
#   plain BT/TCV MLE→ observed-information SE (from `_strength_vcov`, statsapi.jl).
_lambda_se(f::FittedComparativeModel{M, Bayesian}) where {M} = posterior_std(f)

function _lambda_se(f::FittedComparativeModel{M, I, CovariateMLEResult}) where {M <: Covariates, I}
    r = f.result
    isempty(r.selected) && return zeros(size(r.Z, 1))
    Zsel = r.Z[:, r.selected]
    V = Zsel * r.vcov * Zsel'
    return sqrt.(max.(diag(V), 0.0))
end

# Unlike vcov/stderror/confint (where a missing covariance is an error), SSR is a
# descriptive summary, so a singular design degrades to NaN with a warning rather
# than throwing — keeping reliability pipelines running.
function _lambda_se(f::FittedComparativeModel{<:Union{BradleyTerry, ThurstoneCaseV}, MLE})
    try
        return sqrt.(max.(diag(_strength_vcov(f)), 0.0))
    catch err
        err isa SingularInformationError || rethrow()
        @warn "ssr: observed-information standard errors are unavailable for this " *
              "MLE fit (singular design); returning NaN. Use a Bayesian fit for SSR." exception = err
        return fill(NaN, length(f.labels))
    end
end

"""
    ssr(fitted)

Scale separation reliability `(σ²λ − meanSE²)/σ²λ`: the proportion of the
observed variance in the estimated strengths attributable to true differences
between items rather than estimation error. Higher is better (the CJ literature
treats ≥ 0.7 as adequate).

The per-item standard errors come from the posterior for a [`Bayesian`](@ref)
fit, the coefficient covariance for a covariate [`MLE`](@ref) fit, and the
observed information for a plain Bradley–Terry/Thurstone `MLE` fit.

SSR is a descriptive summary, not a model-selection criterion: it is inflated by
adaptive pair selection and is in part an artefact of the design.
"""
ssr(f::FittedComparativeModel) = _ssr(strengths(f), _lambda_se(f))

# ─── Split-half reliability ──────────────────────────────────────────────────

"""
    ReliabilityResult

Result of [`split_half_reliability`](@ref): the `mean` split-half correlation
across the random halvings, its `std`, the Spearman–Brown stepped-up estimate of
full-data reliability `spearman_brown = 2r̄/(1+r̄)`, the `per_split` correlations,
and the number of splits `n_splits`.
"""
struct ReliabilityResult
    mean::Float64
    std::Float64
    spearman_brown::Float64
    per_split::Vector{Float64}
    n_splits::Int
end

function Base.show(io::IO, r::ReliabilityResult)
    println(io, "ReliabilityResult (split-half, ", r.n_splits, " splits)")
    println(io, "  mean correlation = ", round(r.mean, digits=3), " ± ", round(r.std, digits=3))
    print(io,   "  Spearman–Brown   = ", round(r.spearman_brown, digits=3))
end

# Refit the same model/method to a data half (Bayesian threads the rng / prior).
_refit(model, method::Bayesian, data, prior; rng) =
    prior === nothing ? fit(model, method, data; rng=rng) : fit(model, method, data, prior; rng=rng)
_refit(model, method, data, prior; rng) = fit(model, method, data)

"""
    split_half_reliability(model, method, data; n_splits=100, rng, prior=nothing)

Estimate the reliability of `model`'s strength estimates by repeatedly splitting
the comparisons into two random halves, fitting the same `model`/`method` to
each, and correlating (Spearman) the two strength vectors. Returns a
[`ReliabilityResult`](@ref) with the mean correlation over `n_splits` halvings
and its Spearman–Brown step-up; the CJ literature treats ≥ 0.7 as good.

`prior` (for [`Bayesian`](@ref) fits) is forwarded to [`fit`](@ref); pass dense
enough data that every item is compared within each half.
"""
function split_half_reliability(model, method, data; n_splits::Int=100,
                                rng::AbstractRNG=Random.default_rng(), prior=nothing)
    n_splits >= 1 || throw(ArgumentError("n_splits must be at least 1, got $n_splits"))
    cors = Vector{Float64}(undef, n_splits)
    for s in 1:n_splits
        half1, half2 = train_test_split(data; frac=0.5, rng=rng)
        f1 = _refit(model, method, half1, prior; rng=rng)
        f2 = _refit(model, method, half2, prior; rng=rng)
        cors[s] = _corspearman(strengths(f1), strengths(f2))
    end
    m = mean(cors)
    return ReliabilityResult(m, std(cors), 2m / (1 + m), cors, n_splits)
end
