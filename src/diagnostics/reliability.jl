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
#   plain BT/TCV MLE→ observed-information SE (finite-difference Hessian).
_lambda_se(f::FittedComparativeModel{M, Bayesian}) where {M} = posterior_std(f)

function _lambda_se(f::FittedComparativeModel{M, I, CovariateMLEResult}) where {M <: Covariates, I}
    r = f.result
    isempty(r.selected) && return zeros(size(r.Z, 1))
    Zsel = r.Z[:, r.selected]
    V = Zsel * r.vcov * Zsel'
    return sqrt.(max.(diag(V), 0.0))
end

# Central-difference Hessian of `f` at `x` (small dimension; reuses the fit's
# own negative log-likelihood objective).
function _hessian_fd(f, x::Vector{Float64}; h::Float64=1e-4)
    n = length(x)
    H = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in i:n
        xpp = copy(x); xpp[i] += h; xpp[j] += h
        xpm = copy(x); xpm[i] += h; xpm[j] -= h
        xmp = copy(x); xmp[i] -= h; xmp[j] += h
        xmm = copy(x); xmm[i] -= h; xmm[j] -= h
        H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h^2)
        H[j, i] = H[i, j]
    end
    return H
end

# Observed-information SE of the centred strengths for a plain BT/TCV MLE fit.
function _plain_lambda_se(f::FittedComparativeModel{M, MLE}, data::PairwiseData) where {M}
    K = length(f.labels)
    θ̂ = Optim.minimizer(f.result)                       # free params (item 1 fixed at 0)
    negll = f.model isa BradleyTerry ? (x -> _bt_neg_loglik(x, data.wins)) :
                                       (x -> _tcv_neg_loglik(x, data.wins))
    H = _hessian_fd(negll, collect(float.(θ̂)))
    Σfree = inv(Symmetric(H))                            # covariance of the free strengths
    Σfull = zeros(K, K)
    Σfull[2:K, 2:K] .= Σfree
    C = Matrix{Float64}(I, K, K) .- fill(1.0 / K, K, K)  # centring projection
    Σc = C * Σfull * C'
    return sqrt.(max.(diag(Σc), 0.0))
end

"""
    ssr(fitted)
    ssr(fitted, data)

Scale separation reliability `(σ²λ − meanSE²)/σ²λ`: the proportion of the
observed variance in the estimated strengths attributable to true differences
between items rather than estimation error. Higher is better (the CJ literature
treats ≥ 0.7 as adequate).

The standard errors come from the posterior for a [`Bayesian`](@ref) fit and
from the coefficient covariance for a covariate [`MLE`](@ref) fit (no `data`
needed); for a plain Bradley–Terry/Thurstone `MLE` fit pass the `data` so the
observed-information standard errors can be computed.

SSR is a descriptive summary, not a model-selection criterion: it is inflated by
adaptive pair selection and is in part an artefact of the design.
"""
ssr(f::FittedComparativeModel{M, Bayesian}) where {M} = _ssr(strengths(f), _lambda_se(f))
ssr(f::FittedComparativeModel{M, I, CovariateMLEResult}) where {M <: Covariates, I} =
    _ssr(strengths(f), _lambda_se(f))
ssr(f::FittedComparativeModel{M, MLE}, data::PairwiseData) where {M <: Union{BradleyTerry, ThurstoneCaseV}} =
    _ssr(strengths(f), _plain_lambda_se(f, data))

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
