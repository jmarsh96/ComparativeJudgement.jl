# ─── Thurstone Case V: P(i beats j) = Φ(λᵢ − λⱼ) ─────────────────────────────
#
# The equal-variance, uncorrelated discriminal-process model. It differs from
# Bradley–Terry only in the link — probit (Φ) instead of logit — so the latent
# strengths λ enter directly (no log reparametrisation). The MLE reuses the same
# L-BFGS pattern with the probit log-likelihood; the Bayesian fit uses Albert–Chib
# truncated-normal data augmentation instead of Pólya-Gamma.

_tcv_check(model::ThurstoneCaseV) = model.distribution === :normal ||
    throw(ArgumentError("Only the :normal Thurstone Case V is implemented, got :$(model.distribution)"))

# ─── Maximum likelihood ──────────────────────────────────────────────────────

# Negative probit log-likelihood. λ₁ is fixed at 0 (λ_free holds λ₂..λ_K) for
# identifiability; strengths() returns the centred estimates.
function _tcv_neg_loglik(λ_free::AbstractVector, wins::Matrix{Int})
    λ = _full_theta(λ_free)
    n = length(λ)
    ll = zero(eltype(λ))
    @inbounds for i in 1:n, j in 1:n
        i == j && continue
        w = wins[i, j]
        iszero(w) && continue
        ll += w * _log_normcdf(λ[i] - λ[j])
    end
    return -ll
end

function _tcv_neg_grad!(G::AbstractVector, λ_free::AbstractVector, wins::Matrix{Int})
    λ = _full_theta(λ_free)
    n = length(λ)
    grad = zeros(n)              # full gradient incl. the pinned item 1
    @inbounds for i in 1:n, j in 1:n
        i == j && continue
        w = wins[i, j]
        iszero(w) && continue
        h = w * _inv_mills(λ[i] - λ[j])   # ∂ logΦ(λᵢ−λⱼ)/∂(λᵢ−λⱼ) = φ/Φ
        grad[i] += h
        grad[j] -= h
    end
    @inbounds for k in 2:n
        G[k - 1] = -grad[k]
    end
    return G
end

# Aggregated probit log-likelihood (per-pair binomial form), for the Bayesian recorder.
function _tcv_loglik(λ::Vector{Float64}, agg::_AggregatedPairData)
    ll = 0.0
    @inbounds for p in 1:agg.P
        i, j = agg.pairs[p]
        d = λ[i] - λ[j]
        y = agg.κ[p] + agg.Nvec[p] / 2       # wins of i in this pair
        ll += y * _log_normcdf(d) + (agg.Nvec[p] - y) * _log_normcdf(-d)
    end
    return ll
end

"""
    fit(model::ThurstoneCaseV, method::MLE, data::PairwiseData)

Maximum-likelihood fit of the Thurstone Case V model via L-BFGS on the probit
log-likelihood. The first item's strength is fixed at zero during optimisation
for identifiability; [`strengths`](@ref) returns the centred estimates.
"""
function fit(model::ThurstoneCaseV, method::MLE, data::PairwiseData{L}) where {L}
    _tcv_check(model)
    wins = data.wins
    n = length(data.labels)
    n >= 2 || throw(ArgumentError("Need at least 2 items to fit ThurstoneCaseV, got $n"))
    f(λ_free) = _tcv_neg_loglik(λ_free, wins)
    g!(G, λ_free) = _tcv_neg_grad!(G, λ_free, wins)
    result = optimize(f, g!, zeros(n - 1), LBFGS())
    return FittedComparativeModel(
        model, method, result, data.labels,
        Optim.converged(result), Optim.iterations(result),
    )
end

function fit(model::ThurstoneCaseV, method::MLE, wins::Matrix{Int}, labels::Vector{L}) where {L}
    return fit(model, method, PairwiseData(wins, labels))
end

function fit(model::ThurstoneCaseV, data::PairwiseData)
    return fit(model, MLE(), data)
end

function fit(model::ThurstoneCaseV, wins::Matrix{Int}, labels::Vector{L}) where {L}
    return fit(model, MLE(), PairwiseData(wins, labels))
end

function loglikelihood(fitted::FittedComparativeModel{ThurstoneCaseV, MLE})
    return -Optim.minimum(fitted.result)
end

function strengths(fitted::FittedComparativeModel{ThurstoneCaseV, MLE})
    λ = _full_theta(Optim.minimizer(fitted.result))
    return λ .- mean(λ)
end

function probability(fitted::FittedComparativeModel{ThurstoneCaseV, MLE}, i::Integer, j::Integer)
    λ = _full_theta(Optim.minimizer(fitted.result))
    return _normcdf(λ[i] - λ[j])
end

function probability(fitted::FittedComparativeModel{ThurstoneCaseV, MLE, R, L},
                     item_i::L, item_j::L) where {R, L}
    idx_i = findfirst(==(item_i), fitted.labels)
    idx_j = findfirst(==(item_j), fitted.labels)
    idx_i === nothing && throw(ArgumentError("Label $(item_i) not found in fitted model"))
    idx_j === nothing && throw(ArgumentError("Label $(item_j) not found in fitted model"))
    return probability(fitted, idx_i, idx_j)
end

# ─── Bayesian: Albert–Chib truncated-normal augmented Gibbs ───────────────────

"""
    fit(model::ThurstoneCaseV, method::Bayesian, data::PairwiseData,
        [prior::NormalPrior]; rng=Random.default_rng())

Bayesian fit of the Thurstone Case V model by Albert–Chib augmented Gibbs
sampling: each comparison gets a latent normal utility truncated by its outcome,
rendering λ conditionally Gaussian. `prior` is a `K`-variate
[`NormalPrior`](@ref) on the latent strengths (default `NormalPrior(K)`). The
result holds posterior draws ([`BTMCMCSamples`](@ref)); query them with
[`posterior_mean`](@ref), [`posterior_std`](@ref), [`credible_interval`](@ref)
and [`probability`](@ref).
"""
function fit(model::ThurstoneCaseV, method::Bayesian, data::PairwiseData{L},
             prior::NormalPrior; rng::AbstractRNG=Random.default_rng()) where {L}
    _tcv_check(model)
    K = length(data.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit ThurstoneCaseV, got $K"))
    length(prior.μ) == K || throw(DimensionMismatch(
        "prior.μ has length $(length(prior.μ)), expected $K"))

    wins = data.wins
    agg = _aggregate_pairs(wins, K)

    # Under the probit augmentation each latent utility has unit variance, so the
    # λ-precision A = Σ⁻¹ + XᵀNX is *constant* — factor it once. (XᵀNX[i,i] += N,
    # [j,j] += N, [i,j] -= N for each pair.)
    Σ_inv   = inv(prior.Σ)
    Σ_inv_μ = Σ_inv * prior.μ
    A = Matrix(Σ_inv)
    @inbounds for p in 1:agg.P
        i, j = agg.pairs[p]
        N = agg.Nvec[p]
        A[i, i] += N; A[j, j] += N
        A[i, j] -= N; A[j, i] -= N
    end
    @inbounds for k in 1:K; A[k, k] += 1e-10; end
    C = cholesky!(Symmetric(A))

    total = method.n_samples + method.n_burnin
    samples        = Matrix{Float64}(undef, method.n_samples, K)
    loglikelihoods = Vector{Float64}(undef, method.n_samples)

    λ = zeros(K)
    h = Vector{Float64}(undef, K)
    m = Vector{Float64}(undef, K)
    z = Vector{Float64}(undef, K)

    for s in 1:total
        # Latent utilities u | λ, summed per pair into h = Σ⁻¹μ₀ + Xᵀu.
        h .= Σ_inv_μ
        @inbounds for p in 1:agg.P
            i, j = agg.pairs[p]
            μ = λ[i] - λ[j]
            N = agg.Nvec[p]
            y = wins[i, j]              # i-wins: utility > 0; j-wins: utility < 0
            S = 0.0
            for _ in 1:y
                S += _sample_truncated_normal(rng, μ, true)
            end
            for _ in 1:(N - y)
                S += _sample_truncated_normal(rng, μ, false)
            end
            h[i] += S; h[j] -= S
        end

        # λ ~ N(A⁻¹h, A⁻¹)
        m .= h
        ldiv!(C, m)
        randn!(rng, z)
        ldiv!(C.U, z)
        λ .= m .+ z
        method.center && (λ .-= mean(λ))

        if s > method.n_burnin
            idx = s - method.n_burnin
            samples[idx, :]    .= λ
            loglikelihoods[idx] = _tcv_loglik(λ, agg)
        end
    end

    result = BTMCMCSamples(samples, loglikelihoods, method.n_samples, method.n_burnin)
    return FittedComparativeModel(model, method, result, data.labels, true, total)
end

function fit(model::ThurstoneCaseV, method::Bayesian, data::PairwiseData{L};
             rng::AbstractRNG=Random.default_rng()) where {L}
    return fit(model, method, data, NormalPrior(length(data.labels)); rng=rng)
end

function fit(model::ThurstoneCaseV, method::Bayesian, wins::Matrix{Int},
             labels::Vector{L}, prior::NormalPrior;
             rng::AbstractRNG=Random.default_rng()) where {L}
    return fit(model, method, PairwiseData(wins, labels), prior; rng=rng)
end

function posterior_mean(fitted::FittedComparativeModel{ThurstoneCaseV, Bayesian})
    return vec(mean(fitted.result.samples, dims=1))
end

function posterior_std(fitted::FittedComparativeModel{ThurstoneCaseV, Bayesian})
    return vec(std(fitted.result.samples, dims=1))
end

function credible_interval(fitted::FittedComparativeModel{ThurstoneCaseV, Bayesian},
                            k::Integer; prob::Float64=0.95)
    α = (1.0 - prob) / 2.0
    col = fitted.result.samples[:, k]
    return (quantile(col, α), quantile(col, 1.0 - α))
end

function loglikelihood(fitted::FittedComparativeModel{ThurstoneCaseV, Bayesian})
    return fitted.result.loglikelihoods
end

function strengths(fitted::FittedComparativeModel{ThurstoneCaseV, Bayesian})
    return posterior_mean(fitted)
end

function probability(fitted::FittedComparativeModel{ThurstoneCaseV, Bayesian},
                     i::Integer, j::Integer)
    S = fitted.result.samples
    return mean(_normcdf.(S[:, i] .- S[:, j]))
end

function probability(fitted::FittedComparativeModel{ThurstoneCaseV, Bayesian, R, L},
                     item_i::L, item_j::L) where {R, L}
    idx_i = findfirst(==(item_i), fitted.labels)
    idx_j = findfirst(==(item_j), fitted.labels)
    idx_i === nothing && throw(ArgumentError("Label $(item_i) not found in fitted model"))
    idx_j === nothing && throw(ArgumentError("Label $(item_j) not found in fitted model"))
    return probability(fitted, idx_i, idx_j)
end
