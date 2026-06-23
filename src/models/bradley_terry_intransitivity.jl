# ─── Bradley-Terry with intransitivity: logit P(i≻j) = (λᵢ − λⱼ) + γᵢⱼ ───────
#
# A skew-symmetric per-pair term γᵢⱼ = −γⱼᵢ is added to the Bradley-Terry linear
# predictor, absorbing the component of the preference that a single transitive
# scale cannot explain. With one γ per observed pair the model is saturated, so
# γ must be regularised: the MLE adds a ridge penalty (1/2σ²γ)·Σγ², and the
# Bayesian fit puts γᵢⱼ ~ N(0, σ²γ) with an inverse-gamma hyperprior on σ²γ that
# learns the overall amount of intransitivity. The ridge is what identifies λ —
# it becomes the best transitive fit and γ the intransitive residual.
#
# Both reuse the plain Bradley-Terry Pólya-Gamma machinery in bradley_terry.jl
# (`_aggregate_pairs`, `log1pexp`); the comparison data is an ordinary
# `PairwiseData`, since γ is a per-pair (not per-rater) effect.

# Data log p(wins | λ, γ) (unpenalised), using the aggregated pair representation.
function _intransitive_data_loglik(λ::AbstractVector, γ::AbstractVector,
                                   agg::_AggregatedPairData)
    ll = 0.0
    @inbounds for p in 1:agg.P
        i, j = agg.pairs[p]
        η = λ[i] - λ[j] + γ[p]
        ll += (agg.κ[p] + agg.Nvec[p] / 2) * η - agg.Nvec[p] * log1pexp(η)
    end
    return ll
end

# ─── Maximum likelihood (ridge-penalised) ────────────────────────────────────
#
# Parameter vector θ = [λ₂,…,λ_K (item 1 pinned at 0); γ₁,…,γ_P]; the predictor
# for pair p = (i, j) is η_p = λᵢ − λⱼ + γ_p.

@inline _itr_lambda(θ, i) = i == 1 ? 0.0 : θ[i - 1]

function _intransitive_neg_loglik(θ::AbstractVector, agg::_AggregatedPairData, σ²γ::Float64)
    K = agg.K
    ll = 0.0
    @inbounds for p in 1:agg.P
        i, j = agg.pairs[p]
        γp = θ[(K - 1) + p]
        η = _itr_lambda(θ, i) - _itr_lambda(θ, j) + γp
        ll += (agg.κ[p] + agg.Nvec[p] / 2) * η - agg.Nvec[p] * log1pexp(η)
    end
    pen = 0.0
    @inbounds for p in 1:agg.P
        pen += θ[(K - 1) + p]^2
    end
    return -ll + pen / (2.0 * σ²γ)
end

function _intransitive_neg_grad!(G::AbstractVector, θ::AbstractVector,
                                 agg::_AggregatedPairData, σ²γ::Float64)
    K = agg.K
    fill!(G, 0.0)
    @inbounds for p in 1:agg.P
        i, j = agg.pairs[p]
        γp = θ[(K - 1) + p]
        η = _itr_lambda(θ, i) - _itr_lambda(θ, j) + γp
        w = agg.κ[p] + agg.Nvec[p] / 2
        r = w - agg.Nvec[p] / (1.0 + exp(-η))    # residual: observed − expected wins of i
        i != 1 && (G[i - 1] -= r)
        j != 1 && (G[j - 1] += r)
        G[(K - 1) + p] += -r + γp / σ²γ
    end
    return G
end

"""
    fit(model::Intransitive{BradleyTerry}, method::MLE, data::PairwiseData; σ²γ=1.0)

Ridge-penalised maximum-likelihood fit of the intransitive Bradley–Terry model
via L-BFGS, maximising `ℓ(λ, γ) − (1/2σ²γ)·Σγᵢⱼ²`. The penalty `σ²γ` (default
`1.0`) identifies the latent scale: λ becomes the best transitive fit and γ the
intransitive residual. Query with [`strengths`](@ref), [`intransitivity`](@ref)
and [`probability`](@ref).
"""
function fit(model::Intransitive{BradleyTerry}, method::MLE, data::PairwiseData{L};
             σ²γ::Real=1.0) where {L}
    K = length(data.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit intransitive BradleyTerry, got $K"))
    σ²γ > 0 || throw(ArgumentError("σ²γ must be positive, got $σ²γ"))
    agg = _aggregate_pairs(data.wins, K)
    agg.P >= 1 || throw(ArgumentError("No pairs with comparison data to fit"))
    s²γ = Float64(σ²γ)
    f(θ) = _intransitive_neg_loglik(θ, agg, s²γ)
    g!(G, θ) = _intransitive_neg_grad!(G, θ, agg, s²γ)
    res = optimize(f, g!, zeros((K - 1) + agg.P), LBFGS())
    θ = Optim.minimizer(res)
    λ = vcat(0.0, θ[1:(K - 1)])
    λ .-= mean(λ)
    γ = θ[K:((K - 1) + agg.P)]
    ll = _intransitive_data_loglik(λ, γ, agg)
    result = IntransitiveMLEResult(λ, agg.pairs, γ, s²γ, ll)
    return FittedComparativeModel(model, method, result, data.labels,
                                  Optim.converged(res), Optim.iterations(res))
end

function fit(model::Intransitive{BradleyTerry}, data::PairwiseData)
    return fit(model, MLE(), data)
end

# ─── Bayesian: Pólya-Gamma Gibbs with conjugate γ and σ²γ updates ────────────

"""
    fit(model::Intransitive{BradleyTerry}, method::Bayesian, data::PairwiseData,
        [prior::IntransitivityPrior]; rng=Random.default_rng())

Bayesian fit of the intransitive Bradley–Terry model by Pólya-Gamma augmented
Gibbs sampling. Each sweep draws the augmentation `ω`, the strengths `λ` (a BT
update with the γ offset folded into the working response), the independent
skew-symmetric terms `γᵢⱼ ~ N(0, σ²γ)`, and the variance `σ²γ` from its
inverse-gamma full conditional (see [`IntransitivityPrior`](@ref)). The result
holds posterior draws ([`IntransitiveMCMCSamples`](@ref)); query with
[`posterior_mean`](@ref), [`credible_interval`](@ref), [`intransitivity`](@ref)
and [`probability`](@ref).
"""
function fit(model::Intransitive{BradleyTerry}, method::Bayesian, data::PairwiseData{L},
             prior::IntransitivityPrior; rng::AbstractRNG=Random.default_rng()) where {L}
    K = length(data.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit intransitive BradleyTerry, got $K"))
    agg = _aggregate_pairs(data.wins, K)
    P = agg.P
    P >= 1 || throw(ArgumentError("No pairs with comparison data to fit"))

    λ_prior = prior.λ_prior === nothing ? NormalPrior(K) : prior.λ_prior
    length(λ_prior.μ) == K || throw(DimensionMismatch(
        "λ_prior.μ has length $(length(λ_prior.μ)), expected $K"))
    α0 = prior.σ²γ_prior.α
    β0 = prior.σ²γ_prior.β

    Σ_inv   = inv(λ_prior.Σ)
    Σ_inv_μ = Σ_inv * λ_prior.μ

    total = method.n_burnin + method.thin * method.n_samples
    λ_samples   = Matrix{Float64}(undef, method.n_samples, K)
    γ_samples   = Matrix{Float64}(undef, method.n_samples, P)
    σ²γ_samples = Vector{Float64}(undef, method.n_samples)
    lls         = Vector{Float64}(undef, method.n_samples)

    λ   = zeros(K)
    γ   = zeros(P)
    ω   = Vector{Float64}(undef, P)
    σ²γ = 1.0
    V   = Matrix{Float64}(undef, K, K)
    m   = Vector{Float64}(undef, K)
    z   = Vector{Float64}(undef, K)

    for s in 1:total
        # ω | λ, γ
        @inbounds for p in 1:P
            i, j = agg.pairs[p]
            ω[p] = _sample_pg(rng, agg.Nvec[p], λ[i] - λ[j] + γ[p])
        end

        # λ | ω, γ : BT Gaussian update with adjusted κ′ = κ − ω·γ
        copyto!(V, Σ_inv)
        m .= Σ_inv_μ
        @inbounds for p in 1:P
            i, j = agg.pairs[p]
            op = ω[p]
            V[i, i] += op; V[j, j] += op
            V[i, j] -= op; V[j, i] -= op
            κ′ = agg.κ[p] - op * γ[p]
            m[i] += κ′; m[j] -= κ′
        end
        @inbounds for k in 1:K; V[k, k] += 1e-10; end
        C = cholesky!(Symmetric(V))
        ldiv!(C, m)
        randn!(rng, z); ldiv!(C.U, z)
        λ .= m .+ z
        method.center && (λ .-= mean(λ))

        # γ_p | ω_p, λ : independent Normals (γ enters only pair p)
        @inbounds for p in 1:P
            i, j = agg.pairs[p]
            prec = 1.0 / σ²γ + ω[p]
            mean_p = (agg.κ[p] - ω[p] * (λ[i] - λ[j])) / prec
            γ[p] = mean_p + randn(rng) / sqrt(prec)
        end

        # σ²γ | γ : inverse-gamma full conditional
        ss = 0.0
        @inbounds for p in 1:P; ss += γ[p]^2; end
        σ²γ = _sample_inv_gamma(rng, α0 + P / 2.0, β0 + ss / 2.0)

        if s > method.n_burnin && (s - method.n_burnin) % method.thin == 0
            idx = (s - method.n_burnin) ÷ method.thin
            λ_samples[idx, :]  .= λ
            γ_samples[idx, :]  .= γ
            σ²γ_samples[idx]    = σ²γ
            lls[idx]            = _intransitive_data_loglik(λ, γ, agg)
        end
    end

    result = IntransitiveMCMCSamples(λ_samples, γ_samples, agg.pairs, σ²γ_samples,
                                     lls, method.n_samples, method.n_burnin, method.thin)
    return FittedComparativeModel(model, method, result, data.labels, true, total)
end

function fit(model::Intransitive{BradleyTerry}, method::Bayesian, data::PairwiseData{L};
             rng::AbstractRNG=Random.default_rng()) where {L}
    return fit(model, method, data, IntransitivityPrior(); rng=rng)
end

# ─── Accessors (MLE; dispatch on IntransitiveMLEResult) ──────────────────────

function strengths(fitted::FittedComparativeModel{<:Intransitive, MLE, <:IntransitiveMLEResult})
    return copy(fitted.result.λ)
end

function loglikelihood(fitted::FittedComparativeModel{<:Intransitive, MLE, <:IntransitiveMLEResult})
    return fitted.result.loglik
end

# γᵢⱼ for an ordered (i, j): +γ if (i,j) is a stored pair, −γ for (j,i), else 0.
function _gamma_lookup(pairs::Vector{Tuple{Int,Int}}, γ::AbstractVector, i::Integer, j::Integer)
    @inbounds for p in 1:length(pairs)
        a, b = pairs[p]
        a == i && b == j && return γ[p]
        a == j && b == i && return -γ[p]
    end
    return 0.0
end

function _skew_matrix(pairs::Vector{Tuple{Int,Int}}, γ::AbstractVector, K::Int)
    Γ = zeros(K, K)
    @inbounds for p in 1:length(pairs)
        i, j = pairs[p]
        Γ[i, j] = γ[p]
        Γ[j, i] = -γ[p]
    end
    return Γ
end

function intransitivity(fitted::FittedComparativeModel{<:Intransitive, MLE, <:IntransitiveMLEResult})
    r = fitted.result
    return _skew_matrix(r.pairs, r.γ, length(fitted.labels))
end

function probability(fitted::FittedComparativeModel{<:Intransitive, MLE, <:IntransitiveMLEResult},
                     i::Integer, j::Integer)
    r = fitted.result
    d = r.λ[i] - r.λ[j] + _gamma_lookup(r.pairs, r.γ, i, j)
    return 1.0 / (1.0 + exp(-d))
end

# ─── Accessors (Bayesian; dispatch on IntransitiveMCMCSamples) ───────────────

function posterior_mean(fitted::FittedComparativeModel{<:Intransitive, Bayesian, IntransitiveMCMCSamples})
    return vec(mean(fitted.result.λ_samples, dims=1))
end

function posterior_std(fitted::FittedComparativeModel{<:Intransitive, Bayesian, IntransitiveMCMCSamples})
    return vec(std(fitted.result.λ_samples, dims=1))
end

function credible_interval(fitted::FittedComparativeModel{<:Intransitive, Bayesian, IntransitiveMCMCSamples},
                           k::Integer; prob::Float64=0.95)
    α = (1.0 - prob) / 2.0
    col = fitted.result.λ_samples[:, k]
    return (quantile(col, α), quantile(col, 1.0 - α))
end

function strengths(fitted::FittedComparativeModel{<:Intransitive, Bayesian, IntransitiveMCMCSamples})
    return posterior_mean(fitted)
end

function loglikelihood(fitted::FittedComparativeModel{<:Intransitive, Bayesian, IntransitiveMCMCSamples})
    return fitted.result.loglikelihoods
end

function intransitivity(fitted::FittedComparativeModel{<:Intransitive, Bayesian, IntransitiveMCMCSamples})
    r = fitted.result
    γ̄ = vec(mean(r.γ_samples, dims=1))
    return _skew_matrix(r.pairs, γ̄, length(fitted.labels))
end

function probability(fitted::FittedComparativeModel{<:Intransitive, Bayesian, IntransitiveMCMCSamples},
                     i::Integer, j::Integer)
    r = fitted.result
    Λ = r.λ_samples
    d = Λ[:, i] .- Λ[:, j]
    @inbounds for p in 1:length(r.pairs)
        a, b = r.pairs[p]
        if a == i && b == j
            d .+= @view r.γ_samples[:, p]; break
        elseif a == j && b == i
            d .-= @view r.γ_samples[:, p]; break
        end
    end
    return mean(1.0 ./ (1.0 .+ exp.(-d)))
end

# ─── Label-based probability (shared across MLE and Bayesian) ────────────────

function probability(fitted::FittedComparativeModel{<:Intransitive, I, R, L},
                     item_i::L, item_j::L) where {I, R, L}
    idx_i = findfirst(==(item_i), fitted.labels)
    idx_j = findfirst(==(item_j), fitted.labels)
    idx_i === nothing && throw(ArgumentError("Label $(item_i) not found in fitted model"))
    idx_j === nothing && throw(ArgumentError("Label $(item_j) not found in fitted model"))
    return probability(fitted, idx_i, idx_j)
end
