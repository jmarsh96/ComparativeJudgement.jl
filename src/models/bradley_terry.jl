function _full_theta(θ_free::AbstractVector{T}) where {T}
    return vcat(zero(T), θ_free)
end

function _bt_neg_loglik(θ_free::AbstractVector, wins::Matrix{Int})
    θ = _full_theta(θ_free)
    λ = exp.(θ)
    n = length(λ)
    ll = zero(eltype(θ))
    for i in 1:n, j in 1:n
        i == j && continue
        w = wins[i, j]
        iszero(w) && continue
        ll += w * log(λ[i] / (λ[i] + λ[j]))
    end
    return -ll
end

function _bt_neg_grad!(G::AbstractVector, θ_free::AbstractVector, wins::Matrix{Int})
    θ = _full_theta(θ_free)
    λ = exp.(θ)
    n = length(λ)
    for k in 2:n
        expected = zero(eltype(θ))
        for j in 1:n
            j == k && continue
            n_kj = wins[k, j] + wins[j, k]
            iszero(n_kj) && continue
            expected += n_kj * λ[k] / (λ[k] + λ[j])
        end
        G[k - 1] = -(sum(wins[k, :]) - expected)
    end
    return G
end

"""
    fit(model::BradleyTerry, method::MLE, data::PairwiseData)

Maximum-likelihood fit of the Bradley–Terry model via L-BFGS. The first
item's strength is fixed at zero during optimisation for identifiability;
[`strengths`](@ref) returns the centred estimates.
"""
function fit(model::BradleyTerry, method::MLE, data::PairwiseData{L}) where {L}
    wins = data.wins
    n = length(data.labels)
    n >= 2 || throw(ArgumentError("Need at least 2 items to fit BradleyTerry, got $n"))
    θ₀ = zeros(n - 1)
    f(θ_free) = _bt_neg_loglik(θ_free, wins)
    g!(G, θ_free) = _bt_neg_grad!(G, θ_free, wins)
    result = optimize(f, g!, θ₀, LBFGS())
    return FittedComparativeModel(
        model, 
        method, 
        result, 
        data.labels,
        Optim.converged(result), 
        Optim.iterations(result)
    )
end

function fit(model::BradleyTerry, method::MLE, wins::Matrix{Int}, labels::Vector{L}) where {L}
    return fit(model, method, PairwiseData(wins, labels))
end

function fit(model::BradleyTerry, data::PairwiseData)
    return fit(model, MLE(), data)
end

function fit(model::BradleyTerry, wins::Matrix{Int}, labels::Vector{L}) where {L}
    return fit(model, MLE(), PairwiseData(wins, labels))
end

function loglikelihood(fitted::FittedComparativeModel{BradleyTerry, MLE})
    return -Optim.minimum(fitted.result)
end

function strengths(fitted::FittedComparativeModel{BradleyTerry, MLE})
    θ = _full_theta(Optim.minimizer(fitted.result))
    return θ .- mean(θ)
end

function probability(fitted::FittedComparativeModel{BradleyTerry, MLE}, i::Integer, j::Integer)
    θ = _full_theta(Optim.minimizer(fitted.result))
    λᵢ = exp(θ[i])
    λⱼ = exp(θ[j])
    return λᵢ / (λᵢ + λⱼ)
end

function probability(fitted::FittedComparativeModel{BradleyTerry, MLE, R, L},
                     item_i::L, item_j::L) where {R, L}
    idx_i = findfirst(==(item_i), fitted.labels)
    idx_j = findfirst(==(item_j), fitted.labels)
    idx_i === nothing && throw(ArgumentError("Label $(item_i) not found in fitted model"))
    idx_j === nothing && throw(ArgumentError("Label $(item_j) not found in fitted model"))
    return probability(fitted, idx_i, idx_j)
end
struct _AggregatedPairData
    X::Matrix{Float64}          # P×K design matrix (kept for loglik / external use)
    pairs::Vector{Tuple{Int,Int}} # (i, j) index of each aggregated pair (i < j)
    Nvec::Vector{Int}           # trial counts per pair
    κ::Vector{Float64}          # y - N/2 per pair (constant)
    P::Int
    K::Int
    # Upper-triangle (i,j) entries with i<j that are NOT a pair, in column-major
    # order so zeroing them in V_buf is cache-friendly. Used to reset Cholesky
    # fill-in without a full O(K²) fill! each iteration.
    upper_zero::Vector{Tuple{Int,Int}}
end

function _aggregate_pairs(wins::Matrix{Int}, K::Int)
    pairs = Tuple{Int,Int}[]
    Nvec  = Int[]
    yvec  = Int[]
    for i in 1:K, j in (i + 1):K
        n_ij = wins[i, j] + wins[j, i]
        iszero(n_ij) && continue
        push!(pairs, (i, j))
        push!(Nvec, n_ij)
        push!(yvec, wins[i, j])
    end
    P = length(pairs)
    X = zeros(P, K)
    for (p, (i, j)) in enumerate(pairs)
        X[p, i] =  1.0
        X[p, j] = -1.0
    end
    κ = Float64.(yvec) .- Float64.(Nvec) ./ 2
    pair_set = Set{Tuple{Int,Int}}(pairs)
    upper_zero = Tuple{Int,Int}[]
    for j in 2:K, i in 1:(j-1)   # column-major order for cache-friendly V_buf access
        (i, j) ∉ pair_set && push!(upper_zero, (i, j))
    end
    return _AggregatedPairData(X, pairs, Nvec, κ, P, K, upper_zero)
end

# log p(wins | λ) using aggregated pair representation
function _bt_loglik(λ::Vector{Float64}, agg::_AggregatedPairData)
    ll = 0.0
    @inbounds for p in 1:agg.P
        i, j = agg.pairs[p]
        ψ = λ[i] - λ[j]
        ll += (agg.κ[p] + agg.Nvec[p] / 2) * ψ - agg.Nvec[p] * log1pexp(ψ)
    end
    return ll
end

# log1pexp: numerically stable log(1 + exp(x))
function log1pexp(x::Float64)
    x < -36.0 && return exp(x)
    x >  36.0 && return x
    return log1p(exp(x))
end

"""
    fit(model::BradleyTerry, method::Bayesian, data::PairwiseData,
        [prior::NormalPrior]; rng=Random.default_rng())

Bayesian fit of the Bradley–Terry model by Pólya-Gamma augmented Gibbs
sampling. `prior` is a `K`-variate [`NormalPrior`](@ref) on the latent
strengths (default `NormalPrior(K)`). The result holds posterior draws
([`BTMCMCSamples`](@ref)); query them with [`posterior_mean`](@ref),
[`posterior_std`](@ref), [`credible_interval`](@ref) and
[`probability`](@ref).
"""
function fit(model::BradleyTerry, method::Bayesian, data::PairwiseData{L},
             prior::NormalPrior; rng::AbstractRNG=Random.default_rng()) where {L}
    K = length(data.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit BradleyTerry, got $K"))
    length(prior.μ) == K || throw(DimensionMismatch(
        "prior.μ has length $(length(prior.μ)), expected $K"))

    agg = _aggregate_pairs(data.wins, K)

    # Pre-computation (once)
    Σ_inv       = inv(prior.Σ)
    Σ_inv_μ     = Σ_inv * prior.μ
    Xt_κ        = agg.X' * agg.κ    # K-vector, constant
    rhs_const   = Σ_inv_μ .+ Xt_κ
    # For diagonal Σ (the common default), avoid copying the full K×K matrix each
    # iteration: only zero the non-pair upper-triangle entries from Cholesky fill-in,
    # then write diagonal and pair entries directly — O(K + P) instead of O(K²).
    diag_Σ_inv  = isdiag(Σ_inv) ? diag(Σ_inv) : nothing

    total = method.n_samples + method.n_burnin
    samples        = Matrix{Float64}(undef, method.n_samples, K)
    loglikelihoods = Vector{Float64}(undef, method.n_samples)

    # Pre-allocate all per-iteration buffers to avoid heap pressure in the loop.
    λ     = zeros(K)
    ω     = Vector{Float64}(undef, agg.P)
    V_buf = Matrix{Float64}(undef, K, K)   # will hold V_inv, then its Cholesky
    m     = Vector{Float64}(undef, K)
    z     = Vector{Float64}(undef, K)

    for s in 1:total
        # Sample ω | λ (ψ = λᵢ − λⱼ computed inline, no separate buffer needed)
        @inbounds for p in 1:agg.P
            i, j = agg.pairs[p]
            ω[p] = _sample_pg(rng, agg.Nvec[p], λ[i] - λ[j])
        end

        # Build V_inv = Σ_inv + XtΩX directly into V_buf.
        # Diagonal prior (common case): O(K + P) — zero only Cholesky fill-in entries,
        # then write diagonal and pair off-diagonals. Each (i,j) pair appears once so
        # we SET V_buf[i,j] = −ω instead of accumulating.
        # General prior: O(K²) copy + O(P) updates.
        if diag_Σ_inv !== nothing
            @inbounds for (i, j) in agg.upper_zero
                V_buf[i, j] = 0.0
            end
            @inbounds for i in 1:K
                V_buf[i, i] = diag_Σ_inv[i]
            end
            @inbounds for p in 1:agg.P
                i, j = agg.pairs[p]
                op = ω[p]
                V_buf[i, i] += op
                V_buf[j, j] += op
                V_buf[i, j] = -op   # upper triangle only; Symmetric reads upper
            end
        else
            copyto!(V_buf, Σ_inv)
            @inbounds for p in 1:agg.P
                i, j = agg.pairs[p]
                op = ω[p]
                V_buf[i, i] += op
                V_buf[j, j] += op
                V_buf[i, j] -= op
                V_buf[j, i] -= op
            end
        end
        # Small diagonal ridge guards against PosDefException when PG weights
        # approach zero and V_buf ≈ Σ_inv, which can have numerical errors
        # O(cond(Σ) * eps) from matrix inversion (critical for large ill-conditioned Σ).
        @inbounds for k in 1:K; V_buf[k, k] += 1e-10; end
        C = cholesky!(Symmetric(V_buf))

        # Posterior mean: m = V_inv \ rhs_const (in-place)
        m .= rhs_const
        ldiv!(C, m)

        # Sample λ ~ N(m, V_inv⁻¹): z = C.U \ randn, λ = m + z
        randn!(rng, z)
        ldiv!(C.U, z)
        λ .= m .+ z

        method.center && (λ .-= mean(λ))

        if s > method.n_burnin
            idx = s - method.n_burnin
            samples[idx, :]     .= λ
            loglikelihoods[idx]  = _bt_loglik(λ, agg)
        end
    end

    result = BTMCMCSamples(samples, loglikelihoods, method.n_samples, method.n_burnin)
    return FittedComparativeModel(model, method, result, data.labels, true, total)
end

function fit(model::BradleyTerry, method::Bayesian, data::PairwiseData{L};
             rng::AbstractRNG=Random.default_rng()) where {L}
    return fit(model, method, data, NormalPrior(length(data.labels)); rng=rng)
end

function fit(model::BradleyTerry, method::Bayesian, wins::Matrix{Int},
             labels::Vector{L}, prior::NormalPrior;
             rng::AbstractRNG=Random.default_rng()) where {L}
    return fit(model, method, PairwiseData(wins, labels), prior; rng=rng)
end

function posterior_mean(fitted::FittedComparativeModel{BradleyTerry, Bayesian})
    return vec(mean(fitted.result.samples, dims=1))
end

function posterior_std(fitted::FittedComparativeModel{BradleyTerry, Bayesian})
    return vec(std(fitted.result.samples, dims=1))
end

function credible_interval(fitted::FittedComparativeModel{BradleyTerry, Bayesian},
                            k::Integer; prob::Float64=0.95)
    α = (1.0 - prob) / 2.0
    col = fitted.result.samples[:, k]
    return (quantile(col, α), quantile(col, 1.0 - α))
end

function loglikelihood(fitted::FittedComparativeModel{BradleyTerry, Bayesian})
    return fitted.result.loglikelihoods
end

function strengths(fitted::FittedComparativeModel{BradleyTerry, Bayesian})
    return posterior_mean(fitted)
end

function probability(fitted::FittedComparativeModel{BradleyTerry, Bayesian},
                     i::Integer, j::Integer)
    S = fitted.result.samples
    return mean(1.0 ./ (1.0 .+ exp.(-(S[:, i] .- S[:, j]))))
end

function probability(fitted::FittedComparativeModel{BradleyTerry, Bayesian, R, L},
                     item_i::L, item_j::L) where {R, L}
    idx_i = findfirst(==(item_i), fitted.labels)
    idx_j = findfirst(==(item_j), fitted.labels)
    idx_i === nothing && throw(ArgumentError("Label $(item_i) not found in fitted model"))
    idx_j === nothing && throw(ArgumentError("Label $(item_j) not found in fitted model"))
    return probability(fitted, idx_i, idx_j)
end
