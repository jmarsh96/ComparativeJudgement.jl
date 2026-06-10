struct _AggregatedPairData
    X::Matrix{Float64}          # P×K design matrix (kept for loglik / external use)
    pairs::Vector{Tuple{Int,Int}} # (i, j) index of each aggregated pair
    Nvec::Vector{Int}           # trial counts per pair
    κ::Vector{Float64}          # y - N/2 per pair (constant)
    P::Int
    K::Int
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
    return _AggregatedPairData(X, pairs, Nvec, κ, P, K)
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

    total = method.n_samples + method.n_burnin
    samples        = Matrix{Float64}(undef, method.n_samples, K)
    loglikelihoods = Vector{Float64}(undef, method.n_samples)

    # Pre-allocate all per-iteration buffers to avoid heap pressure in the loop.
    λ     = zeros(K)
    ω     = Vector{Float64}(undef, agg.P)
    ψ     = Vector{Float64}(undef, agg.P)
    XtΩX  = Matrix{Float64}(undef, K, K)
    V_buf = Matrix{Float64}(undef, K, K)   # will hold V_inv, then its Cholesky
    m     = Vector{Float64}(undef, K)
    z     = Vector{Float64}(undef, K)

    for s in 1:total
        # Step 1: ψ = X * λ — exploit X sparsity (each row is +e_i − e_j)
        @inbounds for p in 1:agg.P
            i, j = agg.pairs[p]
            ψ[p] = λ[i] - λ[j]
        end

        # Sample ω | λ
        @inbounds for p in 1:agg.P
            ω[p] = _sample_pg(rng, agg.Nvec[p], ψ[p])
        end

        # Step 2: XtΩX = X'ΩX — sparse rank-1 updates (4 ops per pair vs O(P·K²))
        fill!(XtΩX, 0.0)
        @inbounds for p in 1:agg.P
            i, j  = agg.pairs[p]
            op    = ω[p]
            XtΩX[i, i] += op
            XtΩX[j, j] += op
            XtΩX[i, j] -= op
            XtΩX[j, i] -= op
        end

        # V_inv = Σ_inv + XtΩX; Cholesky in-place (no allocation)
        V_buf .= Σ_inv .+ XtΩX
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
