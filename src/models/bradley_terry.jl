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

# ─── Anchored Bradley-Terry (BTA): joint BT + linear calibration Gibbs sampler ───
#
# Anchor measurements y_i = a + b·λ_i + ε for a subset S of items are modelled
# jointly with the pairwise comparisons. Pólya-Gamma augmentation renders the
# BT likelihood conditionally Gaussian, giving closed-form full conditionals
# for λ, β = (a, b), and σ².

# OLS of y on [1 λ_S] for the initial β and σ²; falls back to (mean(y), 1) and
# the prior mean of σ² when the regression is degenerate.
function _anchored_init_β(λS::Vector{Float64}, y::Vector{Float64}, prior::AnchoredPrior)
    r = length(y)
    a = mean(y)
    b = 1.0
    α₀, β₀ = prior.σ²_prior.α, prior.σ²_prior.β
    σ² = α₀ > 1.0 ? β₀ / (α₀ - 1.0) : β₀
    if r >= 2
        sλ  = sum(λS)
        sλλ = sum(abs2, λS)
        denom = r * sλλ - sλ^2
        if denom > 1e-10
            sy  = sum(y)
            sλy = dot(λS, y)
            b = (r * sλy - sλ * sy) / denom
            a = (sy - b * sλ) / r
            if r > 2
                rss = sum((y[k] - a - b * λS[k])^2 for k in 1:r)
                σ² = max(rss / (r - 2), 1e-6)
            end
        end
    end
    return a, b, σ²
end

"""
    fit(model::Anchored{BradleyTerry}, [method::Bayesian],
        data::AnchoredData, [prior::AnchoredPrior]; rng=Random.default_rng())

Joint Bayesian fit of the anchored Bradley–Terry model by Gibbs sampling:
pairwise comparisons inform the latent strengths λ through the Bradley–Terry
likelihood (Pólya-Gamma augmented), while anchor measurements
`y = a + b·λ + ε` for the anchored subset calibrate the latent scale. The
result holds posterior draws of λ, `β = (a, b)` and `σ²`
([`AnchoredMCMCSamples`](@ref)); query them with [`posterior_mean`](@ref),
[`credible_interval`](@ref), [`calibration`](@ref), [`predict`](@ref) and
[`probability`](@ref).
"""
function fit(model::Anchored{BradleyTerry}, method::Bayesian,
             data::AnchoredData{PairwiseData{L}, L},
             prior::AnchoredPrior=AnchoredPrior();
             rng::AbstractRNG=Random.default_rng()) where {L}
    pdata = data.data
    K = length(pdata.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit BradleyTerryAnchored, got $K"))

    S = data.anchor_idx
    y = data.anchor_values
    r = length(S)

    agg = _aggregate_pairs(pdata.wins, K)

    # Pre-computation (once)
    τ²        = prior.τ²
    V₀_inv    = inv(prior.β_prior.Σ)
    V₀_inv_β₀ = V₀_inv * prior.β_prior.μ
    α₀, b₀    = prior.σ²_prior.α, prior.σ²_prior.β
    Xt_κ      = agg.X' * agg.κ

    # Initialisation: λ from the standalone MLE (centred), β and σ² from OLS.
    λ = zeros(K)
    mle = fit(BradleyTerry(), MLE(), pdata)
    if mle.converged
        λ .= _full_theta(Optim.minimizer(mle.result))
        λ .-= mean(λ)
    end
    a, b, σ² = _anchored_init_β(λ[S], y, prior)

    total = method.n_burnin + method.thin * method.n_samples
    λ_samples      = Matrix{Float64}(undef, method.n_samples, K)
    β_samples      = Matrix{Float64}(undef, method.n_samples, 2)
    σ²_samples     = Vector{Float64}(undef, method.n_samples)
    loglikelihoods = Vector{Float64}(undef, method.n_samples)

    # Pre-allocate all per-iteration buffers to avoid heap pressure in the loop.
    ω     = Vector{Float64}(undef, agg.P)
    V_buf = Matrix{Float64}(undef, K, K)   # precision, then its Cholesky
    h     = Vector{Float64}(undef, K)
    m     = Vector{Float64}(undef, K)
    z     = Vector{Float64}(undef, K)

    for s in 1:total
        # ω | λ — Pólya-Gamma step (ψ computed inline, no separate buffer)
        @inbounds for p in 1:agg.P
            i, j = agg.pairs[p]
            ω[p] = _sample_pg(rng, agg.Nvec[p], λ[i] - λ[j])
        end

        # λ | ω, β, σ² — build V_inv = τ²I + XtΩX + (b²/σ²)P_S into V_buf.
        # Prior is always diagonal (τ²I), so use O(K + P) assembly: zero only the
        # non-pair upper-triangle entries left by the previous Cholesky, then set
        # diagonal and pair off-diagonals directly.
        @inbounds for (i, j) in agg.upper_zero
            V_buf[i, j] = 0.0
        end
        @inbounds for i in 1:K
            V_buf[i, i] = τ²
        end
        @inbounds for p in 1:agg.P
            i, j = agg.pairs[p]
            op   = ω[p]
            V_buf[i, i] += op
            V_buf[j, j] += op
            V_buf[i, j] = -op   # SET (each pair appears once); upper triangle only
        end
        h .= Xt_κ
        b2_σ2 = b^2 / σ²
        b_σ2  = b / σ²
        @inbounds for (k, i) in enumerate(S)
            V_buf[i, i] += b2_σ2
            h[i]        += b_σ2 * (y[k] - a)
        end
        C = cholesky!(Symmetric(V_buf))
        m .= h
        ldiv!(C, m)
        randn!(rng, z)
        ldiv!(C.U, z)
        λ .= m .+ z
        method.center && (λ .-= mean(λ))

        # β | λ_S, σ² — conjugate 2×2 Bayesian linear regression
        sλ = 0.0; sλλ = 0.0; sy = 0.0; sλy = 0.0
        @inbounds for (k, i) in enumerate(S)
            λᵢ = λ[i]; yₖ = y[k]
            sλ += λᵢ; sλλ += λᵢ^2; sy += yₖ; sλy += λᵢ * yₖ
        end
        A11 = V₀_inv[1, 1] + r
        A12 = V₀_inv[1, 2] + sλ
        A22 = V₀_inv[2, 2] + sλλ
        detA = A11 * A22 - A12^2
        Vβ11 =  A22 / detA
        Vβ12 = -A12 / detA
        Vβ22 =  A11 / detA
        rhs1 = V₀_inv_β₀[1] + sy
        rhs2 = V₀_inv_β₀[2] + sλy
        μa = Vβ11 * rhs1 + Vβ12 * rhs2
        μb = Vβ12 * rhs1 + Vβ22 * rhs2
        # Draw from N(μ_β, σ²V_β) via the 2×2 Cholesky of V_β
        L11 = sqrt(Vβ11)
        L21 = Vβ12 / L11
        L22 = sqrt(max(Vβ22 - L21^2, 0.0))
        σ  = sqrt(σ²)
        z1 = randn(rng); z2 = randn(rng)
        a = μa + σ * L11 * z1
        b = μb + σ * (L21 * z1 + L22 * z2)

        # σ² | λ_S, β — inverse-gamma step
        rss = 0.0
        @inbounds for (k, i) in enumerate(S)
            rss += (y[k] - a - b * λ[i])^2
        end
        σ² = _sample_inv_gamma(rng, α₀ + r / 2, b₀ + rss / 2)

        if s > method.n_burnin && (s - method.n_burnin) % method.thin == 0
            idx = (s - method.n_burnin) ÷ method.thin
            λ_samples[idx, :] .= λ
            β_samples[idx, 1]  = a
            β_samples[idx, 2]  = b
            σ²_samples[idx]    = σ²
            loglikelihoods[idx] = _bt_loglik(λ, agg) -
                                  0.5 * (r * log(2π * σ²) + rss / σ²)
        end
    end

    result = AnchoredMCMCSamples(λ_samples, β_samples, σ²_samples, loglikelihoods,
                                 method.n_samples, method.n_burnin, method.thin)
    return FittedComparativeModel(model, method, result, pdata.labels, true, total)
end

function fit(model::Anchored{BradleyTerry}, data::AnchoredData{PairwiseData{L}, L};
             rng::AbstractRNG=Random.default_rng()) where {L}
    return fit(model, Bayesian(), data; rng=rng)
end

# ─── Accessors for anchored models ───
# These dispatch on Anchored{<:Any} since AnchoredMCMCSamples is model-agnostic;
# a future anchored Thurstone/Plackett-Luce model gets them for free.

function posterior_mean(fitted::FittedComparativeModel{<:Anchored, Bayesian})
    return vec(mean(fitted.result.λ_samples, dims=1))
end

function posterior_std(fitted::FittedComparativeModel{<:Anchored, Bayesian})
    return vec(std(fitted.result.λ_samples, dims=1))
end

function credible_interval(fitted::FittedComparativeModel{<:Anchored, Bayesian},
                           k::Integer; prob::Float64=0.95)
    α = (1.0 - prob) / 2.0
    col = fitted.result.λ_samples[:, k]
    return (quantile(col, α), quantile(col, 1.0 - α))
end

function loglikelihood(fitted::FittedComparativeModel{<:Anchored, Bayesian})
    return fitted.result.loglikelihoods
end

function strengths(fitted::FittedComparativeModel{<:Anchored, Bayesian})
    return posterior_mean(fitted)
end

function calibration(fitted::FittedComparativeModel{<:Anchored, Bayesian})
    res = fitted.result
    return (a = mean(res.β_samples[:, 1]),
            b = mean(res.β_samples[:, 2]),
            σ² = mean(res.σ²_samples))
end

# Posterior-predictive draws y* = a + b·λ_k + ε on the anchor measurement scale.
# With prob given, returns the symmetric credible interval of y* instead.
function predict(fitted::FittedComparativeModel{<:Anchored, Bayesian}, k::Integer;
                 prob::Union{Nothing, Float64}=nothing,
                 rng::AbstractRNG=Random.default_rng())
    res = fitted.result
    draws = res.β_samples[:, 1] .+ res.β_samples[:, 2] .* res.λ_samples[:, k] .+
            sqrt.(res.σ²_samples) .* randn(rng, res.n_samples)
    prob === nothing && return draws
    α = (1.0 - prob) / 2.0
    return (quantile(draws, α), quantile(draws, 1.0 - α))
end

function predict(fitted::FittedComparativeModel{M, Bayesian, R, L}, label::L;
                 prob::Union{Nothing, Float64}=nothing,
                 rng::AbstractRNG=Random.default_rng()) where {M <: Anchored, R, L}
    idx = findfirst(==(label), fitted.labels)
    idx === nothing && throw(ArgumentError("Label $(label) not found in fitted model"))
    return predict(fitted, idx; prob=prob, rng=rng)
end

# Posterior-predictive means for all items (the noise term has mean zero).
function predict(fitted::FittedComparativeModel{<:Anchored, Bayesian})
    res = fitted.result
    return vec(mean(res.β_samples[:, 1] .+ res.β_samples[:, 2] .* res.λ_samples, dims=1))
end

# ─── BT-specific accessors for the anchored model ───

function probability(fitted::FittedComparativeModel{Anchored{BradleyTerry}, Bayesian},
                     i::Integer, j::Integer)
    Sλ = fitted.result.λ_samples
    return mean(1.0 ./ (1.0 .+ exp.(-(Sλ[:, i] .- Sλ[:, j]))))
end

function probability(fitted::FittedComparativeModel{Anchored{BradleyTerry}, Bayesian, R, L},
                     item_i::L, item_j::L) where {R, L}
    idx_i = findfirst(==(item_i), fitted.labels)
    idx_j = findfirst(==(item_j), fitted.labels)
    idx_i === nothing && throw(ArgumentError("Label $(item_i) not found in fitted model"))
    idx_j === nothing && throw(ArgumentError("Label $(item_j) not found in fitted model"))
    return probability(fitted, idx_i, idx_j)
end
