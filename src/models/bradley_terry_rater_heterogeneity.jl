# ─── Bradley-Terry with rater heterogeneity ──────────────────────────────────
#
# A mixture in which rater r follows Bradley-Terry with reliability q_r and
# guesses at random otherwise:
#   P(rater r judges i ≻ j) = q_r·σ(λᵢ − λⱼ) + (1 − q_r)/2.
# The data are per-rater comparisons (`RaterData`), aggregated here into
# (rater, unordered pair) cells. The MLE maximises the mixture log-likelihood
# directly; the Bayesian fit augments each comparison with a latent
# informed/guess indicator, so the informed comparisons feed an ordinary
# Pólya-Gamma Bradley-Terry update for λ (reusing `_aggregate_pairs` and the
# machinery in bradley_terry.jl) and q_r gets a conjugate Beta update.

@inline _sigmoid(x::Float64) = x >= 0.0 ? 1.0 / (1.0 + exp(-x)) : (e = exp(x); e / (1.0 + e))

# (rater, pair) cells: rater r, items i < j, n comparisons, w wins of item i.
struct _RaterCells
    rater::Vector{Int}
    i::Vector{Int}
    j::Vector{Int}
    n::Vector{Int}
    w::Vector{Int}
    M::Int
    K::Int
end

function _rater_aggregate(data::RaterData)
    K = length(data.labels)
    M = length(data.raters)
    idx = Dict{Tuple{Int,Int,Int}, Int}()
    rc = Int[]; ic = Int[]; jc = Int[]; nc = Int[]; wc = Int[]
    for c in 1:length(data.winner)
        wnr = data.winner[c]; lsr = data.loser[c]; r = data.rater[c]
        i = min(wnr, lsr); j = max(wnr, lsr)
        key = (r, i, j)
        t = get(idx, key, 0)
        if t == 0
            push!(rc, r); push!(ic, i); push!(jc, j)
            push!(nc, 1); push!(wc, wnr == i ? 1 : 0)
            idx[key] = length(rc)
        else
            nc[t] += 1
            wnr == i && (wc[t] += 1)
        end
    end
    return _RaterCells(rc, ic, jc, nc, wc, M, K)
end

# Mixture data log-likelihood at (λ, q).
function _rater_loglik(λ::AbstractVector, q::AbstractVector, cells::_RaterCells)
    ll = 0.0
    @inbounds for c in 1:length(cells.n)
        s = _sigmoid(λ[cells.i[c]] - λ[cells.j[c]])
        qr = q[cells.rater[c]]
        pwin = qr * s + (1.0 - qr) / 2.0
        w = cells.w[c]; n = cells.n[c]
        ll += w * log(pwin) + (n - w) * log(1.0 - pwin)
    end
    return ll
end

# ─── Maximum likelihood (ridge-penalised) ────────────────────────────────────
#
# θ = [λ₁,…,λ_K; α₁,…,α_M] with q_r = σ(α_r). The joint mixture MLE is
# unbounded — inflating the λ scale saturates σ(λᵢ−λⱼ) and is absorbed by q — so
# a ridge penalty (1/2σ²λ)·Σλ² bounds the strengths and, by preferring the
# mean-zero solution, also fixes the otherwise-free additive constant.

function _rater_neg_loglik(θ::AbstractVector, cells::_RaterCells, σ²λ::Float64)
    K = cells.K
    ll = 0.0
    @inbounds for c in 1:length(cells.n)
        δ = θ[cells.i[c]] - θ[cells.j[c]]
        s = _sigmoid(δ)
        q = _sigmoid(θ[K + cells.rater[c]])
        pwin = q * s + (1.0 - q) / 2.0
        w = cells.w[c]; n = cells.n[c]
        ll += w * log(pwin) + (n - w) * log(1.0 - pwin)
    end
    pen = 0.0
    @inbounds for k in 1:K; pen += θ[k]^2; end
    return -ll + pen / (2.0 * σ²λ)
end

function _rater_neg_grad!(G::AbstractVector, θ::AbstractVector, cells::_RaterCells, σ²λ::Float64)
    K = cells.K
    fill!(G, 0.0)
    @inbounds for c in 1:length(cells.n)
        i = cells.i[c]; j = cells.j[c]; r = cells.rater[c]
        δ = θ[i] - θ[j]
        s = _sigmoid(δ)
        q = _sigmoid(θ[K + r])
        pwin = q * s + (1.0 - q) / 2.0
        ploss = 1.0 - pwin
        w = cells.w[c]; n = cells.n[c]
        A = w / pwin - (n - w) / ploss
        dδ = A * q * s * (1.0 - s)            # ∂ℓ/∂δ
        G[i] -= dδ
        G[j] += dδ
        G[K + r] -= A * (s - 0.5) * q * (1.0 - q)   # ∂ℓ/∂α_r
    end
    @inbounds for k in 1:K; G[k] += θ[k] / σ²λ; end
    return G
end

"""
    fit(model::RaterHeterogeneity{BradleyTerry}, method::MLE, data::RaterData; σ²λ=4.0)

Ridge-penalised maximum-likelihood fit of the rater-heterogeneity Bradley–Terry
mixture via L-BFGS, maximising the mixture log-likelihood with a ridge penalty
`(1/2σ²λ)·Σλ²` (default `σ²λ = 4.0`) on the latent strengths. The penalty bounds
the otherwise-unbounded joint optimum and centres λ. Query with
[`strengths`](@ref), [`rater_reliabilities`](@ref) and [`probability`](@ref).
"""
function fit(model::RaterHeterogeneity{BradleyTerry}, method::MLE,
             data::RaterData{L,R}; σ²λ::Real=4.0) where {L,R}
    K = length(data.labels)
    M = length(data.raters)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit rater-heterogeneity BradleyTerry, got $K"))
    σ²λ > 0 || throw(ArgumentError("σ²λ must be positive, got $σ²λ"))
    cells = _rater_aggregate(data)
    s²λ = Float64(σ²λ)
    f(θ) = _rater_neg_loglik(θ, cells, s²λ)
    g!(G, θ) = _rater_neg_grad!(G, θ, cells, s²λ)
    res = optimize(f, g!, zeros(K + M), LBFGS())
    θ = Optim.minimizer(res)
    λ = θ[1:K]
    λ .-= mean(λ)
    q = [_sigmoid(θ[K + r]) for r in 1:M]
    result = RaterMLEResult(λ, q, data.raters, _rater_loglik(λ, q, cells))
    return FittedComparativeModel(model, method, result, data.labels, data,
                                  Optim.converged(res), Optim.iterations(res))
end

function fit(model::RaterHeterogeneity{BradleyTerry}, data::RaterData)
    return fit(model, MLE(), data)
end

# ─── Bayesian: latent informed/guess indicator + PG λ + conjugate Beta q ─────

# Count of successes in n Bernoulli(p) draws (n is small: comparisons by one
# rater on one pair).
function _rand_binomial(rng::AbstractRNG, n::Int, p::Float64)
    p <= 0.0 && return 0
    p >= 1.0 && return n
    k = 0
    for _ in 1:n
        rand(rng) < p && (k += 1)
    end
    return k
end

"""
    fit(model::RaterHeterogeneity{BradleyTerry}, method::Bayesian, data::RaterData,
        [prior::RaterHeterogeneityPrior]; rng=Random.default_rng())

Bayesian fit of the rater-heterogeneity Bradley–Terry mixture by Gibbs sampling.
Each comparison is augmented with a latent informed/guess indicator; the
informed comparisons drive a Pólya-Gamma Bradley–Terry update of the strengths
λ, while each reliability `q_r` is drawn from its conjugate Beta full conditional
(see [`RaterHeterogeneityPrior`](@ref)). The result holds posterior draws
([`RaterMCMCSamples`](@ref)); query with [`posterior_mean`](@ref),
[`credible_interval`](@ref), [`rater_reliabilities`](@ref) and
[`probability`](@ref).
"""
function fit(model::RaterHeterogeneity{BradleyTerry}, method::Bayesian,
             data::RaterData{L,R}, prior::RaterHeterogeneityPrior;
             rng::AbstractRNG=Random.default_rng()) where {L,R}
    K = length(data.labels)
    M = length(data.raters)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit rater-heterogeneity BradleyTerry, got $K"))
    cells = _rater_aggregate(data)

    λ_prior = prior.λ_prior === nothing ? NormalPrior(K) : prior.λ_prior
    length(λ_prior.μ) == K || throw(DimensionMismatch(
        "λ_prior.μ has length $(length(λ_prior.μ)), expected $K"))
    a0 = prior.q_prior.a
    b0 = prior.q_prior.b

    Σ_inv   = inv(λ_prior.Σ)
    Σ_inv_μ = Σ_inv * λ_prior.μ
    Σ_chol  = cholesky(Symmetric(Matrix(λ_prior.Σ)))   # for the no-information fallback

    total = method.n_burnin + method.thin * method.n_samples
    λ_samples = Matrix{Float64}(undef, method.n_samples, K)
    q_samples = Matrix{Float64}(undef, method.n_samples, M)
    lls       = Vector{Float64}(undef, method.n_samples)

    ncell = length(cells.n)
    λ = zeros(K)
    q = fill(0.5, M)
    winf = Matrix{Int}(undef, K, K)
    inf_count = Vector{Int}(undef, M)
    gss_count = Vector{Int}(undef, M)
    ω = Vector{Float64}(undef, K * (K - 1) ÷ 2)
    V = Matrix{Float64}(undef, K, K)
    m = Vector{Float64}(undef, K)
    z = Vector{Float64}(undef, K)

    for s in 1:total
        # z | λ, q : split each cell's wins into informed and guessing counts
        fill!(winf, 0); fill!(inf_count, 0); fill!(gss_count, 0)
        @inbounds for c in 1:ncell
            i = cells.i[c]; j = cells.j[c]; r = cells.rater[c]
            w = cells.w[c]; n = cells.n[c]
            sδ = _sigmoid(λ[i] - λ[j]); qr = q[r]; half = (1.0 - qr) / 2.0
            pi_i = qr * sδ / (qr * sδ + half)               # P(informed | i won)
            pi_j = qr * (1.0 - sδ) / (qr * (1.0 - sδ) + half)  # P(informed | j won)
            ci = _rand_binomial(rng, w, pi_i)
            cj = _rand_binomial(rng, n - w, pi_j)
            winf[i, j] += ci; winf[j, i] += cj
            inf_count[r] += ci + cj
            gss_count[r] += (w - ci) + (n - w - cj)
        end

        # λ | informed wins : Pólya-Gamma Bradley-Terry update
        agg = _aggregate_pairs(winf, K)
        if agg.P > 0
            @inbounds for p in 1:agg.P
                ii, jj = agg.pairs[p]
                ω[p] = _sample_pg(rng, agg.Nvec[p], λ[ii] - λ[jj])
            end
            copyto!(V, Σ_inv); m .= Σ_inv_μ
            @inbounds for p in 1:agg.P
                ii, jj = agg.pairs[p]; op = ω[p]
                V[ii, ii] += op; V[jj, jj] += op
                V[ii, jj] -= op; V[jj, ii] -= op
                m[ii] += agg.κ[p]; m[jj] -= agg.κ[p]
            end
            @inbounds for k in 1:K; V[k, k] += 1e-10; end
            C = cholesky!(Symmetric(V))
            ldiv!(C, m)
            randn!(rng, z); ldiv!(C.U, z)
            λ .= m .+ z
        else
            randn!(rng, z)
            λ .= λ_prior.μ .+ Σ_chol.L * z
        end
        method.center && (λ .-= mean(λ))

        # q_r | counts : conjugate Beta
        @inbounds for r in 1:M
            q[r] = _sample_beta(rng, a0 + inf_count[r], b0 + gss_count[r])
        end

        if s > method.n_burnin && (s - method.n_burnin) % method.thin == 0
            idx = (s - method.n_burnin) ÷ method.thin
            λ_samples[idx, :] .= λ
            q_samples[idx, :] .= q
            lls[idx] = _rater_loglik(λ, q, cells)
        end
    end

    result = RaterMCMCSamples(λ_samples, q_samples, data.raters, lls,
                              method.n_samples, method.n_burnin, method.thin)
    return FittedComparativeModel(model, method, result, data.labels, data, true, total)
end

function fit(model::RaterHeterogeneity{BradleyTerry}, method::Bayesian,
             data::RaterData{L,R}; rng::AbstractRNG=Random.default_rng()) where {L,R}
    return fit(model, method, data, RaterHeterogeneityPrior(); rng=rng)
end

# ─── Accessors ───────────────────────────────────────────────────────────────

# Named tuple of reliabilities keyed by (stringified) rater label.
_rater_named(labels, vals) = (; (Symbol(string(labels[k])) => vals[k] for k in 1:length(labels))...)

function strengths(fitted::FittedComparativeModel{<:RaterHeterogeneity, MLE, <:RaterMLEResult})
    return copy(fitted.result.λ)
end

function rater_reliabilities(fitted::FittedComparativeModel{<:RaterHeterogeneity, MLE, <:RaterMLEResult})
    r = fitted.result
    return _rater_named(r.rater_labels, r.q)
end

function probability(fitted::FittedComparativeModel{<:RaterHeterogeneity, MLE, <:RaterMLEResult},
                     i::Integer, j::Integer)
    return _sigmoid(fitted.result.λ[i] - fitted.result.λ[j])
end

function posterior_mean(fitted::FittedComparativeModel{<:RaterHeterogeneity, Bayesian, <:RaterMCMCSamples})
    return vec(mean(fitted.result.λ_samples, dims=1))
end

function posterior_std(fitted::FittedComparativeModel{<:RaterHeterogeneity, Bayesian, <:RaterMCMCSamples})
    return vec(std(fitted.result.λ_samples, dims=1))
end

function credible_interval(fitted::FittedComparativeModel{<:RaterHeterogeneity, Bayesian, <:RaterMCMCSamples},
                           k::Integer; prob::Float64=0.95)
    α = (1.0 - prob) / 2.0
    col = fitted.result.λ_samples[:, k]
    return (quantile(col, α), quantile(col, 1.0 - α))
end

function strengths(fitted::FittedComparativeModel{<:RaterHeterogeneity, Bayesian, <:RaterMCMCSamples})
    return posterior_mean(fitted)
end

function rater_reliabilities(fitted::FittedComparativeModel{<:RaterHeterogeneity, Bayesian, <:RaterMCMCSamples})
    r = fitted.result
    return _rater_named(r.rater_labels, vec(mean(r.q_samples, dims=1)))
end

function probability(fitted::FittedComparativeModel{<:RaterHeterogeneity, Bayesian, <:RaterMCMCSamples},
                     i::Integer, j::Integer)
    S = fitted.result.λ_samples
    return mean(1.0 ./ (1.0 .+ exp.(-(S[:, i] .- S[:, j]))))
end

# Label-based probability (shared across MLE and Bayesian).
function probability(fitted::FittedComparativeModel{<:RaterHeterogeneity, I, R, L},
                     item_i::L, item_j::L) where {I, R, L}
    idx_i = findfirst(==(item_i), fitted.labels)
    idx_j = findfirst(==(item_j), fitted.labels)
    idx_i === nothing && throw(ArgumentError("Label $(item_i) not found in fitted model"))
    idx_j === nothing && throw(ArgumentError("Label $(item_j) not found in fitted model"))
    return probability(fitted, idx_i, idx_j)
end
