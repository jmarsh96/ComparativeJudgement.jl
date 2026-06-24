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

# ─── Shared anchored MLE machinery (model-agnostic) ──────────────────────────
#
# The pairwise likelihood already identifies λ up to its (centred) location — the
# probit/logit scale is fixed — so the anchored MLE is two-stage: take the plain
# pairwise-MLE strengths, then calibrate the scale by weighted least squares of
# the anchor values on the group-mean strengths (weights n_g, var σ²/n_g). This
# is well-posed, unlike jointly profiling σ², whose objective is unbounded as the
# anchor residuals are driven to zero.

# Weighted least squares of y on group-mean strengths μ. Returns (a, b, σ², cal_ll)
# where cal_ll is the Gaussian calibration log-likelihood Σ log N(y_g | a+b·μ_g, σ²/n_g).
function _anchored_profile(μ::Vector{Float64}, ng::Vector{Float64},
                           y::Vector{Float64}, G::Int, sum_log_ng::Float64)
    sρ = 0.0; sρμ = 0.0; sρμμ = 0.0; sρy = 0.0; sρμy = 0.0
    @inbounds for g in 1:G
        ρ = ng[g]; μg = μ[g]; yg = y[g]
        sρ += ρ; sρμ += ρ * μg; sρμμ += ρ * μg^2; sρy += ρ * yg; sρμy += ρ * μg * yg
    end
    det = sρ * sρμμ - sρμ^2
    b = det > 1e-10 ? (sρ * sρμy - sρμ * sρy) / det : 1.0   # fall back to b=1 if μ collinear
    a = (sρy - b * sρμ) / sρ
    rss = 0.0
    @inbounds for g in 1:G
        rss += ng[g] * (y[g] - a - b * μ[g])^2
    end
    σ² = max(rss / G, 1e-12)
    cal_ll = -0.5 * (G * log(2π * σ²) - sum_log_ng + rss / σ²)
    return a, b, σ², cal_ll
end

# Build an AnchoredMLEResult from centred strengths λ and the pairwise log-likelihood.
function _anchored_mle_result(λ::Vector{Float64}, pairwise_ll::Float64,
                              groups::Vector{Vector{Int}}, ng::Vector{Float64},
                              y::Vector{Float64}, sum_log_ng::Float64)
    G = length(groups)
    μ = Float64[sum(@view λ[g]) / length(g) for g in groups]
    a, b, σ², cal_ll = _anchored_profile(μ, ng, y, G, sum_log_ng)
    return AnchoredMLEResult(λ, a, b, σ², pairwise_ll + cal_ll)
end

"""
    fit(model::Anchored{BradleyTerry}, method::MLE, data::AnchoredData)

Maximum-likelihood fit of the anchored Bradley–Terry model: the latent strengths
λ are estimated by the plain Bradley–Terry MLE, then the anchor measurements
`y = a + b·λ + ε` calibrate the scale by weighted least squares
([`calibration`](@ref)). Query with [`strengths`](@ref), [`predict`](@ref) and
[`loglikelihood`](@ref) (the maximised joint log-likelihood).
"""
function fit(model::Anchored{BradleyTerry}, method::MLE,
             data::AnchoredData{PairwiseData{L}, L}) where {L}
    pdata = data.data
    K = length(pdata.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit BradleyTerryAnchored, got $K"))
    ng = Float64[length(g) for g in data.anchor_groups]
    sum_log_ng = sum(log, ng)
    mle = fit(BradleyTerry(), MLE(), pdata)
    λ = _full_theta(Optim.minimizer(mle.result))
    λ .-= mean(λ)
    result = _anchored_mle_result(λ, loglikelihood(mle), data.anchor_groups, ng,
                                  data.anchor_values, sum_log_ng)
    return FittedComparativeModel(model, method, result, pdata.labels, data,
                                  mle.converged, mle.iterations)
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

    groups = data.anchor_groups
    y = data.anchor_values
    G = length(groups)
    ng = Float64[length(g) for g in groups]
    sum_log_ng = sum(log, ng)

    agg = _aggregate_pairs(pdata.wins, K)

    # Pre-computation (once)
    τ²        = prior.τ²
    V₀_inv    = inv(prior.β_prior.Σ)
    V₀_inv_β₀ = V₀_inv * prior.β_prior.μ
    α₀, b₀    = prior.σ²_prior.α, prior.σ²_prior.β
    Xt_κ      = agg.X' * agg.κ

    # Anchor averaging operator M (G×K, M[g,i] = 1/n_g). The anchor layer adds
    # (b²/σ²)·MᵀWM to the λ precision and (b/σ²)·MᵀW(y−a) to its mean, with
    # W = diag(n_g). (MᵀWM)_{ij} = Σ_{g: i,j∈G_g} 1/n_g (diagonal for singletons,
    # reproducing the per-item scatter); (MᵀW(y−a))_i = Σ_{g∋i}(y_g−a). Precompute.
    anchor_diag = zeros(K)                       # diagonal of MᵀWM
    anchor_off  = Tuple{Int,Int,Float64}[]       # (i, j, coef) upper triangle, i<j
    member      = [Int[] for _ in 1:K]           # groups each item belongs to
    let offdict = Dict{Tuple{Int,Int}, Float64}()
        for (g, grp) in enumerate(groups)
            w = 1.0 / ng[g]
            for i in grp
                anchor_diag[i] += w
                push!(member[i], g)
            end
            for ia in 1:length(grp), ib in (ia + 1):length(grp)
                i, j = grp[ia], grp[ib]
                key = i < j ? (i, j) : (j, i)
                offdict[key] = get(offdict, key, 0.0) + w
            end
        end
        for (key, c) in offdict
            push!(anchor_off, (key[1], key[2], c))
        end
    end

    # Initialisation: λ from the standalone MLE (centred), β and σ² from OLS of the
    # anchor values on the group-mean strengths.
    λ = zeros(K)
    mle = fit(BradleyTerry(), MLE(), pdata)
    if mle.converged
        λ .= _full_theta(Optim.minimizer(mle.result))
        λ .-= mean(λ)
    end
    μ = Float64[sum(@view λ[g]) / length(g) for g in groups]
    a, b, σ² = _anchored_init_β(μ, y, prior)

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
        # Anchor precision (b²/σ²)·MᵀWM: diagonal then off-diagonals (cleared above
        # via upper_zero for non-pair locations; pair locations were SET, so accumulate).
        @inbounds for i in 1:K
            V_buf[i, i] += b2_σ2 * anchor_diag[i]
        end
        @inbounds for (i, j, c) in anchor_off
            V_buf[i, j] += b2_σ2 * c
        end
        # Anchor mean shift (b/σ²)·MᵀW(y−a): h[i] += (b/σ²)·Σ_{g∋i}(y_g−a).
        @inbounds for i in 1:K
            isempty(member[i]) && continue
            acc = 0.0
            for g in member[i]
                acc += y[g] - a
            end
            h[i] += b_σ2 * acc
        end
        C = cholesky!(Symmetric(V_buf))
        m .= h
        ldiv!(C, m)
        randn!(rng, z)
        ldiv!(C.U, z)
        λ .= m .+ z
        method.center && (λ .-= mean(λ))

        # Group-mean strengths μ_g = mean(λ over G_g) for the calibration updates.
        @inbounds for g in 1:G
            acc = 0.0
            for i in groups[g]
                acc += λ[i]
            end
            μ[g] = acc / ng[g]
        end

        # β | μ, σ² — conjugate 2×2 weighted regression (weights ρ_g = n_g)
        sρ = 0.0; sρμ = 0.0; sρμμ = 0.0; sρy = 0.0; sρμy = 0.0
        @inbounds for g in 1:G
            ρ = ng[g]; μg = μ[g]; yg = y[g]
            sρ += ρ; sρμ += ρ * μg; sρμμ += ρ * μg^2; sρy += ρ * yg; sρμy += ρ * μg * yg
        end
        A11 = V₀_inv[1, 1] + sρ
        A12 = V₀_inv[1, 2] + sρμ
        A22 = V₀_inv[2, 2] + sρμμ
        detA = A11 * A22 - A12^2
        Vβ11 =  A22 / detA
        Vβ12 = -A12 / detA
        Vβ22 =  A11 / detA
        rhs1 = V₀_inv_β₀[1] + sρy
        rhs2 = V₀_inv_β₀[2] + sρμy
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

        # σ² | μ, β — inverse-gamma on the weighted anchor RSS (G groups, weights n_g)
        rss = 0.0
        @inbounds for g in 1:G
            rss += ng[g] * (y[g] - a - b * μ[g])^2
        end
        σ² = _sample_inv_gamma(rng, α₀ + G / 2, b₀ + rss / 2)

        if s > method.n_burnin && (s - method.n_burnin) % method.thin == 0
            idx = (s - method.n_burnin) ÷ method.thin
            λ_samples[idx, :] .= λ
            β_samples[idx, 1]  = a
            β_samples[idx, 2]  = b
            σ²_samples[idx]    = σ²
            loglikelihoods[idx] = _bt_loglik(λ, agg) -
                                  0.5 * (G * log(2π * σ²) - sum_log_ng + rss / σ²)
        end
    end

    result = AnchoredMCMCSamples(λ_samples, β_samples, σ²_samples, loglikelihoods,
                                 method.n_samples, method.n_burnin, method.thin)
    return FittedComparativeModel(model, method, result, pdata.labels, data, true, total)
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

function strengths(fitted::FittedComparativeModel{<:Anchored, Bayesian})
    return posterior_mean(fitted)
end

function calibration(fitted::FittedComparativeModel{<:Anchored, Bayesian})
    res = fitted.result
    return (a = mean(res.β_samples[:, 1]),
            b = mean(res.β_samples[:, 2]),
            σ² = mean(res.σ²_samples))
end

"""
    predict(fitted, k; prob=nothing, rng=Random.default_rng())
    predict(fitted, label; prob=nothing, rng=Random.default_rng())
    predict(fitted)

Posterior-predictive anchor measurements `y* = a + b·λ + ε` for an
[`Anchored`](@ref) fit, on the scale of the anchor values. With an item index
`k` or `label`, returns a vector of posterior-predictive draws, or the symmetric
`prob` credible interval `(lo, hi)` when `prob` is given. With no item argument,
returns the vector of posterior-predictive means for all items.
"""
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

# ─── Anchored MLE accessors (dispatch on AnchoredMLEResult, model-agnostic) ──

function strengths(fitted::FittedComparativeModel{<:Anchored, MLE, AnchoredMLEResult})
    return fitted.result.λ
end

function calibration(fitted::FittedComparativeModel{<:Anchored, MLE, AnchoredMLEResult})
    r = fitted.result
    return (a = r.a, b = r.b, σ² = r.σ²)
end

# Point prediction y* = a + b·λ_k on the anchor measurement scale. With prob
# given, returns the plug-in normal prediction interval a + b·λ_k ± z·σ.
function predict(fitted::FittedComparativeModel{<:Anchored, MLE, AnchoredMLEResult},
                 k::Integer; prob::Union{Nothing, Float64}=nothing)
    r = fitted.result
    ŷ = r.a + r.b * r.λ[k]
    prob === nothing && return ŷ
    0.0 < prob < 1.0 || throw(ArgumentError("prob must be in (0, 1), got $prob"))
    z = _norm_quantile(1.0 - (1.0 - prob) / 2.0) * sqrt(r.σ²)
    return (ŷ - z, ŷ + z)
end

function predict(fitted::FittedComparativeModel{M, MLE, AnchoredMLEResult, L}, label::L;
                 prob::Union{Nothing, Float64}=nothing) where {M <: Anchored, L}
    idx = findfirst(==(label), fitted.labels)
    idx === nothing && throw(ArgumentError("Label $(label) not found in fitted model"))
    return predict(fitted, idx; prob=prob)
end

function predict(fitted::FittedComparativeModel{<:Anchored, MLE, AnchoredMLEResult})
    r = fitted.result
    return r.a .+ r.b .* r.λ
end

# ─── BT-specific accessors for the anchored model ───

function probability(fitted::FittedComparativeModel{Anchored{BradleyTerry}, MLE, AnchoredMLEResult},
                     i::Integer, j::Integer)
    λ = fitted.result.λ
    return 1.0 / (1.0 + exp(-(λ[i] - λ[j])))
end

function probability(fitted::FittedComparativeModel{Anchored{BradleyTerry}, MLE, AnchoredMLEResult, L},
                     item_i::L, item_j::L) where {L}
    idx_i = findfirst(==(item_i), fitted.labels)
    idx_j = findfirst(==(item_j), fitted.labels)
    idx_i === nothing && throw(ArgumentError("Label $(item_i) not found in fitted model"))
    idx_j === nothing && throw(ArgumentError("Label $(item_j) not found in fitted model"))
    return probability(fitted, idx_i, idx_j)
end

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
