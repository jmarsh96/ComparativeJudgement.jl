# ─── Anchored Thurstone Case V: probit comparisons + linear calibration ──────
#
# Same joint structure as the anchored Bradley–Terry model, but the comparison
# likelihood is probit. The MLE is two-stage (plain Thurstone MLE then weighted
# least-squares calibration, via the shared `_anchored_mle_result`); the Bayesian
# fit augments the probit likelihood with Albert–Chib latent utilities, which
# keeps the λ-precision's comparison part constant (XᵀNX) and replaces the
# Pólya-Gamma mean term Xᵀκ with Xᵀu from the latent utilities. The β and σ²
# calibration updates are identical to the Bradley–Terry case.

"""
    fit(model::Anchored{ThurstoneCaseV}, method::MLE, data::AnchoredData)

Maximum-likelihood fit of the anchored Thurstone Case V model: λ is estimated by
the plain Thurstone MLE, then the anchor measurements `y = a + b·λ + ε` calibrate
the scale by weighted least squares. Query with [`strengths`](@ref),
[`calibration`](@ref), [`predict`](@ref) and [`loglikelihood`](@ref).
"""
function fit(model::Anchored{ThurstoneCaseV}, method::MLE,
             data::AnchoredData{PairwiseData{L}, L}) where {L}
    pdata = data.data
    K = length(pdata.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit ThurstoneCaseVAnchored, got $K"))
    ng = Float64[length(g) for g in data.anchor_groups]
    sum_log_ng = sum(log, ng)
    mle = fit(ThurstoneCaseV(), MLE(), pdata)
    λ = _full_theta(Optim.minimizer(mle.result))
    λ .-= mean(λ)
    result = _anchored_mle_result(λ, loglikelihood(mle), data.anchor_groups, ng,
                                  data.anchor_values, sum_log_ng)
    return FittedComparativeModel(model, method, result, pdata.labels,
                                  mle.converged, mle.iterations)
end

"""
    fit(model::Anchored{ThurstoneCaseV}, [method::Bayesian],
        data::AnchoredData, [prior::AnchoredPrior]; rng=Random.default_rng())

Joint Bayesian fit of the anchored Thurstone Case V model by Gibbs sampling:
pairwise comparisons inform λ through the probit likelihood (Albert–Chib
augmented), while anchor measurements `y = a + b·λ + ε` calibrate the latent
scale. The result holds posterior draws of λ, `β = (a, b)` and `σ²`
([`AnchoredMCMCSamples`](@ref)); query with [`posterior_mean`](@ref),
[`credible_interval`](@ref), [`calibration`](@ref), [`predict`](@ref) and
[`probability`](@ref).
"""
function fit(model::Anchored{ThurstoneCaseV}, method::Bayesian,
             data::AnchoredData{PairwiseData{L}, L},
             prior::AnchoredPrior=AnchoredPrior();
             rng::AbstractRNG=Random.default_rng()) where {L}
    pdata = data.data
    K = length(pdata.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit ThurstoneCaseVAnchored, got $K"))

    wins = pdata.wins
    groups = data.anchor_groups
    y = data.anchor_values
    G = length(groups)
    ng = Float64[length(g) for g in groups]
    sum_log_ng = sum(log, ng)

    agg = _aggregate_pairs(wins, K)

    # Pre-computation (once)
    τ²        = prior.τ²
    V₀_inv    = inv(prior.β_prior.Σ)
    V₀_inv_β₀ = V₀_inv * prior.β_prior.μ
    α₀, b₀    = prior.σ²_prior.α, prior.σ²_prior.β

    # Anchor averaging operator MᵀWM / MᵀW(y−a) precompute (see anchored BT).
    anchor_diag = zeros(K)
    anchor_off  = Tuple{Int,Int,Float64}[]
    member      = [Int[] for _ in 1:K]
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

    # Initialisation: λ from the plain Thurstone MLE (centred); β, σ² from OLS.
    λ = zeros(K)
    mle = fit(ThurstoneCaseV(), MLE(), pdata)
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

    V_buf = Matrix{Float64}(undef, K, K)
    h     = Vector{Float64}(undef, K)
    m     = Vector{Float64}(undef, K)
    z     = Vector{Float64}(undef, K)

    for s in 1:total
        # λ | u, β, σ² — precision τ²I + XᵀNX + (b²/σ²)·MᵀWM (the comparison part
        # XᵀNX is constant under the probit augmentation).
        @inbounds for (i, j) in agg.upper_zero
            V_buf[i, j] = 0.0
        end
        @inbounds for i in 1:K
            V_buf[i, i] = τ²
        end
        @inbounds for p in 1:agg.P
            i, j = agg.pairs[p]
            N = agg.Nvec[p]
            V_buf[i, i] += N
            V_buf[j, j] += N
            V_buf[i, j] = -Float64(N)   # SET (each pair appears once); upper triangle
        end
        b2_σ2 = b^2 / σ²
        b_σ2  = b / σ²
        @inbounds for i in 1:K
            V_buf[i, i] += b2_σ2 * anchor_diag[i]
        end
        @inbounds for (i, j, c) in anchor_off
            V_buf[i, j] += b2_σ2 * c
        end

        # Mean term h = Xᵀu (latent utilities) + (b/σ²)·MᵀW(y−a).
        fill!(h, 0.0)
        @inbounds for p in 1:agg.P
            i, j = agg.pairs[p]
            μij = λ[i] - λ[j]
            N = agg.Nvec[p]
            yij = wins[i, j]
            S = 0.0
            for _ in 1:yij
                S += _sample_truncated_normal(rng, μij, true)
            end
            for _ in 1:(N - yij)
                S += _sample_truncated_normal(rng, μij, false)
            end
            h[i] += S; h[j] -= S
        end
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

        # Group-mean strengths for the calibration updates.
        @inbounds for g in 1:G
            acc = 0.0
            for i in groups[g]
                acc += λ[i]
            end
            μ[g] = acc / ng[g]
        end

        # β | μ, σ² — conjugate 2×2 weighted regression (weights ρ_g = n_g).
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
        L11 = sqrt(Vβ11)
        L21 = Vβ12 / L11
        L22 = sqrt(max(Vβ22 - L21^2, 0.0))
        σ  = sqrt(σ²)
        z1 = randn(rng); z2 = randn(rng)
        a = μa + σ * L11 * z1
        b = μb + σ * (L21 * z1 + L22 * z2)

        # σ² | μ, β — inverse-gamma on the weighted anchor RSS.
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
            loglikelihoods[idx] = _tcv_loglik(λ, agg) -
                                  0.5 * (G * log(2π * σ²) - sum_log_ng + rss / σ²)
        end
    end

    result = AnchoredMCMCSamples(λ_samples, β_samples, σ²_samples, loglikelihoods,
                                 method.n_samples, method.n_burnin, method.thin)
    return FittedComparativeModel(model, method, result, pdata.labels, true, total)
end

function fit(model::Anchored{ThurstoneCaseV}, data::AnchoredData{PairwiseData{L}, L};
             rng::AbstractRNG=Random.default_rng()) where {L}
    return fit(model, Bayesian(), data; rng=rng)
end

# ─── Thurstone-specific probability for the anchored model (probit link) ─────

function probability(fitted::FittedComparativeModel{Anchored{ThurstoneCaseV}, Bayesian},
                     i::Integer, j::Integer)
    Sλ = fitted.result.λ_samples
    return mean(_normcdf.(Sλ[:, i] .- Sλ[:, j]))
end

function probability(fitted::FittedComparativeModel{Anchored{ThurstoneCaseV}, Bayesian, R, L},
                     item_i::L, item_j::L) where {R, L}
    idx_i = findfirst(==(item_i), fitted.labels)
    idx_j = findfirst(==(item_j), fitted.labels)
    idx_i === nothing && throw(ArgumentError("Label $(item_i) not found in fitted model"))
    idx_j === nothing && throw(ArgumentError("Label $(item_j) not found in fitted model"))
    return probability(fitted, idx_i, idx_j)
end

function probability(fitted::FittedComparativeModel{Anchored{ThurstoneCaseV}, MLE, AnchoredMLEResult},
                     i::Integer, j::Integer)
    λ = fitted.result.λ
    return _normcdf(λ[i] - λ[j])
end

function probability(fitted::FittedComparativeModel{Anchored{ThurstoneCaseV}, MLE, AnchoredMLEResult, L},
                     item_i::L, item_j::L) where {L}
    idx_i = findfirst(==(item_i), fitted.labels)
    idx_j = findfirst(==(item_j), fitted.labels)
    idx_i === nothing && throw(ArgumentError("Label $(item_i) not found in fitted model"))
    idx_j === nothing && throw(ArgumentError("Label $(item_j) not found in fitted model"))
    return probability(fitted, idx_i, idx_j)
end
