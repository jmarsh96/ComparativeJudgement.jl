# ─── Bradley-Terry with covariates: λ_i = z_iᵀβ ──────────────────────────────
#
# When item strengths are a linear function of item covariates, the comparison
# log-odds become `logit P(i beats j) = (z_i − z_j)ᵀβ`. This is logistic
# regression on the covariate-difference design matrix D (rows d_p = z_i − z_j),
# so the same Pólya-Gamma machinery used for the plain Bradley-Terry model
# applies — only the design matrix and the target (β instead of λ) change.

# Aggregated covariate representation: one row per pair with data, the row being
# the covariate difference z_i − z_j. Mirrors `_aggregate_pairs` for plain BT.
struct _AggregatedCovariateData
    D::Matrix{Float64}    # P×p difference design: row p = Z[i,:] − Z[j,:]
    Nvec::Vector{Int}     # trial counts per pair
    κ::Vector{Float64}    # y_p − N_p/2 per pair (constant)
    yvec::Vector{Int}     # wins of item i per pair
    P::Int
    p::Int
    Z::Matrix{Float64}    # K×p item covariates (to recover λ = Zβ)
end

function _aggregate_covariate_pairs(cd::CovariateData)
    wins = cd.data.wins
    Z = cd.Z
    K, p = size(Z)
    pairs_i = Int[]; pairs_j = Int[]
    Nvec = Int[]; yvec = Int[]
    for i in 1:K, j in (i + 1):K
        n_ij = wins[i, j] + wins[j, i]
        iszero(n_ij) && continue
        push!(pairs_i, i); push!(pairs_j, j)
        push!(Nvec, n_ij); push!(yvec, wins[i, j])
    end
    P = length(Nvec)
    D = Matrix{Float64}(undef, P, p)
    @inbounds for q in 1:P
        i = pairs_i[q]; j = pairs_j[q]
        for k in 1:p
            D[q, k] = Z[i, k] - Z[j, k]
        end
    end
    κ = Float64.(yvec) .- Float64.(Nvec) ./ 2
    return _AggregatedCovariateData(D, Nvec, κ, yvec, P, p, Z)
end

# Subset the covariates of an aggregated representation (for stepwise selection).
function _subset_agg(agg::_AggregatedCovariateData, cols::Vector{Int})
    return _AggregatedCovariateData(agg.D[:, cols], agg.Nvec, agg.κ, agg.yvec,
                                    agg.P, length(cols), agg.Z[:, cols])
end

# log p(wins | β) for the covariate model (binomial form, numerically stable).
function _btcov_loglik(β::AbstractVector, agg::_AggregatedCovariateData)
    ll = 0.0
    @inbounds for q in 1:agg.P
        ψ = 0.0
        for k in 1:agg.p
            ψ += agg.D[q, k] * β[k]
        end
        ll += agg.yvec[q] * ψ - agg.Nvec[q] * log1pexp(ψ)
    end
    return ll
end

# ─── Maximum likelihood ──────────────────────────────────────────────────────

function _btcov_neg_loglik(β::AbstractVector, agg::_AggregatedCovariateData)
    return -_btcov_loglik(β, agg)
end

function _btcov_neg_grad!(G::AbstractVector, β::AbstractVector, agg::_AggregatedCovariateData)
    fill!(G, 0.0)
    @inbounds for q in 1:agg.P
        ψ = 0.0
        for k in 1:agg.p
            ψ += agg.D[q, k] * β[k]
        end
        μ = 1.0 / (1.0 + exp(-ψ))
        r = agg.yvec[q] - agg.Nvec[q] * μ      # observed − expected wins
        for k in 1:agg.p
            G[k] -= agg.D[q, k] * r
        end
    end
    return G
end

# Inverse Fisher information DᵀWD (W = N·μ(1−μ)) at β, for coefficient SEs.
function _btcov_vcov(β::AbstractVector, agg::_AggregatedCovariateData)
    W = Vector{Float64}(undef, agg.P)
    @inbounds for q in 1:agg.P
        ψ = 0.0
        for k in 1:agg.p
            ψ += agg.D[q, k] * β[k]
        end
        μ = 1.0 / (1.0 + exp(-ψ))
        W[q] = agg.Nvec[q] * μ * (1.0 - μ)
    end
    info = agg.D' * (W .* agg.D)
    try
        return Matrix(inv(Symmetric(info)))
    catch
        return Matrix(inv(Symmetric(info + 1e-8 * I)))
    end
end

# Core MLE solve on an (already-subsetted) aggregated representation.
function _fit_covariate_mle(agg::_AggregatedCovariateData)
    if agg.p == 0
        ll = -log(2.0) * sum(agg.Nvec)   # ψ ≡ 0 ⇒ every comparison is a coin flip
        return (β = Float64[], vcov = zeros(0, 0), loglik = ll,
                converged = true, iterations = 0)
    end
    f(β) = _btcov_neg_loglik(β, agg)
    g!(G, β) = _btcov_neg_grad!(G, β, agg)
    res = optimize(f, g!, zeros(agg.p), LBFGS())
    β = Optim.minimizer(res)
    return (β = β, vcov = _btcov_vcov(β, agg), loglik = -Optim.minimum(res),
            converged = Optim.converged(res), iterations = Optim.iterations(res))
end

"""
    fit(model::Covariates{BradleyTerry}, method::MLE, data::CovariateData)

Maximum-likelihood fit of the covariate Bradley–Terry model via L-BFGS: the
comparison log-odds are `(z_i − z_j)ᵀβ`, so this is logistic regression on the
covariate-difference design. [`coefficients`](@ref) returns the estimated β and
[`strengths`](@ref) the recovered latent strengths `λ = Zβ`.
"""
function fit(model::Covariates{BradleyTerry}, method::MLE, cd::CovariateData{L}) where {L}
    K = length(cd.data.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit covariate BradleyTerry, got $K"))
    agg = _aggregate_covariate_pairs(cd)
    agg.P >= 1 || throw(ArgumentError("No pairs with comparison data to fit"))
    fr = _fit_covariate_mle(agg)
    result = CovariateMLEResult(fr.β, fr.vcov, fr.loglik, cd.Z, cd.names,
                                collect(1:agg.p), NamedTuple[])
    return FittedComparativeModel(model, method, result, cd.data.labels,
                                  fr.converged, fr.iterations)
end

function fit(model::Covariates{BradleyTerry}, cd::CovariateData)
    return fit(model, MLE(), cd)
end

# ─── Stepwise selection ──────────────────────────────────────────────────────

_ic(loglik::Float64, npar::Int, n::Int, criterion::Symbol) =
    -2.0 * loglik + (criterion === :AIC ? 2.0 : log(n)) * npar

"""
    fit(model::Covariates{BradleyTerry}, method::StepwiseMLE, data::CovariateData)

Stepwise maximum-likelihood selection of covariates by AIC or BIC (see
[`StepwiseMLE`](@ref)). Greedily adds and/or removes covariates until the
information criterion can no longer be improved, then returns the fit of the
selected subset. The selected indices and the search trace are recorded in the
result; query with [`coefficients`](@ref) and [`strengths`](@ref).
"""
function fit(model::Covariates{BradleyTerry}, method::StepwiseMLE, cd::CovariateData{L}) where {L}
    K = length(cd.data.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit covariate BradleyTerry, got $K"))
    agg = _aggregate_covariate_pairs(cd)
    agg.P >= 1 || throw(ArgumentError("No pairs with comparison data to fit"))
    p = agg.p
    n = sum(agg.Nvec)
    allow_add = method.direction in (:forward, :both)
    allow_remove = method.direction in (:backward, :both)

    fit_subset(cols) = _fit_covariate_mle(_subset_agg(agg, cols))

    selected = method.direction === :backward ? collect(1:p) : Int[]
    cur = fit_subset(selected)
    cur_ic = _ic(cur.loglik, length(selected), n, method.criterion)
    trace = NamedTuple[(; step = 0, selected = copy(selected), ic = cur_ic, loglik = cur.loglik)]

    step = 0
    while true
        best_ic = cur_ic
        best_fit = nothing
        best_sel = nothing
        if allow_add
            for c in 1:p
                c in selected && continue
                cand = sort!(vcat(selected, c))
                fr = fit_subset(cand)
                ic = _ic(fr.loglik, length(cand), n, method.criterion)
                if ic < best_ic - 1e-8
                    best_ic = ic; best_fit = fr; best_sel = cand
                end
            end
        end
        if allow_remove
            for c in selected
                cand = filter(!=(c), selected)
                fr = fit_subset(cand)
                ic = _ic(fr.loglik, length(cand), n, method.criterion)
                if ic < best_ic - 1e-8
                    best_ic = ic; best_fit = fr; best_sel = cand
                end
            end
        end
        best_sel === nothing && break
        selected = best_sel; cur = best_fit; cur_ic = best_ic
        step += 1
        push!(trace, (; step = step, selected = copy(selected), ic = cur_ic, loglik = cur.loglik))
    end

    result = CovariateMLEResult(cur.β, cur.vcov, cur.loglik, cd.Z, cd.names, selected, trace)
    return FittedComparativeModel(model, method, result, cd.data.labels, cur.converged, step)
end

# ─── MLE / Stepwise accessors (dispatch on CovariateMLEResult) ───────────────

function strengths(fitted::FittedComparativeModel{M, I, CovariateMLEResult}) where {M <: Covariates, I}
    r = fitted.result
    λ = isempty(r.selected) ? zeros(size(r.Z, 1)) : r.Z[:, r.selected] * r.β
    return λ .- mean(λ)
end

"""
    coefficients(fitted)

Estimated covariate coefficients β of a [`Covariates`](@ref) fit, as a named
tuple keyed by covariate name. For [`MLE`](@ref)/[`StepwiseMLE`](@ref) fits these
are the point estimates (only the selected covariates); for [`Bayesian`](@ref)
fits, the posterior means.
"""
function coefficients(fitted::FittedComparativeModel{M, I, CovariateMLEResult}) where {M <: Covariates, I}
    r = fitted.result
    return (; (r.names[r.selected] .=> r.β)...)
end

function loglikelihood(fitted::FittedComparativeModel{M, I, CovariateMLEResult}) where {M <: Covariates, I}
    return fitted.result.loglik
end

function probability(fitted::FittedComparativeModel{M, I, CovariateMLEResult},
                     i::Integer, j::Integer) where {M <: Covariates, I}
    r = fitted.result
    d = 0.0
    for (c, k) in enumerate(r.selected)
        d += (r.Z[i, k] - r.Z[j, k]) * r.β[c]
    end
    return 1.0 / (1.0 + exp(-d))
end

function probability(fitted::FittedComparativeModel{M, I, CovariateMLEResult, L},
                     item_i::L, item_j::L) where {M <: Covariates, I, L}
    idx_i = findfirst(==(item_i), fitted.labels)
    idx_j = findfirst(==(item_j), fitted.labels)
    idx_i === nothing && throw(ArgumentError("Label $(item_i) not found in fitted model"))
    idx_j === nothing && throw(ArgumentError("Label $(item_j) not found in fitted model"))
    return probability(fitted, idx_i, idx_j)
end

# ─── Bayesian: one generic PG Gibbs, shrinkage dispatched on prior type ──────
#
# Per sweep: draw ω | β (Pólya-Gamma), update prior-specific shrinkage state
# given β, then draw β | ω from its Gaussian full conditional with precision
# `Binv + DᵀΩD`. Each prior implements `_init_shrinkage` and
# `_update_shrinkage!`; everything else is shared.

abstract type _ShrinkState end

mutable struct _NormalState <: _ShrinkState
    Binv::Matrix{Float64}
    Binv_μ::Vector{Float64}
end

mutable struct _HorseshoeState <: _ShrinkState
    Binv::Matrix{Float64}     # diagonal precision 1/(τ²λ²ₖ)
    Binv_μ::Vector{Float64}   # zero (mean-zero prior)
    λ²::Vector{Float64}       # local scales²
    ν::Vector{Float64}        # local auxiliaries
    τ²::Float64               # global scale²
    ξ::Float64                # global auxiliary
    τ₀::Float64
end

mutable struct _SpikeSlabState <: _ShrinkState
    Binv::Matrix{Float64}     # diagonal precision 1/(γₖv_slab + (1−γₖ)v_spike)
    Binv_μ::Vector{Float64}   # zero
    γ::Vector{Float64}        # inclusion indicators (0/1)
    v_slab::Float64
    v_spike::Float64
    π₀::Float64
end

_records_inclusion(::_ShrinkState) = false
_records_inclusion(::_SpikeSlabState) = true

function _init_shrinkage(prior::NormalPrior, p::Int)
    p == length(prior.μ) || throw(DimensionMismatch(
        "prior has dimension $(length(prior.μ)) but there are $p covariates"))
    Binv = Matrix(inv(Symmetric(prior.Σ)))
    return _NormalState(Binv, Binv * prior.μ)
end

function _init_shrinkage(prior::HorseshoePrior, p::Int)
    return _HorseshoeState(Matrix{Float64}(I, p, p), zeros(p),
                           ones(p), ones(p), 1.0, 1.0, prior.τ₀)
end

function _init_shrinkage(prior::SpikeSlabPrior, p::Int)
    Binv = Matrix{Float64}(I, p, p) ./ prior.v_slab
    return _SpikeSlabState(Binv, zeros(p), ones(p), prior.v_slab, prior.v_spike, prior.π₀)
end

_update_shrinkage!(::_NormalState, ::NormalPrior, ::AbstractVector, ::AbstractRNG) = nothing

# Makalic & Schmidt (2016) inverse-gamma auxiliary scheme for the horseshoe.
function _update_shrinkage!(s::_HorseshoeState, ::HorseshoePrior, β::AbstractVector, rng::AbstractRNG)
    p = length(β)
    @inbounds for k in 1:p
        s.λ²[k] = _sample_inv_gamma(rng, 1.0, 1.0 / s.ν[k] + β[k]^2 / (2.0 * s.τ²))
        s.ν[k]  = _sample_inv_gamma(rng, 1.0, 1.0 + 1.0 / s.λ²[k])
    end
    sβ = 0.0
    @inbounds for k in 1:p
        sβ += β[k]^2 / s.λ²[k]
    end
    s.τ² = _sample_inv_gamma(rng, (p + 1) / 2.0, 1.0 / s.ξ + sβ / 2.0)
    s.ξ  = _sample_inv_gamma(rng, 1.0, 1.0 / s.τ₀^2 + 1.0 / s.τ²)
    @inbounds for k in 1:p
        s.Binv[k, k] = 1.0 / (s.τ² * s.λ²[k])
    end
    return nothing
end

# Stochastic-search (continuous) spike-and-slab inclusion update.
function _update_shrinkage!(s::_SpikeSlabState, ::SpikeSlabPrior, β::AbstractVector, rng::AbstractRNG)
    @inbounds for k in 1:length(β)
        l1 = log(s.π₀)       - 0.5 * log(s.v_slab)  - β[k]^2 / (2.0 * s.v_slab)
        l0 = log(1 - s.π₀)   - 0.5 * log(s.v_spike) - β[k]^2 / (2.0 * s.v_spike)
        prob1 = 1.0 / (1.0 + exp(l0 - l1))
        g = rand(rng) < prob1 ? 1.0 : 0.0
        s.γ[k] = g
        s.Binv[k, k] = 1.0 / (g * s.v_slab + (1.0 - g) * s.v_spike)
    end
    return nothing
end

function _pg_logistic_gibbs(agg::_AggregatedCovariateData, prior::AbstractPrior,
                            method::Bayesian, rng::AbstractRNG)
    p = agg.p
    P = agg.P
    Dtκ = agg.D' * agg.κ                 # p-vector, constant
    state = _init_shrinkage(prior, p)
    record_incl = _records_inclusion(state)

    total = method.n_burnin + method.thin * method.n_samples
    β_samples = Matrix{Float64}(undef, method.n_samples, p)
    lls       = Vector{Float64}(undef, method.n_samples)
    incl      = record_incl ? Matrix{Float64}(undef, method.n_samples, p) : nothing

    β = zeros(p)
    ψ = Vector{Float64}(undef, P)
    ω = Vector{Float64}(undef, P)
    z = Vector{Float64}(undef, p)

    for s in 1:total
        # ω | β
        mul!(ψ, agg.D, β)
        @inbounds for q in 1:P
            ω[q] = _sample_pg(rng, agg.Nvec[q], ψ[q])
        end

        # shrinkage hyperparameters | β
        _update_shrinkage!(state, prior, β, rng)

        # β | ω : precision V = Binv + DᵀΩD
        V = Matrix(state.Binv) .+ agg.D' * (ω .* agg.D)
        @inbounds for k in 1:p
            V[k, k] += 1e-10
        end
        C = cholesky!(Symmetric(V))
        m = C \ (Dtκ .+ state.Binv_μ)
        randn!(rng, z)
        ldiv!(C.U, z)
        β = m .+ z

        if s > method.n_burnin && (s - method.n_burnin) % method.thin == 0
            idx = (s - method.n_burnin) ÷ method.thin
            β_samples[idx, :] .= β
            lls[idx] = _btcov_loglik(β, agg)
            record_incl && (incl[idx, :] .= state.γ)
        end
    end
    return β_samples, lls, incl
end

"""
    fit(model::Covariates{BradleyTerry}, method::Bayesian, data::CovariateData,
        [prior]; rng=Random.default_rng())

Bayesian fit of the covariate Bradley–Terry model by Pólya-Gamma augmented Gibbs
sampling of the coefficients β. `prior` is one of [`NormalPrior`](@ref) (default
`NormalPrior(p)`), [`HorseshoePrior`](@ref) for global-local shrinkage, or
[`SpikeSlabPrior`](@ref) for variable selection with posterior inclusion
probabilities. The result holds posterior draws ([`CovariateMCMCSamples`](@ref));
query with [`coefficients`](@ref), [`strengths`](@ref), [`posterior_mean`](@ref),
[`credible_interval`](@ref), [`inclusion_probabilities`](@ref).
"""
function fit(model::Covariates{BradleyTerry}, method::Bayesian, cd::CovariateData{L},
             prior::AbstractPrior; rng::AbstractRNG=Random.default_rng()) where {L}
    K = length(cd.data.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit covariate BradleyTerry, got $K"))
    agg = _aggregate_covariate_pairs(cd)
    agg.P >= 1 || throw(ArgumentError("No pairs with comparison data to fit"))
    β_samples, lls, incl = _pg_logistic_gibbs(agg, prior, method, rng)
    result = CovariateMCMCSamples(β_samples, lls, incl, cd.Z, cd.names,
                                  method.n_samples, method.n_burnin, method.thin)
    total = method.n_burnin + method.thin * method.n_samples
    return FittedComparativeModel(model, method, result, cd.data.labels, true, total)
end

function fit(model::Covariates{BradleyTerry}, method::Bayesian, cd::CovariateData{L};
             rng::AbstractRNG=Random.default_rng()) where {L}
    return fit(model, method, cd, NormalPrior(size(cd.Z, 2)); rng=rng)
end

# ─── Bayesian accessors (dispatch on CovariateMCMCSamples) ───────────────────

# Centred latent-strength draws λ = Zβ, n_samples × K (centring matches the
# sum-to-zero convention used for the plain Bradley-Terry strengths).
function _lambda_draws(r::CovariateMCMCSamples)
    Λ = r.β_samples * r.Z'
    Λ .-= mean(Λ, dims=2)
    return Λ
end

function posterior_mean(fitted::FittedComparativeModel{M, Bayesian, CovariateMCMCSamples}) where {M <: Covariates}
    return vec(mean(_lambda_draws(fitted.result), dims=1))
end

function posterior_std(fitted::FittedComparativeModel{M, Bayesian, CovariateMCMCSamples}) where {M <: Covariates}
    return vec(std(_lambda_draws(fitted.result), dims=1))
end

function credible_interval(fitted::FittedComparativeModel{M, Bayesian, CovariateMCMCSamples},
                           k::Integer; prob::Float64=0.95) where {M <: Covariates}
    α = (1.0 - prob) / 2.0
    col = _lambda_draws(fitted.result)[:, k]
    return (quantile(col, α), quantile(col, 1.0 - α))
end

function strengths(fitted::FittedComparativeModel{M, Bayesian, CovariateMCMCSamples}) where {M <: Covariates}
    return posterior_mean(fitted)
end

function loglikelihood(fitted::FittedComparativeModel{M, Bayesian, CovariateMCMCSamples}) where {M <: Covariates}
    return fitted.result.loglikelihoods
end

function coefficients(fitted::FittedComparativeModel{M, Bayesian, CovariateMCMCSamples}) where {M <: Covariates}
    r = fitted.result
    β̄ = vec(mean(r.β_samples, dims=1))
    return (; (r.names .=> β̄)...)
end

"""
    inclusion_probabilities(fitted)

Posterior inclusion probabilities of each covariate from a [`Bayesian`](@ref)
[`Covariates`](@ref) fit with a [`SpikeSlabPrior`](@ref), as a named tuple keyed
by covariate name. Errors for priors that do not perform variable selection.
"""
function inclusion_probabilities(fitted::FittedComparativeModel{M, Bayesian, CovariateMCMCSamples}) where {M <: Covariates}
    r = fitted.result
    r.inclusion === nothing && throw(ArgumentError(
        "inclusion probabilities are only available for a SpikeSlabPrior fit"))
    pip = vec(mean(r.inclusion, dims=1))
    return (; (r.names .=> pip)...)
end

function probability(fitted::FittedComparativeModel{M, Bayesian, CovariateMCMCSamples},
                     i::Integer, j::Integer) where {M <: Covariates}
    r = fitted.result
    d = @view(r.Z[i, :]) .- @view(r.Z[j, :])      # p-vector
    ψ = r.β_samples * d                            # n_samples
    return mean(1.0 ./ (1.0 .+ exp.(-ψ)))
end

function probability(fitted::FittedComparativeModel{M, Bayesian, CovariateMCMCSamples, L},
                     item_i::L, item_j::L) where {M <: Covariates, L}
    idx_i = findfirst(==(item_i), fitted.labels)
    idx_j = findfirst(==(item_j), fitted.labels)
    idx_i === nothing && throw(ArgumentError("Label $(item_i) not found in fitted model"))
    idx_j === nothing && throw(ArgumentError("Label $(item_j) not found in fitted model"))
    return probability(fitted, idx_i, idx_j)
end
