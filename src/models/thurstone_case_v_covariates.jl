# ─── Thurstone Case V with covariates: λ_i = z_iᵀβ, probit link ──────────────
#
# Mirrors the covariate Bradley–Terry model but with a probit comparison
# likelihood: `probit P(i beats j) = Φ((z_i − z_j)ᵀβ)`. This is probit regression
# on the covariate-difference design D, so it reuses the aggregation
# (`_aggregate_covariate_pairs`), the stepwise scaffolding (`_ic`, `_subset_agg`)
# and the shrinkage machinery (`_init_shrinkage`, `_update_shrinkage!`); only the
# link — and hence the likelihood, Fisher information and Gibbs augmentation —
# changes.

# log p(wins | β) for the probit covariate model.
function _tcvcov_loglik(β::AbstractVector, agg::_AggregatedCovariateData)
    ll = 0.0
    @inbounds for q in 1:agg.P
        ψ = 0.0
        for k in 1:agg.p
            ψ += agg.D[q, k] * β[k]
        end
        y = agg.yvec[q]
        ll += y * _log_normcdf(ψ) + (agg.Nvec[q] - y) * _log_normcdf(-ψ)
    end
    return ll
end

# ─── Maximum likelihood ──────────────────────────────────────────────────────

_tcvcov_neg_loglik(β::AbstractVector, agg::_AggregatedCovariateData) = -_tcvcov_loglik(β, agg)

function _tcvcov_neg_grad!(G::AbstractVector, β::AbstractVector, agg::_AggregatedCovariateData)
    fill!(G, 0.0)
    @inbounds for q in 1:agg.P
        ψ = 0.0
        for k in 1:agg.p
            ψ += agg.D[q, k] * β[k]
        end
        y = agg.yvec[q]
        # d/dψ [y logΦ(ψ) + (N−y) logΦ(−ψ)] = y·(φ/Φ)(ψ) − (N−y)·(φ/Φ)(−ψ)
        score = y * _inv_mills(ψ) - (agg.Nvec[q] - y) * _inv_mills(-ψ)
        for k in 1:agg.p
            G[k] -= agg.D[q, k] * score
        end
    end
    return G
end

# Inverse Fisher information DᵀWD with the probit weight W_q = N_q·φ(ψ)²/(Φ(ψ)Φ(−ψ)).
function _tcvcov_vcov(β::AbstractVector, agg::_AggregatedCovariateData)
    W = Vector{Float64}(undef, agg.P)
    @inbounds for q in 1:agg.P
        ψ = 0.0
        for k in 1:agg.p
            ψ += agg.D[q, k] * β[k]
        end
        φ = _INV_SQRT_2π * exp(-0.5 * ψ^2)
        Φp = _normcdf(ψ); Φm = _normcdf(-ψ)
        W[q] = agg.Nvec[q] * φ^2 / max(Φp * Φm, 1e-12)
    end
    info = agg.D' * (W .* agg.D)
    try
        return Matrix(inv(Symmetric(info)))
    catch
        return Matrix(inv(Symmetric(info + 1e-8 * I)))
    end
end

# Core probit MLE solve on an (already-subsetted) aggregated representation.
function _fit_tcvcov_mle(agg::_AggregatedCovariateData)
    if agg.p == 0
        ll = -log(2.0) * sum(agg.Nvec)   # ψ ≡ 0 ⇒ Φ(0) = 0.5 for every comparison
        return (β = Float64[], vcov = zeros(0, 0), loglik = ll,
                converged = true, iterations = 0)
    end
    f(β) = _tcvcov_neg_loglik(β, agg)
    g!(G, β) = _tcvcov_neg_grad!(G, β, agg)
    res = optimize(f, g!, zeros(agg.p), LBFGS())
    β = Optim.minimizer(res)
    return (β = β, vcov = _tcvcov_vcov(β, agg), loglik = -Optim.minimum(res),
            converged = Optim.converged(res), iterations = Optim.iterations(res))
end

"""
    fit(model::Covariates{ThurstoneCaseV}, method::MLE, data::CovariateData)

Maximum-likelihood fit of the covariate Thurstone Case V model via L-BFGS: the
comparison link is `Φ((z_i − z_j)ᵀβ)`, so this is probit regression on the
covariate-difference design. [`coef`](@ref) returns the estimated β and
[`strengths`](@ref) the recovered latent strengths `λ = Zβ`.
"""
function fit(model::Covariates{ThurstoneCaseV}, method::MLE, cd::CovariateData{L}) where {L}
    K = length(cd.data.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit covariate ThurstoneCaseV, got $K"))
    agg = _aggregate_covariate_pairs(cd)
    agg.P >= 1 || throw(ArgumentError("No pairs with comparison data to fit"))
    fr = _fit_tcvcov_mle(agg)
    result = CovariateMLEResult(fr.β, fr.vcov, fr.loglik, cd.Z, cd.names,
                                collect(1:agg.p), NamedTuple[])
    return FittedComparativeModel(model, method, result, cd.data.labels, cd,
                                  fr.converged, fr.iterations)
end

function fit(model::Covariates{ThurstoneCaseV}, cd::CovariateData)
    return fit(model, MLE(), cd)
end

"""
    fit(model::Covariates{ThurstoneCaseV}, method::StepwiseMLE, data::CovariateData)

Stepwise maximum-likelihood selection of covariates by AIC or BIC for the probit
Thurstone Case V model (see [`StepwiseMLE`](@ref)). Greedily adds and/or removes
covariates until the information criterion can no longer be improved, then returns
the fit of the selected subset; query with [`coef`](@ref) and
[`strengths`](@ref).
"""
function fit(model::Covariates{ThurstoneCaseV}, method::StepwiseMLE, cd::CovariateData{L}) where {L}
    K = length(cd.data.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit covariate ThurstoneCaseV, got $K"))
    agg = _aggregate_covariate_pairs(cd)
    agg.P >= 1 || throw(ArgumentError("No pairs with comparison data to fit"))
    p = agg.p
    n = sum(agg.Nvec)
    allow_add = method.direction in (:forward, :both)
    allow_remove = method.direction in (:backward, :both)

    fit_subset(cols) = _fit_tcvcov_mle(_subset_agg(agg, cols))

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
    return FittedComparativeModel(model, method, result, cd.data.labels, cd, cur.converged, step)
end

# Thurstone-specific MLE/Stepwise probability (probit link). Strengths,
# coefficients, std and intervals are link-agnostic and inherited from the
# generic covariate accessors.
function probability(fitted::FittedComparativeModel{Covariates{ThurstoneCaseV}, I, CovariateMLEResult},
                     i::Integer, j::Integer) where {I}
    r = fitted.result
    d = 0.0
    for (c, k) in enumerate(r.selected)
        d += (r.Z[i, k] - r.Z[j, k]) * r.β[c]
    end
    return _normcdf(d)
end

# ─── Bayesian: Albert–Chib truncated-normal augmented Gibbs on β ──────────────
#
# Per sweep: draw latent utilities u | β, then β | u from its Gaussian full
# conditional with precision `Binv + DᵀNᵈⁱᵃᵍD` (constant comparison part) using
# the same shrinkage state (`_update_shrinkage!`) as the Bradley–Terry covariate
# sampler — only the augmentation differs.
function _probit_covariate_gibbs(agg::_AggregatedCovariateData, prior::AbstractPrior,
                                 method::Bayesian, rng::AbstractRNG)
    p = agg.p
    P = agg.P
    DtND = agg.D' * (Float64.(agg.Nvec) .* agg.D)   # p×p, constant
    state = _init_shrinkage(prior, p)
    record_incl = _records_inclusion(state)

    total = method.n_burnin + method.thin * method.n_samples
    β_samples = Matrix{Float64}(undef, method.n_samples, p)
    lls       = Vector{Float64}(undef, method.n_samples)
    incl      = record_incl ? Matrix{Float64}(undef, method.n_samples, p) : nothing

    β = zeros(p)
    ψ = Vector{Float64}(undef, P)
    g = Vector{Float64}(undef, p)     # Dᵀu accumulator
    z = Vector{Float64}(undef, p)

    for s in 1:total
        # latent utilities u | β, summed per pair into g = Dᵀu
        mul!(ψ, agg.D, β)
        fill!(g, 0.0)
        @inbounds for q in 1:P
            μ = ψ[q]; Nq = agg.Nvec[q]; yq = agg.yvec[q]
            S = 0.0
            for _ in 1:yq
                S += _sample_truncated_normal(rng, μ, true)
            end
            for _ in 1:(Nq - yq)
                S += _sample_truncated_normal(rng, μ, false)
            end
            for k in 1:p
                g[k] += agg.D[q, k] * S
            end
        end

        # shrinkage hyperparameters | β
        _update_shrinkage!(state, prior, β, rng)

        # β | u : precision V = Binv + DᵀNᵈⁱᵃᵍD
        V = Matrix(state.Binv) .+ DtND
        @inbounds for k in 1:p
            V[k, k] += 1e-10
        end
        C = cholesky!(Symmetric(V))
        m = C \ (g .+ state.Binv_μ)
        randn!(rng, z)
        ldiv!(C.U, z)
        β = m .+ z

        if s > method.n_burnin && (s - method.n_burnin) % method.thin == 0
            idx = (s - method.n_burnin) ÷ method.thin
            β_samples[idx, :] .= β
            lls[idx] = _tcvcov_loglik(β, agg)
            record_incl && (incl[idx, :] .= state.γ)
        end
    end
    return β_samples, lls, incl
end

"""
    fit(model::Covariates{ThurstoneCaseV}, method::Bayesian, data::CovariateData,
        [prior]; rng=Random.default_rng())

Bayesian fit of the covariate Thurstone Case V model by Albert–Chib augmented
Gibbs sampling of the coefficients β. `prior` is one of [`NormalPrior`](@ref)
(default `NormalPrior(p)`), [`HorseshoePrior`](@ref) for global-local shrinkage,
or [`SpikeSlabPrior`](@ref) for variable selection with posterior inclusion
probabilities. The result holds posterior draws ([`CovariateMCMCSamples`](@ref));
query with [`coef`](@ref), [`strengths`](@ref), [`posterior_mean`](@ref),
[`credible_interval`](@ref), [`inclusion_probabilities`](@ref).
"""
function fit(model::Covariates{ThurstoneCaseV}, method::Bayesian, cd::CovariateData{L},
             prior::AbstractPrior; rng::AbstractRNG=Random.default_rng()) where {L}
    K = length(cd.data.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit covariate ThurstoneCaseV, got $K"))
    agg = _aggregate_covariate_pairs(cd)
    agg.P >= 1 || throw(ArgumentError("No pairs with comparison data to fit"))
    β_samples, lls, incl = _probit_covariate_gibbs(agg, prior, method, rng)
    result = CovariateMCMCSamples(β_samples, lls, incl, cd.Z, cd.names,
                                  method.n_samples, method.n_burnin, method.thin)
    total = method.n_burnin + method.thin * method.n_samples
    return FittedComparativeModel(model, method, result, cd.data.labels, cd, true, total)
end

function fit(model::Covariates{ThurstoneCaseV}, method::Bayesian, cd::CovariateData{L};
             rng::AbstractRNG=Random.default_rng()) where {L}
    return fit(model, method, cd, NormalPrior(size(cd.Z, 2)); rng=rng)
end

# Thurstone-specific Bayesian probability (probit link).
function probability(fitted::FittedComparativeModel{Covariates{ThurstoneCaseV}, Bayesian, CovariateMCMCSamples},
                     i::Integer, j::Integer)
    r = fitted.result
    d = @view(r.Z[i, :]) .- @view(r.Z[j, :])
    ψ = r.β_samples * d
    return mean(_normcdf.(ψ))
end
