# ─── Anchored Bradley-Terry with covariates: λ_i = z_iᵀβ + anchor calibration ──
#
# Composes the two wrappers. Strengths are covariate-driven (λ = Zβ, the
# covariate model) and an anchor layer y_k = a + b·λ_k + ε calibrates the latent
# scale (the anchored model). Estimation is fully joint over (β, a, b, σ²):
#
#   logit P(i beats j) = (z_i − z_j)ᵀβ              # comparisons
#   y_k = a + b·(z_kᵀβ) + ε,  ε ~ N(0, σ²),  k ∈ S   # anchors
#
# β is identified by the comparisons (constant covariate columns are rejected by
# CovariateData), so λ = Zβ is used UNCENTRED — its absolute scale is what (a, b)
# calibrate to, which lets the fit predict measurements for unmeasured items.
#
# The Bayesian Gibbs sampler is the covariate β-update with two extra terms from
# the anchor layer, then the anchored (a, b) and σ² updates verbatim. The MLE is
# the covariate logistic regression with σ² profiled and the anchor (a, b) profiled
# by OLS, optimised over β alone (envelope-theorem gradient).

# Group-averaged anchor design from an `AnchoredData{CovariateData}`. Each anchor g
# targets a group G_g of items; its mean strength is z̄_gᵀβ with z̄_g = mean(Z[G_g,:]),
# so the anchor layer is a weighted regression of y on the group-averaged covariates
# Zbar (G×p) with precision weights n_g = |G_g| (variance σ²/n_g). Singletons give
# n_g = 1 and z̄_g = z_g, recovering the per-item model exactly.
function _anchor_design(data::AnchoredData{<:CovariateData})
    cd = data.data
    groups = data.anchor_groups
    y = data.anchor_values
    G = length(groups)
    p = size(cd.Z, 2)
    Zbar = Matrix{Float64}(undef, G, p)
    ng = Vector{Float64}(undef, G)
    @inbounds for (g, grp) in enumerate(groups)
        ng[g] = length(grp)
        for c in 1:p
            acc = 0.0
            for i in grp
                acc += cd.Z[i, c]
            end
            Zbar[g, c] = acc / ng[g]
        end
    end
    return (cd = cd, groups = groups, y = y, Zbar = Zbar, ng = ng)
end

# ─── Maximum likelihood (joint, concentrated over β) ─────────────────────────

# Weighted OLS of y on [1 μ] (weights w = n_g) and the MLE σ² = (Σ w·resid²)/G.
# Falls back to (weighted mean(y), 0) when the regression is degenerate (G < 2, or
# μ has no spread, or no covariates). With w ≡ 1 this is ordinary OLS.
function _profile_calibration(μ::AbstractVector, y::AbstractVector, w::AbstractVector)
    G = length(y)
    sw = 0.0; swμ = 0.0; swμμ = 0.0; swy = 0.0; swμy = 0.0
    @inbounds for g in 1:G
        wg = w[g]; μg = μ[g]; yg = y[g]
        sw += wg; swμ += wg * μg; swμμ += wg * μg^2; swy += wg * yg; swμy += wg * μg * yg
    end
    denom = sw * swμμ - swμ^2
    if G >= 2 && denom > 1e-10
        b = (sw * swμy - swμ * swy) / denom
        a = (swy - b * swμ) / sw
    else
        a = swy / sw
        b = 0.0
    end
    rss = 0.0
    @inbounds for g in 1:G
        rss += w[g] * (y[g] - a - b * μ[g])^2
    end
    σ² = max(rss / G, 1e-8)
    return a, b, σ², rss
end

# Concentrated joint negative log-likelihood at β (a, b, σ² profiled out). The anchor
# layer is a weighted Gaussian (weights ng, variance σ²/n_g); the −½Σ log n_g term is
# constant in β but kept so the reported log-likelihood is a proper density.
function _anchored_cov_neg_loglik(β::AbstractVector, agg::_AggregatedCovariateData,
                                  Zbar::AbstractMatrix, y::AbstractVector,
                                  ng::AbstractVector)
    μ = Zbar * β
    _, _, σ², rss = _profile_calibration(μ, y, ng)
    G = length(y)
    return _btcov_neg_loglik(β, agg) +
           0.5 * (G * log(2π * σ²) - sum(log, ng) + rss / σ²)
end

# Gradient of the concentrated objective. By the envelope theorem (a, b, σ² are at
# their conditional optima) this equals the partial gradient holding them fixed:
# comparison gradient − (b/σ²)·Zbarᵀ W e, with W = diag(n_g).
function _anchored_cov_neg_grad!(G::AbstractVector, β::AbstractVector,
                                 agg::_AggregatedCovariateData,
                                 Zbar::AbstractMatrix, y::AbstractVector,
                                 ng::AbstractVector)
    _btcov_neg_grad!(G, β, agg)
    μ = Zbar * β
    a, b, σ², _ = _profile_calibration(μ, y, ng)
    e = ng .* (y .- a .- b .* μ)              # W e
    G .+= (-b / σ²) .* (Zbar' * e)
    return G
end

# Joint observed (Fisher) information over (β, a, b); σ² is orthogonal at the MLE.
# Anchor contributions are weighted by W = diag(n_g). Returns the β-block of the inverse.
function _anchored_cov_vcov(β::AbstractVector, agg::_AggregatedCovariateData,
                            Zbar::AbstractMatrix, y::AbstractVector, ng::AbstractVector,
                            a::Float64, b::Float64, σ²::Float64)
    p = agg.p
    # comparison Fisher information on β: DᵀWD, W = N·μ(1−μ)
    Wc = Vector{Float64}(undef, agg.P)
    @inbounds for q in 1:agg.P
        ψ = 0.0
        for k in 1:p
            ψ += agg.D[q, k] * β[k]
        end
        μq = 1.0 / (1.0 + exp(-ψ))
        Wc[q] = agg.Nvec[q] * μq * (1.0 - μq)
    end
    info_cmp = agg.D' * (Wc .* agg.D)
    μ = Zbar * β
    Wμ = ng .* μ
    M = zeros(p + 2, p + 2)
    M[1:p, 1:p] .= info_cmp .+ (b^2 / σ²) .* (Zbar' * (ng .* Zbar))
    βa = (b / σ²) .* (Zbar' * ng)             # ∂²/∂β∂a contribution: Zbarᵀ W 1
    βb = (b / σ²) .* (Zbar' * Wμ)             # ∂²/∂β∂b contribution: Zbarᵀ W μ
    @inbounds for k in 1:p
        M[k, p + 1] = βa[k]; M[p + 1, k] = βa[k]
        M[k, p + 2] = βb[k]; M[p + 2, k] = βb[k]
    end
    M[p + 1, p + 1] = sum(ng) / σ²
    M[p + 1, p + 2] = sum(Wμ) / σ²; M[p + 2, p + 1] = M[p + 1, p + 2]
    M[p + 2, p + 2] = dot(Wμ, μ) / σ²
    Vfull = try
        Matrix(inv(Symmetric(M)))
    catch
        Matrix(inv(Symmetric(M + 1e-8 * I)))
    end
    return Vfull[1:p, 1:p]
end

# Core joint MLE on an (already-subsetted) aggregated representation + group-averaged
# anchor rows Zbar (G×p) with weights ng.
function _fit_anchored_covariate_mle(agg::_AggregatedCovariateData,
                                     Zbar::AbstractMatrix, y::AbstractVector,
                                     ng::AbstractVector)
    G = length(y)
    if agg.p == 0
        ll_cmp = -log(2.0) * sum(agg.Nvec)        # ψ ≡ 0 ⇒ coin flips
        a, b, σ², rss = _profile_calibration(zeros(G), y, ng)   # μ ≡ 0 (null model)
        ll_anchor = -0.5 * (G * log(2π * σ²) - sum(log, ng) + rss / σ²)
        return (β = Float64[], a = a, b = b, σ² = σ², vcov = zeros(0, 0),
                loglik = ll_cmp + ll_anchor, converged = true, iterations = 0)
    end
    f(β) = _anchored_cov_neg_loglik(β, agg, Zbar, y, ng)
    g!(G_, β) = _anchored_cov_neg_grad!(G_, β, agg, Zbar, y, ng)
    res = optimize(f, g!, zeros(agg.p), LBFGS())
    β = Optim.minimizer(res)
    μ = Zbar * β
    a, b, σ², _ = _profile_calibration(μ, y, ng)
    vcov = _anchored_cov_vcov(β, agg, Zbar, y, ng, a, b, σ²)
    return (β = β, a = a, b = b, σ² = σ², vcov = vcov, loglik = -Optim.minimum(res),
            converged = Optim.converged(res), iterations = Optim.iterations(res))
end

"""
    fit(model::Anchored{Covariates{BradleyTerry}}, method::MLE, data::AnchoredData)

Joint maximum-likelihood fit of the anchored covariate Bradley–Terry model: the
covariate coefficients β, calibration `(a, b)` and anchor variance σ² maximise the
joint likelihood of the comparisons and the anchor measurements. [`coefficients`](@ref)
returns β, [`calibration`](@ref) the `(a, b, σ²)`, and [`predict`](@ref) anchor
measurements (including for unseen items, from a covariate vector).
"""
function fit(model::Anchored{Covariates{BradleyTerry}}, method::MLE,
             data::AnchoredData{CovariateData{L}, L}) where {L}
    cd = data.data
    K = length(cd.data.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit BradleyTerryCovariatesAnchored, got $K"))
    agg = _aggregate_covariate_pairs(cd)
    agg.P >= 1 || throw(ArgumentError("No pairs with comparison data to fit"))
    des = _anchor_design(data)
    fr = _fit_anchored_covariate_mle(agg, des.Zbar, des.y, des.ng)
    result = AnchoredCovariateMLEResult(fr.β, fr.vcov, fr.a, fr.b, fr.σ², fr.loglik,
                                        cd.Z, cd.names, collect(1:agg.p), NamedTuple[])
    return FittedComparativeModel(model, method, result, cd.data.labels,
                                  fr.converged, fr.iterations)
end

function fit(model::Anchored{Covariates{BradleyTerry}}, data::AnchoredData{CovariateData{L}, L};
             rng::AbstractRNG=Random.default_rng()) where {L}
    return fit(model, Bayesian(), data; rng=rng)
end

# ─── Stepwise selection (joint loglik, AIC/BIC) ──────────────────────────────

"""
    fit(model::Anchored{Covariates{BradleyTerry}}, method::StepwiseMLE, data::AnchoredData)

Stepwise maximum-likelihood covariate selection for the anchored covariate model
(see [`StepwiseMLE`](@ref)). Greedily adds/removes covariates to optimise the joint
information criterion, then refits the selected subset. The selected indices and the
search trace are recorded in the result.
"""
function fit(model::Anchored{Covariates{BradleyTerry}}, method::StepwiseMLE,
             data::AnchoredData{CovariateData{L}, L}) where {L}
    cd = data.data
    K = length(cd.data.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit BradleyTerryCovariatesAnchored, got $K"))
    agg = _aggregate_covariate_pairs(cd)
    agg.P >= 1 || throw(ArgumentError("No pairs with comparison data to fit"))
    des = _anchor_design(data); y = des.y; ng = des.ng; Zbar = des.Zbar
    p = agg.p
    n = sum(agg.Nvec)
    allow_add = method.direction in (:forward, :both)
    allow_remove = method.direction in (:backward, :both)

    # Column-subsetting commutes with group averaging, so subset the columns of Zbar.
    fit_subset(cols) = _fit_anchored_covariate_mle(_subset_agg(agg, cols), Zbar[:, cols], y, ng)
    npar(cols) = length(cols) + 3      # β subset + (a, b, σ²)

    selected = method.direction === :backward ? collect(1:p) : Int[]
    cur = fit_subset(selected)
    cur_ic = _ic(cur.loglik, npar(selected), n, method.criterion)
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
                ic = _ic(fr.loglik, npar(cand), n, method.criterion)
                if ic < best_ic - 1e-8
                    best_ic = ic; best_fit = fr; best_sel = cand
                end
            end
        end
        if allow_remove
            for c in selected
                cand = filter(!=(c), selected)
                fr = fit_subset(cand)
                ic = _ic(fr.loglik, npar(cand), n, method.criterion)
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

    result = AnchoredCovariateMLEResult(cur.β, cur.vcov, cur.a, cur.b, cur.σ², cur.loglik,
                                        cd.Z, cd.names, selected, trace)
    return FittedComparativeModel(model, method, result, cd.data.labels, cur.converged, step)
end

# ─── Bayesian: joint PG Gibbs, β prior dispatched on the shrinkage hook ───────

function _pg_anchored_cov_gibbs(agg::_AggregatedCovariateData, Z::Matrix{Float64},
                                Zbar::Matrix{Float64}, ng::Vector{Float64},
                                y::Vector{Float64},
                                prior::AnchoredCovariatePrior, method::Bayesian,
                                rng::AbstractRNG)
    p = agg.p
    P = agg.P
    K = size(Z, 1)
    G = length(y)
    sum_log_ng = sum(log, ng)
    WZbar = ng .* Zbar                       # W Zbar, W = diag(n_g)
    ZbartWZbar = Zbar' * WZbar               # (b²/σ²) coefficient block
    Dtκ = agg.D' * agg.κ
    state = _init_shrinkage(prior.β_prior, p)
    record_incl = _records_inclusion(state)

    V₀_inv = inv(prior.calib_prior.Σ)
    V₀_inv_β₀ = V₀_inv * prior.calib_prior.μ
    α₀, b₀ = prior.σ²_prior.α, prior.σ²_prior.β

    total = method.n_burnin + method.thin * method.n_samples
    βcov_samples = Matrix{Float64}(undef, method.n_samples, p)
    λ_samples    = Matrix{Float64}(undef, method.n_samples, K)
    βcal_samples = Matrix{Float64}(undef, method.n_samples, 2)
    σ²_samples   = Vector{Float64}(undef, method.n_samples)
    lls          = Vector{Float64}(undef, method.n_samples)
    incl         = record_incl ? Matrix{Float64}(undef, method.n_samples, p) : nothing

    # Initialise β from the covariate-only MLE, then (a, b, σ²) from OLS of y on the
    # group-mean strengths μ = Zbar β.
    fr = _fit_covariate_mle(agg)
    β = fr.converged && p > 0 ? copy(fr.β) : zeros(p)
    λ = Z * β
    μ = Zbar * β
    a, b, σ² = _anchored_init_β(μ, y, AnchoredPrior(σ²_prior = prior.σ²_prior))

    ψ = Vector{Float64}(undef, P)
    ω = Vector{Float64}(undef, P)
    znoise = Vector{Float64}(undef, p)

    for s in 1:total
        # ω | β
        mul!(ψ, agg.D, β)
        @inbounds for q in 1:P
            ω[q] = _sample_pg(rng, agg.Nvec[q], ψ[q])
        end

        # shrinkage hyperparameters | β
        _update_shrinkage!(state, prior.β_prior, β, rng)

        # β | ω, a, b, σ² : V⁻¹ = Binv + DᵀΩD + (b²/σ²)·ZbarᵀWZbar
        V = Matrix(state.Binv) .+ agg.D' * (ω .* agg.D) .+ (b^2 / σ²) .* ZbartWZbar
        rhs = (Dtκ .+ state.Binv_μ) .+ (b / σ²) .* (WZbar' * (y .- a))
        @inbounds for k in 1:p
            V[k, k] += 1e-10
        end
        C = cholesky!(Symmetric(V))
        m = C \ rhs
        randn!(rng, znoise)
        ldiv!(C.U, znoise)
        β = m .+ znoise
        λ = Z * β
        μ = Zbar * β

        # (a, b) | μ, σ² — conjugate 2×2 weighted regression (weights ρ_g = n_g)
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
        σ = sqrt(σ²)
        z1 = randn(rng); z2 = randn(rng)
        a = μa + σ * L11 * z1
        b = μb + σ * (L21 * z1 + L22 * z2)

        # σ² | μ, a, b — inverse-gamma on the weighted anchor RSS (G groups, weights n_g)
        rss = 0.0
        @inbounds for g in 1:G
            rss += ng[g] * (y[g] - a - b * μ[g])^2
        end
        σ² = _sample_inv_gamma(rng, α₀ + G / 2, b₀ + rss / 2)

        if s > method.n_burnin && (s - method.n_burnin) % method.thin == 0
            idx = (s - method.n_burnin) ÷ method.thin
            βcov_samples[idx, :] .= β
            λ_samples[idx, :] .= λ
            βcal_samples[idx, 1] = a
            βcal_samples[idx, 2] = b
            σ²_samples[idx] = σ²
            lls[idx] = _btcov_loglik(β, agg) -
                       0.5 * (G * log(2π * σ²) - sum_log_ng + rss / σ²)
            record_incl && (incl[idx, :] .= state.γ)
        end
    end
    return λ_samples, βcal_samples, σ²_samples, lls, βcov_samples, incl
end

"""
    fit(model::Anchored{Covariates{BradleyTerry}}, method::Bayesian, data::AnchoredData,
        [prior]; rng=Random.default_rng())

Joint Bayesian fit of the anchored covariate Bradley–Terry model by Pólya-Gamma
augmented Gibbs sampling. `prior` is an [`AnchoredCovariatePrior`](@ref) (a β
shrinkage prior plus calibration/variance priors), or just a bare β prior
([`NormalPrior`](@ref) — the default, [`HorseshoePrior`](@ref) or
[`SpikeSlabPrior`](@ref)) which is wrapped with default calibration priors. The
result holds posterior draws ([`AnchoredCovariateMCMCSamples`](@ref)); query with
[`coefficients`](@ref), [`calibration`](@ref), [`strengths`](@ref),
[`predict`](@ref), [`credible_interval`](@ref), [`inclusion_probabilities`](@ref).
"""
function fit(model::Anchored{Covariates{BradleyTerry}}, method::Bayesian,
             data::AnchoredData{CovariateData{L}, L},
             prior::AnchoredCovariatePrior; rng::AbstractRNG=Random.default_rng()) where {L}
    cd = data.data
    K = length(cd.data.labels)
    K >= 2 || throw(ArgumentError("Need at least 2 items to fit BradleyTerryCovariatesAnchored, got $K"))
    agg = _aggregate_covariate_pairs(cd)
    agg.P >= 1 || throw(ArgumentError("No pairs with comparison data to fit"))
    des = _anchor_design(data)
    λs, βcal, σ²s, lls, βcov, incl =
        _pg_anchored_cov_gibbs(agg, cd.Z, des.Zbar, des.ng, des.y, prior, method, rng)
    result = AnchoredCovariateMCMCSamples(λs, βcal, σ²s, lls, βcov, incl, cd.Z, cd.names,
                                          method.n_samples, method.n_burnin, method.thin)
    total = method.n_burnin + method.thin * method.n_samples
    return FittedComparativeModel(model, method, result, cd.data.labels, true, total)
end

# Bare β prior → wrap with default calibration/variance priors.
function fit(model::Anchored{Covariates{BradleyTerry}}, method::Bayesian,
             data::AnchoredData{CovariateData{L}, L}, βprior::AbstractPrior;
             rng::AbstractRNG=Random.default_rng()) where {L}
    return fit(model, method, data, AnchoredCovariatePrior(βprior); rng=rng)
end

function fit(model::Anchored{Covariates{BradleyTerry}}, method::Bayesian,
             data::AnchoredData{CovariateData{L}, L};
             rng::AbstractRNG=Random.default_rng()) where {L}
    return fit(model, method, data, NormalPrior(size(data.data.Z, 2)); rng=rng)
end

# ─── MLE / Stepwise accessors (dispatch on AnchoredCovariateMLEResult) ───────

function coefficients(fitted::FittedComparativeModel{M, I, AnchoredCovariateMLEResult}) where {M, I}
    r = fitted.result
    return (; (r.names[r.selected] .=> r.β)...)
end

function coefficient_std(fitted::FittedComparativeModel{M, I, AnchoredCovariateMLEResult}) where {M, I}
    r = fitted.result
    se = [sqrt(r.vcov[k, k]) for k in 1:length(r.β)]
    return (; (r.names[r.selected] .=> se)...)
end

function coefficient_intervals(fitted::FittedComparativeModel{M, I, AnchoredCovariateMLEResult};
                               level::Float64=0.95) where {M, I}
    0.0 < level < 1.0 || throw(ArgumentError("level must be in (0, 1), got $level"))
    r = fitted.result
    z = _norm_quantile(1.0 - (1.0 - level) / 2.0)
    ints = [(r.β[k] - z * sqrt(r.vcov[k, k]), r.β[k] + z * sqrt(r.vcov[k, k]))
            for k in 1:length(r.β)]
    return (; (r.names[r.selected] .=> ints)...)
end

# Latent strengths λ = Zβ (uncentred — the scale (a, b) calibrate to).
function strengths(fitted::FittedComparativeModel{M, I, AnchoredCovariateMLEResult}) where {M, I}
    r = fitted.result
    return isempty(r.selected) ? zeros(size(r.Z, 1)) : r.Z[:, r.selected] * r.β
end

function calibration(fitted::FittedComparativeModel{M, I, AnchoredCovariateMLEResult}) where {M, I}
    r = fitted.result
    return (a = r.a, b = r.b, σ² = r.σ²)
end

function loglikelihood(fitted::FittedComparativeModel{M, I, AnchoredCovariateMLEResult}) where {M, I}
    return fitted.result.loglik
end

function probability(fitted::FittedComparativeModel{M, I, AnchoredCovariateMLEResult},
                     i::Integer, j::Integer) where {M, I}
    r = fitted.result
    d = 0.0
    for (c, k) in enumerate(r.selected)
        d += (r.Z[i, k] - r.Z[j, k]) * r.β[c]
    end
    return 1.0 / (1.0 + exp(-d))
end

function probability(fitted::FittedComparativeModel{M, I, AnchoredCovariateMLEResult, L},
                     item_i::L, item_j::L) where {M, I, L}
    idx_i = findfirst(==(item_i), fitted.labels)
    idx_j = findfirst(==(item_j), fitted.labels)
    idx_i === nothing && throw(ArgumentError("Label $(item_i) not found in fitted model"))
    idx_j === nothing && throw(ArgumentError("Label $(item_j) not found in fitted model"))
    return probability(fitted, idx_i, idx_j)
end

# Predicted anchor measurement ŷ = a + b·λ. Point estimate, or a normal interval
# from σ̂² when `prob` is given.
function _mle_predict_point(r::AnchoredCovariateMLEResult, λ::Float64, prob)
    μ = r.a + r.b * λ
    prob === nothing && return μ
    0.0 < prob < 1.0 || throw(ArgumentError("prob must be in (0, 1), got $prob"))
    z = _norm_quantile(1.0 - (1.0 - prob) / 2.0)
    sd = sqrt(r.σ²)
    return (μ - z * sd, μ + z * sd)
end

function predict(fitted::FittedComparativeModel{M, I, AnchoredCovariateMLEResult}) where {M, I}
    return fitted.result.a .+ fitted.result.b .* strengths(fitted)
end

function predict(fitted::FittedComparativeModel{M, I, AnchoredCovariateMLEResult},
                 k::Integer; prob::Union{Nothing, Float64}=nothing) where {M, I}
    r = fitted.result
    λk = isempty(r.selected) ? 0.0 : dot(Float64.(@view r.Z[k, r.selected]), r.β)
    return _mle_predict_point(r, λk, prob)
end

function predict(fitted::FittedComparativeModel{M, I, AnchoredCovariateMLEResult, L},
                 label::L; prob::Union{Nothing, Float64}=nothing) where {M, I, L}
    idx = findfirst(==(label), fitted.labels)
    idx === nothing && throw(ArgumentError("Label $(label) not found in fitted model"))
    return predict(fitted, idx; prob=prob)
end

# Prediction for an unseen item from its covariate vector z (length = #covariates).
function predict(fitted::FittedComparativeModel{M, I, AnchoredCovariateMLEResult},
                 z::AbstractVector{<:Real}; prob::Union{Nothing, Float64}=nothing) where {M, I}
    r = fitted.result
    length(z) == size(r.Z, 2) || throw(DimensionMismatch(
        "covariate vector has length $(length(z)), expected $(size(r.Z, 2))"))
    λnew = isempty(r.selected) ? 0.0 : dot(Float64.(z[r.selected]), r.β)
    return _mle_predict_point(r, λnew, prob)
end

# ─── Bayesian accessors (dispatch on AnchoredCovariateMCMCSamples) ───────────
# posterior_mean / posterior_std / credible_interval / loglikelihood / strengths /
# calibration / predict(fitted) / predict(fitted, k) / predict(fitted, label) are
# inherited from the model-agnostic `{<:Anchored, Bayesian}` accessors, which read
# the matching λ_samples / β_samples / σ²_samples fields.

function coefficients(fitted::FittedComparativeModel{M, Bayesian, AnchoredCovariateMCMCSamples}) where {M}
    r = fitted.result
    β = vec(mean(r.βcov_samples, dims=1))
    return (; (r.names .=> β)...)
end

function coefficient_std(fitted::FittedComparativeModel{M, Bayesian, AnchoredCovariateMCMCSamples}) where {M}
    r = fitted.result
    sd = vec(std(r.βcov_samples, dims=1))
    return (; (r.names .=> sd)...)
end

function coefficient_intervals(fitted::FittedComparativeModel{M, Bayesian, AnchoredCovariateMCMCSamples};
                               level::Float64=0.95) where {M}
    0.0 < level < 1.0 || throw(ArgumentError("level must be in (0, 1), got $level"))
    r = fitted.result
    α = (1.0 - level) / 2.0
    ints = [(quantile(r.βcov_samples[:, k], α), quantile(r.βcov_samples[:, k], 1.0 - α))
            for k in 1:size(r.βcov_samples, 2)]
    return (; (r.names .=> ints)...)
end

function inclusion_probabilities(fitted::FittedComparativeModel{M, Bayesian, AnchoredCovariateMCMCSamples}) where {M}
    r = fitted.result
    r.inclusion === nothing && throw(ArgumentError(
        "inclusion probabilities are only available for a SpikeSlabPrior fit"))
    pip = vec(mean(r.inclusion, dims=1))
    return (; (r.names .=> pip)...)
end

function probability(fitted::FittedComparativeModel{M, Bayesian, AnchoredCovariateMCMCSamples},
                     i::Integer, j::Integer) where {M}
    Sλ = fitted.result.λ_samples
    return mean(1.0 ./ (1.0 .+ exp.(-(Sλ[:, i] .- Sλ[:, j]))))
end

function probability(fitted::FittedComparativeModel{M, Bayesian, AnchoredCovariateMCMCSamples, L},
                     item_i::L, item_j::L) where {M, L}
    idx_i = findfirst(==(item_i), fitted.labels)
    idx_j = findfirst(==(item_j), fitted.labels)
    idx_i === nothing && throw(ArgumentError("Label $(item_i) not found in fitted model"))
    idx_j === nothing && throw(ArgumentError("Label $(item_j) not found in fitted model"))
    return probability(fitted, idx_i, idx_j)
end

# Posterior-predictive measurement for an unseen item from its covariate vector z:
# y* = a + b·(zᵀβ) + ε, drawn over the posterior. Returns draws, or the `prob`
# credible interval.
function predict(fitted::FittedComparativeModel{M, Bayesian, AnchoredCovariateMCMCSamples},
                 z::AbstractVector{<:Real}; prob::Union{Nothing, Float64}=nothing,
                 rng::AbstractRNG=Random.default_rng()) where {M}
    r = fitted.result
    length(z) == size(r.βcov_samples, 2) || throw(DimensionMismatch(
        "covariate vector has length $(length(z)), expected $(size(r.βcov_samples, 2))"))
    λnew = r.βcov_samples * collect(Float64.(z))      # n_samples
    draws = r.β_samples[:, 1] .+ r.β_samples[:, 2] .* λnew .+
            sqrt.(r.σ²_samples) .* randn(rng, r.n_samples)
    prob === nothing && return draws
    α = (1.0 - prob) / 2.0
    return (quantile(draws, α), quantile(draws, 1.0 - α))
end
