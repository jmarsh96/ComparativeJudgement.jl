# ─── Pointwise log-likelihood, parameter counts, observation counts ──────────
#
# WAIC and PSIS-LOO need the log density of each observation evaluated at every
# posterior draw. The samplers store only the *total* log-likelihood per draw, so
# we reconstruct the per-observation matrix from the stored parameter draws and
# the original `data` (which the fitted object does not retain). The observation
# unit is one observed pair as a binomial (`Nₚ` trials, `yₚ` wins); the
# parameter-free `log C(N, y)` constant is omitted, matching the model
# log-likelihood helpers (`_bt_loglik`, …). For anchored fits only the
# comparisons enter (the anchor terms are excluded), so WAIC/LOO stay comparable
# to the unanchored fit.

# Per-pair binomial log densities under the two links.
_pair_logit(d::Float64, y::Int, N::Int)::Float64 =
    y * (-log1pexp(-d)) + (N - y) * (-log1pexp(d))
_pair_probit(d::Float64, y::Int, N::Int)::Float64 =
    y * _log_normcdf(d) + (N - y) * _log_normcdf(-d)

# The link of a (possibly wrapped) pairwise model.
_pair_link(::BradleyTerry) = _pair_logit
_pair_link(::ThurstoneCaseV) = _pair_probit
_pair_link(m::Anchored) = _pair_link(m.model)
_pair_link(m::Covariates) = _pair_link(m.model)
_pair_link(m::Intransitive) = _pair_link(m.model)

# Bayesian: S×P pointwise matrix from an S×K matrix of latent-strength draws.
function _pointwise_pairs(Λ::AbstractMatrix, agg::_AggregatedPairData, link)
    S = size(Λ, 1)
    out = Matrix{Float64}(undef, S, agg.P)
    @inbounds for p in 1:agg.P
        i, j = agg.pairs[p]
        N = agg.Nvec[p]
        y = round(Int, agg.κ[p] + N / 2)
        for s in 1:S
            out[s, p] = link(Λ[s, i] - Λ[s, j], y, N)
        end
    end
    return out
end

# MLE: length-P pointwise vector from a single latent-strength vector.
function _pointwise_pairs(λ::AbstractVector, agg::_AggregatedPairData, link)
    out = Vector{Float64}(undef, agg.P)
    @inbounds for p in 1:agg.P
        i, j = agg.pairs[p]
        N = agg.Nvec[p]
        y = round(Int, agg.κ[p] + N / 2)
        out[p] = link(λ[i] - λ[j], y, N)
    end
    return out
end

"""
    pointwise_loglikelihood(fitted, data)

Per-observation log-likelihood of `fitted` on the `data` it was fit to. Returns
an `n_draws × n_obs` matrix for a [`Bayesian`](@ref) fit and a length-`n_obs`
vector (at the point estimate) for an [`MLE`](@ref) fit. The observation unit is
each observed pair as a binomial; for anchored fits only the comparison terms
are included. This is the input to [`waic`](@ref) and [`loo`](@ref).
"""
function pointwise_loglikelihood end

# Plain Bradley–Terry / Thurstone.
function pointwise_loglikelihood(fitted::FittedComparativeModel{M, MLE},
                                 data::PairwiseData) where {M <: Union{BradleyTerry, ThurstoneCaseV}}
    agg = _aggregate_pairs(data.wins, length(data.labels))
    return _pointwise_pairs(strengths(fitted), agg, _pair_link(fitted.model))
end

function pointwise_loglikelihood(fitted::FittedComparativeModel{M, Bayesian, BTMCMCSamples},
                                 data::PairwiseData) where {M <: Union{BradleyTerry, ThurstoneCaseV}}
    agg = _aggregate_pairs(data.wins, length(data.labels))
    return _pointwise_pairs(fitted.result.samples, agg, _pair_link(fitted.model))
end

# Covariate models (latent difference is (zᵢ − zⱼ)ᵀβ = λᵢ − λⱼ).
function pointwise_loglikelihood(fitted::FittedComparativeModel{M, I, CovariateMLEResult},
                                 data::CovariateData) where {M <: Covariates, I}
    agg = _aggregate_pairs(data.data.wins, length(data.data.labels))
    return _pointwise_pairs(strengths(fitted), agg, _pair_link(fitted.model))
end

function pointwise_loglikelihood(fitted::FittedComparativeModel{M, Bayesian, CovariateMCMCSamples},
                                 data::CovariateData) where {M <: Covariates}
    agg = _aggregate_pairs(data.data.wins, length(data.data.labels))
    return _pointwise_pairs(_lambda_draws(fitted.result), agg, _pair_link(fitted.model))
end

# Anchored models (comparison terms only).
function pointwise_loglikelihood(fitted::FittedComparativeModel{M, MLE, AnchoredMLEResult},
                                 data::AnchoredData) where {M <: Anchored}
    pw = data.data
    agg = _aggregate_pairs(pw.wins, length(pw.labels))
    return _pointwise_pairs(fitted.result.λ, agg, _pair_link(fitted.model))
end

function pointwise_loglikelihood(fitted::FittedComparativeModel{M, Bayesian, AnchoredMCMCSamples},
                                 data::AnchoredData) where {M <: Anchored}
    pw = data.data
    agg = _aggregate_pairs(pw.wins, length(pw.labels))
    return _pointwise_pairs(fitted.result.λ_samples, agg, _pair_link(fitted.model))
end

# Intransitive models (latent difference carries the skew-symmetric γ term).
function _gamma_aligned(pairs::Vector{Tuple{Int, Int}}, agg::_AggregatedPairData)
    gidx = Dict(pairs[c] => c for c in eachindex(pairs))
    return [get(gidx, agg.pairs[p], 0) for p in 1:agg.P]
end

function pointwise_loglikelihood(fitted::FittedComparativeModel{M, MLE, IntransitiveMLEResult},
                                 data::PairwiseData) where {M <: Intransitive}
    r = fitted.result
    agg = _aggregate_pairs(data.wins, length(data.labels))
    cols = _gamma_aligned(r.pairs, agg)
    out = Vector{Float64}(undef, agg.P)
    @inbounds for p in 1:agg.P
        i, j = agg.pairs[p]
        N = agg.Nvec[p]; y = round(Int, agg.κ[p] + N / 2)
        γ = cols[p] == 0 ? 0.0 : r.γ[cols[p]]
        out[p] = _pair_logit(r.λ[i] - r.λ[j] + γ, y, N)
    end
    return out
end

function pointwise_loglikelihood(fitted::FittedComparativeModel{M, Bayesian, IntransitiveMCMCSamples},
                                 data::PairwiseData) where {M <: Intransitive}
    r = fitted.result
    agg = _aggregate_pairs(data.wins, length(data.labels))
    cols = _gamma_aligned(r.pairs, agg)
    S = size(r.λ_samples, 1)
    out = Matrix{Float64}(undef, S, agg.P)
    @inbounds for p in 1:agg.P
        i, j = agg.pairs[p]
        N = agg.Nvec[p]; y = round(Int, agg.κ[p] + N / 2)
        c = cols[p]
        for s in 1:S
            γ = c == 0 ? 0.0 : r.γ_samples[s, c]
            out[s, p] = _pair_logit(r.λ_samples[s, i] - r.λ_samples[s, j] + γ, y, N)
        end
    end
    return out
end

# Rater-heterogeneity mixture (observation unit = one (rater, pair) cell).
_rater_cell_logdens(λi_minus_λj::Float64, qr::Float64, w::Int, n::Int) = begin
    s = _sigmoid(λi_minus_λj)
    pwin = qr * s + (1.0 - qr) / 2.0
    w * log(pwin) + (n - w) * log(1.0 - pwin)
end

function pointwise_loglikelihood(fitted::FittedComparativeModel{M, MLE, R},
                                 data::RaterData) where {M <: RaterHeterogeneity, R <: RaterMLEResult}
    r = fitted.result
    cells = _rater_aggregate(data)
    C = length(cells.n)
    out = Vector{Float64}(undef, C)
    @inbounds for c in 1:C
        out[c] = _rater_cell_logdens(r.λ[cells.i[c]] - r.λ[cells.j[c]],
                                     r.q[cells.rater[c]], cells.w[c], cells.n[c])
    end
    return out
end

function pointwise_loglikelihood(fitted::FittedComparativeModel{M, Bayesian, R},
                                 data::RaterData) where {M <: RaterHeterogeneity, R <: RaterMCMCSamples}
    r = fitted.result
    cells = _rater_aggregate(data)
    C = length(cells.n)
    S = size(r.λ_samples, 1)
    out = Matrix{Float64}(undef, S, C)
    @inbounds for c in 1:C
        i = cells.i[c]; j = cells.j[c]; rr = cells.rater[c]; w = cells.w[c]; n = cells.n[c]
        for s in 1:S
            out[s, c] = _rater_cell_logdens(r.λ_samples[s, i] - r.λ_samples[s, j],
                                            r.q_samples[s, rr], w, n)
        end
    end
    return out
end

# ─── Parameter and observation counts ────────────────────────────────────────

"""
    nparams(fitted)

Number of free parameters of an [`MLE`](@ref)/[`StepwiseMLE`](@ref) `fitted`
model: `K-1` for plain Bradley–Terry/Thurstone, the number of selected
covariates for a covariate model, `(K-1)+3` for an anchored model (intercept,
slope, noise variance), `K+M` for a rater-heterogeneity model (`M` raters), and
`(K-1)+P` for an intransitive model (`P` observed pairs).
"""
function nparams end

nparams(f::FittedComparativeModel{BradleyTerry, MLE}) = length(f.labels) - 1
nparams(f::FittedComparativeModel{ThurstoneCaseV, MLE}) = length(f.labels) - 1
nparams(f::FittedComparativeModel{M, I, CovariateMLEResult}) where {M <: Covariates, I} =
    length(f.result.selected)
nparams(f::FittedComparativeModel{M, MLE, AnchoredMLEResult}) where {M <: Anchored} =
    (length(f.labels) - 1) + 3
nparams(f::FittedComparativeModel{M, MLE, R}) where {M <: RaterHeterogeneity, R <: RaterMLEResult} =
    length(f.labels) + length(f.result.q)
nparams(f::FittedComparativeModel{M, MLE, IntransitiveMLEResult}) where {M <: Intransitive} =
    (length(f.labels) - 1) + length(f.result.pairs)

"""
    nobs(data)

Number of pairwise comparisons in `data` (the total trial count), used as the
sample size in the BIC penalty.
"""
function nobs end

nobs(d::PairwiseData) = sum(d.wins)
nobs(d::CovariateData) = sum(d.data.wins)
nobs(d::AnchoredData) = nobs(d.data)
nobs(d::RaterData) = length(d.winner)
