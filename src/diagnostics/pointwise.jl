# ─── Pointwise log-likelihood reconstruction ─────────────────────────────────
#
# WAIC and PSIS-LOO need the log density of each observation at every posterior
# draw; `loglikelihood(model, :)` needs the per-observation vector at the point
# estimate / posterior mean. Both are reconstructed from the parameters and the
# data stored on the fitted model. The observation unit is one observed pair as a
# binomial (`Nₚ` trials, `yₚ` wins) — for the rater model, one (rater, pair) cell.
# For anchored fits only the comparison terms enter (anchor terms excluded), so
# WAIC/LOO stay comparable to the unanchored fit. The parameter-free `log C(N, y)`
# constant is omitted, matching `_bt_loglik` and friends.

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

# The underlying PairwiseData of each data container (rater data handled apart).
_pairwise(d::PairwiseData) = d
_pairwise(d::CovariateData) = d.data
_pairwise(d::AnchoredData) = d.data

# S×P pointwise matrix from an S×K matrix of latent-strength draws.
function _pointwise_pairs(Λ::AbstractMatrix, agg::_AggregatedPairData, link)
    S = size(Λ, 1)
    out = Matrix{Float64}(undef, S, agg.P)
    @inbounds for p in 1:agg.P
        i, j = agg.pairs[p]
        N = agg.Nvec[p]; y = round(Int, agg.κ[p] + N / 2)
        for s in 1:S
            out[s, p] = link(Λ[s, i] - Λ[s, j], y, N)
        end
    end
    return out
end

# length-P pointwise vector from a single latent-strength vector.
function _pointwise_pairs(λ::AbstractVector, agg::_AggregatedPairData, link)
    out = Vector{Float64}(undef, agg.P)
    @inbounds for p in 1:agg.P
        i, j = agg.pairs[p]
        N = agg.Nvec[p]; y = round(Int, agg.κ[p] + N / 2)
        out[p] = link(λ[i] - λ[j], y, N)
    end
    return out
end

# Map each aggregated pair to its column in the intransitivity γ-array (0 = none).
function _gamma_aligned(pairs::Vector{Tuple{Int, Int}}, agg::_AggregatedPairData)
    gidx = Dict(pairs[c] => c for c in eachindex(pairs))
    return [get(gidx, agg.pairs[p], 0) for p in 1:agg.P]
end

# Mixture log density of one (rater, pair) cell.
_rater_cell_logdens(d::Float64, qr::Float64, w::Int, n::Int) = begin
    s = _sigmoid(d)
    pwin = qr * s + (1.0 - qr) / 2.0
    w * log(pwin) + (n - w) * log(1.0 - pwin)
end

# Rater reliabilities as a per-rater-index vector (point est / posterior mean).
_rater_q(f::FittedComparativeModel{<:RaterHeterogeneity, MLE}) = f.result.q
_rater_q(f::FittedComparativeModel{<:RaterHeterogeneity, Bayesian}) =
    vec(mean(f.result.q_samples, dims=1))

# ─── Pointwise at the representative parameters (point est / posterior mean) ──
# Returns a length-`n_obs` vector; `sum` is the scalar `loglikelihood(model)`.
# `strengths`, `intransitivity`, `_rater_q` already return point estimates for an
# MLE fit and posterior means for a Bayesian fit, so one method covers both.

function _pointwise_at_point(f::FittedComparativeModel{<:Union{BradleyTerry, ThurstoneCaseV, Anchored, Covariates}})
    pw = _pairwise(f.data)
    agg = _aggregate_pairs(pw.wins, length(pw.labels))
    return _pointwise_pairs(strengths(f), agg, _pair_link(f.model))
end

function _pointwise_at_point(f::FittedComparativeModel{<:Intransitive})
    pw = _pairwise(f.data)
    agg = _aggregate_pairs(pw.wins, length(pw.labels))
    λ = strengths(f); Γ = intransitivity(f)
    out = Vector{Float64}(undef, agg.P)
    @inbounds for p in 1:agg.P
        i, j = agg.pairs[p]
        N = agg.Nvec[p]; y = round(Int, agg.κ[p] + N / 2)
        out[p] = _pair_logit(λ[i] - λ[j] + Γ[i, j], y, N)
    end
    return out
end

function _pointwise_at_point(f::FittedComparativeModel{<:RaterHeterogeneity})
    cells = _rater_aggregate(f.data)
    λ = strengths(f); q = _rater_q(f)
    out = Vector{Float64}(undef, length(cells.n))
    @inbounds for c in eachindex(cells.n)
        out[c] = _rater_cell_logdens(λ[cells.i[c]] - λ[cells.j[c]],
                                     q[cells.rater[c]], cells.w[c], cells.n[c])
    end
    return out
end

# ─── Per-draw pointwise matrix (Bayesian only), n_draws × n_obs ──────────────
# Feeds WAIC and PSIS-LOO.

function _loglik_draws(f::FittedComparativeModel{<:Union{BradleyTerry, ThurstoneCaseV}, Bayesian, BTMCMCSamples})
    agg = _aggregate_pairs(f.data.wins, length(f.data.labels))
    return _pointwise_pairs(f.result.samples, agg, _pair_link(f.model))
end

function _loglik_draws(f::FittedComparativeModel{<:Covariates, Bayesian, CovariateMCMCSamples})
    pw = f.data.data
    agg = _aggregate_pairs(pw.wins, length(pw.labels))
    return _pointwise_pairs(_lambda_draws(f.result), agg, _pair_link(f.model))
end

function _loglik_draws(f::FittedComparativeModel{<:Anchored, Bayesian, AnchoredMCMCSamples})
    pw = f.data.data
    agg = _aggregate_pairs(pw.wins, length(pw.labels))
    return _pointwise_pairs(f.result.λ_samples, agg, _pair_link(f.model))
end

function _loglik_draws(f::FittedComparativeModel{<:Intransitive, Bayesian, IntransitiveMCMCSamples})
    r = f.result
    agg = _aggregate_pairs(f.data.wins, length(f.data.labels))
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

function _loglik_draws(f::FittedComparativeModel{<:RaterHeterogeneity, Bayesian, R}) where {R <: RaterMCMCSamples}
    r = f.result
    cells = _rater_aggregate(f.data)
    S = size(r.λ_samples, 1)
    out = Matrix{Float64}(undef, S, length(cells.n))
    @inbounds for c in eachindex(cells.n)
        i = cells.i[c]; j = cells.j[c]; rr = cells.rater[c]; w = cells.w[c]; n = cells.n[c]
        for s in 1:S
            out[s, c] = _rater_cell_logdens(r.λ_samples[s, i] - r.λ_samples[s, j],
                                            r.q_samples[s, rr], w, n)
        end
    end
    return out
end
