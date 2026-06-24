# ─── Pareto-smoothed importance sampling LOO-CV (PSIS-LOO) ───────────────────
#
# Approximates leave-one-out cross-validation from the pointwise log-likelihood
# of a Bayesian fit using Pareto-smoothed importance sampling (Vehtari, Gelman,
# Gabry 2017; Vehtari et al. 2022). For held-out observation i the importance
# ratios are r_s = 1/p(y_i | θ_s), i.e. log ratios −loglik_{i,s}; `_psis_smooth`
# (utils.jl) stabilises their upper tail with a generalized-Pareto fit and
# returns the shape diagnostic k̂. A k̂ above 0.7 flags an observation whose LOO
# estimate is unreliable.

"""
    LOOResult

Result of [`loo`](@ref): the expected log pointwise predictive density
`elpd_loo`, the effective number of parameters `p_loo`, `looic = -2·elpd_loo`
(lower is better), a standard error `se`, the per-observation `pointwise` elpd
contributions, and the per-observation Pareto-`k` diagnostics (`> 0.7` flags an
unreliable point).
"""
struct LOOResult
    elpd_loo::Float64
    p_loo::Float64
    looic::Float64
    se::Float64
    pointwise::Vector{Float64}
    pareto_k::Vector{Float64}
end

function Base.show(io::IO, r::LOOResult)
    println(io, "LOOResult")
    println(io, "  elpd_loo = ", round(r.elpd_loo, digits=2), " ± ", round(r.se, digits=2))
    println(io, "  p_loo    = ", round(r.p_loo, digits=2))
    println(io, "  looic    = ", round(r.looic, digits=2))
    nbad = count(>(0.7), r.pareto_k)
    if nbad == 0
        print(io, "  pareto k : all ok (≤ 0.7)")
    else
        print(io, "  pareto k : ", nbad, " of ", length(r.pareto_k),
              " observation(s) with k̂ > 0.7 — LOO unreliable for these")
    end
end

"""
    loo(fitted, data)

Pareto-smoothed importance-sampling leave-one-out cross-validation for a
[`Bayesian`](@ref) `fitted` model on its `data`. Returns a [`LOOResult`](@ref)
holding `elpd_loo`, `p_loo`, `looic`, and per-observation Pareto-`k` diagnostics.
Errors for [`MLE`](@ref) fits. Compare Bayesian models on the same data by
`elpd_loo` (higher) or `looic` (lower); check `pareto_k` for reliability.
"""
function loo(fitted::FittedComparativeModel, data)
    ll = pointwise_loglikelihood(fitted, data)
    ll isa AbstractMatrix || throw(ArgumentError(
        "LOO requires a Bayesian fit with posterior draws; got a point estimate. " *
        "Use `aic`/`bic` for an MLE fit."))
    S, n = size(ll)
    elpd_i = Vector{Float64}(undef, n)
    pareto_k = Vector{Float64}(undef, n)
    lpd_total = 0.0
    @inbounds for i in 1:n
        col = collect(view(ll, :, i))
        lw, k = _psis_smooth(-col)          # log importance ratios = −loglik
        pareto_k[i] = k
        elpd_i[i] = _logsumexp(lw .+ col)   # log Σ w̃_s p(y_i | θ_s), w̃ normalised
        lpd_total += _logsumexp(col) - log(S)
    end
    elpd = sum(elpd_i)
    se = sqrt(n * var(elpd_i))
    return LOOResult(elpd, lpd_total - elpd, -2.0 * elpd, se, elpd_i, pareto_k)
end
