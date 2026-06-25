# Sample Gamma(α, 1) via Marsaglia-Tsang (2000) squeeze method.
function _sample_gamma(rng::AbstractRNG, α::Float64)::Float64
    if α < 1.0
        # Boost: Gamma(α) = Gamma(α + 1) · U^(1/α)
        return _sample_gamma(rng, α + 1.0) * rand(rng)^(1.0 / α)
    end
    d = α - 1.0 / 3.0
    c = 1.0 / sqrt(9.0 * d)
    while true
        x = randn(rng)
        v = 1.0 + c * x
        v <= 0.0 && continue
        v = v^3
        u = rand(rng)
        x² = x^2
        u < 1.0 - 0.0331 * x²^2 && return d * v
        log(u) < 0.5 * x² + d * (1.0 - v + log(v)) && return d * v
    end
end

# Sample InverseGamma(α, β) with shape α and scale β.
function _sample_inv_gamma(rng::AbstractRNG, α::Float64, β::Float64)::Float64
    return β / _sample_gamma(rng, α)
end

# Sample Beta(a, b) as G_a / (G_a + G_b) with G_a ~ Gamma(a), G_b ~ Gamma(b).
function _sample_beta(rng::AbstractRNG, a::Float64, b::Float64)::Float64
    ga = _sample_gamma(rng, a)
    gb = _sample_gamma(rng, b)
    return ga / (ga + gb)
end

# Inverse standard-normal CDF (quantile) via Acklam's rational approximation,
# accurate to ~1.15e-9 over the whole range. Used for Wald confidence intervals;
# avoids pulling in a SpecialFunctions/Distributions dependency for one z-value.
function _norm_quantile(p::Float64)::Float64
    (0.0 < p < 1.0) || throw(DomainError(p, "quantile argument must be in (0, 1)"))
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = ( 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00)
    plow = 0.02425
    if p < plow
        q = sqrt(-2.0 * log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1.0)
    elseif p <= 1.0 - plow
        q = p - 0.5
        r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6]) * q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1.0)
    else
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                 ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1.0)
    end
end

# ─── Probit helpers (shared by the Thurstone Case V model) ───────────────────
#
# `_normcdf` (Abramowitz & Stegun, in polya_gamma.jl) is only good to ~7.5e-8
# absolute error, so it loses all precision in the deep lower tail where Φ(d)
# itself is < 1e-7. The two helpers below switch to the asymptotic Mills-ratio
# expansion Φ(d)/φ(d) = (1/|d|)(1 − 1/d² + 3/d⁴ − 15/d⁶ + …) for d < −4, which is
# both accurate and cheap there.

const _LOG_SQRT_2π = 0.9189385332046727   # 0.5·log(2π)
const _INV_SQRT_2π = 0.3989422804014327   # 1/√(2π)

# Lower-tail Mills ratio R(d) = Φ(d)/φ(d) for d < 0 via the asymptotic series.
_mills_lower(d::Float64) = let d2 = d * d
    (1.0 - 1.0 / d2 + 3.0 / d2^2 - 15.0 / d2^3) / (-d)
end

# Numerically stable log Φ(d).
function _log_normcdf(d::Float64)::Float64
    d < -4.0 && return -0.5 * d^2 - _LOG_SQRT_2π + log(_mills_lower(d))
    return log(_normcdf(d))
end

# Inverse Mills ratio φ(d)/Φ(d): the per-observation score weight of the probit
# log-likelihood. → −d as d → −∞.
function _inv_mills(d::Float64)::Float64
    d < -4.0 && return 1.0 / _mills_lower(d)
    return (_INV_SQRT_2π * exp(-0.5 * d^2)) / _normcdf(d)
end

# Draw Z ~ N(0, 1) truncated to (a, ∞). Plain rejection when the bound is at or
# below the mode; Robert's (1995) exponential-proposal rejection in the tail
# (a > 0), which stays exact and efficient however extreme a is.
function _randn_truncated_lower(rng::AbstractRNG, a::Float64)::Float64
    if a <= 0.0
        while true
            z = randn(rng)
            z > a && return z
        end
    end
    λ = (a + sqrt(a^2 + 4.0)) / 2.0
    while true
        z = a + randexp(rng) / λ          # a + Exp(rate λ)
        rand(rng) <= exp(-(z - λ)^2 / 2.0) && return z
    end
end

# Draw X ~ N(μ, 1) truncated to (0, ∞) if `positive`, else to (−∞, 0). Used for
# the Albert–Chib latent variables of the probit (Thurstone) Gibbs samplers.
function _sample_truncated_normal(rng::AbstractRNG, μ::Float64, positive::Bool)::Float64
    positive && return μ + _randn_truncated_lower(rng, -μ)
    return -(-μ + _randn_truncated_lower(rng, μ))
end

# ─── log-sum-exp ─────────────────────────────────────────────────────────────

# Numerically stable log Σ exp(xᵢ).
function _logsumexp(x::AbstractVector{<:Real})::Float64
    m = maximum(x)
    isfinite(m) || return Float64(m)
    s = 0.0
    @inbounds for v in x
        s += exp(v - m)
    end
    return m + log(s)
end

# ─── log-gamma and the chi-squared survival function ─────────────────────────
#
# Hand-rolled to keep the dependency tree at stdlib + Optim (the package already
# hand-rolls `_norm_quantile` for the same reason). `_lgamma` is the Lanczos
# approximation; `_chisq_sf` is the chi-squared upper tail, used for the
# likelihood-ratio-test p-value.

const _LANCZOS_G = 7.0
const _LANCZOS_C = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                    771.32342877765313, -176.61502916214059, 12.507343278686905,
                    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)

# log Γ(x) via the Lanczos approximation (g = 7), with the reflection formula
# for x < 0.5. Accurate to ~1e-13 over the range used here.
function _lgamma(x::Float64)::Float64
    x < 0.5 && return log(π / abs(sin(π * x))) - _lgamma(1.0 - x)
    x -= 1.0
    a = _LANCZOS_C[1]
    t = x + _LANCZOS_G + 0.5
    @inbounds for i in 2:9
        a += _LANCZOS_C[i] / (x + (i - 1))
    end
    return _LOG_SQRT_2π + (x + 0.5) * log(t) - t + log(a)   # 0.5·log(2π) + …
end

# Regularised lower incomplete gamma P(a, x) via its series expansion (x < a+1).
function _gser(a::Float64, x::Float64)::Float64
    x <= 0.0 && return 0.0
    gln = _lgamma(a)
    ap = a
    del = 1.0 / a
    sum = del
    for _ in 1:300
        ap += 1.0
        del *= x / ap
        sum += del
        abs(del) < abs(sum) * 1e-15 && break
    end
    return sum * exp(-x + a * log(x) - gln)
end

# Regularised upper incomplete gamma Q(a, x) via its continued fraction (x ≥ a+1).
function _gcf(a::Float64, x::Float64)::Float64
    gln = _lgamma(a)
    FPMIN = 1e-300
    b = x + 1.0 - a
    c = 1.0 / FPMIN
    d = 1.0 / b
    h = d
    for i in 1:300
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        abs(d) < FPMIN && (d = FPMIN)
        c = b + an / c
        abs(c) < FPMIN && (c = FPMIN)
        d = 1.0 / d
        del = d * c
        h *= del
        abs(del - 1.0) < 1e-15 && break
    end
    return exp(-x + a * log(x) - gln) * h
end

# Regularised upper incomplete gamma Q(a, x) = 1 − P(a, x).
function _gammq(a::Float64, x::Float64)::Float64
    (x < 0.0 || a <= 0.0) && throw(DomainError((a, x), "require a > 0 and x ≥ 0"))
    x == 0.0 && return 1.0
    return x < a + 1.0 ? 1.0 - _gser(a, x) : _gcf(a, x)
end

# Chi-squared survival function P(X > x) for X ~ χ²(df), df ≥ 1.
function _chisq_sf(x::Real, df::Real)::Float64
    x <= 0.0 && return 1.0
    return _gammq(df / 2.0, x / 2.0)
end

# ─── Rank correlations ───────────────────────────────────────────────────────

# Tie-corrected (fractional) ranks of `x`, used for Spearman's ρ.
function _tiedrank(x::AbstractVector)::Vector{Float64}
    n = length(x)
    p = sortperm(x)
    r = Vector{Float64}(undef, n)
    i = 1
    while i <= n
        j = i
        while j < n && x[p[j + 1]] == x[p[i]]
            j += 1
        end
        avg = (i + j) / 2.0
        for k in i:j
            r[p[k]] = avg
        end
        i = j + 1
    end
    return r
end

# Spearman rank correlation (Pearson correlation of the tie-corrected ranks).
function _corspearman(a::AbstractVector, b::AbstractVector)::Float64
    length(a) == length(b) || throw(DimensionMismatch("vectors must have equal length"))
    return cor(_tiedrank(a), _tiedrank(b))
end

# Σ tᵢ(tᵢ − 1)/2 over the tie groups of `x` (the Kendall τ-b tie correction).
function _tiesum(x::AbstractVector)::Int
    counts = Dict{Any, Int}()
    for v in x
        counts[v] = get(counts, v, 0) + 1
    end
    s = 0
    for (_, t) in counts
        s += t * (t - 1) ÷ 2
    end
    return s
end

# Kendall τ-b rank correlation (O(n²); fine for CJ-sized item sets).
function _corkendall(a::AbstractVector, b::AbstractVector)::Float64
    n = length(a)
    n == length(b) || throw(DimensionMismatch("vectors must have equal length"))
    nc = 0; nd = 0
    @inbounds for i in 1:(n - 1), j in (i + 1):n
        s = sign(a[i] - a[j]) * sign(b[i] - b[j])
        s > 0 && (nc += 1)
        s < 0 && (nd += 1)
    end
    n0 = n * (n - 1) ÷ 2
    n1 = _tiesum(a)
    n2 = _tiesum(b)
    denom = sqrt(float((n0 - n1) * (n0 - n2)))
    denom == 0.0 && return 0.0
    return (nc - nd) / denom
end

# ─── Pareto-smoothed importance sampling (PSIS) ──────────────────────────────
#
# Hand-rolled implementation of the generalized-Pareto tail fit (Zhang &
# Stephens, 2009) and the PSIS smoothing of importance weights (Vehtari et al.),
# used by leave-one-out cross-validation. Mirrors the algorithm in the `loo`
# R package, kept dependency-free.

# Fit a generalized-Pareto distribution to the (ascending, positive) exceedances
# `x` by the Zhang–Stephens (2009) profile method; returns the shape `k` and
# scale `σ`. A weakly informative prior pulls `k` towards 0.5.
function _gpd_fit(x::AbstractVector{Float64})
    n = length(x)
    prior_bs = 3.0
    prior_k = 10.0
    m = 30 + floor(Int, sqrt(n))
    q14 = x[max(1, floor(Int, n / 4 + 0.5))]      # ~ lower-quartile exceedance
    bs = Vector{Float64}(undef, m)
    @inbounds for i in 1:m
        bs[i] = (1.0 - sqrt(m / (i - 0.5))) / (prior_bs * q14) + 1.0 / x[n]
    end
    L = Vector{Float64}(undef, m)
    @inbounds for i in 1:m
        b = bs[i]
        kb = 0.0
        for v in x
            kb += log1p(-b * v)
        end
        kb /= n
        L[i] = n * (log(-b / kb) - kb - 1.0)
    end
    # Posterior weights over the grid (softmax of the profile log-likelihood).
    b = 0.0; wsum = 0.0
    @inbounds for j in 1:m
        s = 0.0
        for i in 1:m
            s += exp(L[i] - L[j])
        end
        w = 1.0 / s
        b += bs[j] * w
        wsum += w
    end
    b /= wsum
    k = 0.0
    @inbounds for v in x
        k += log1p(-b * v)
    end
    k /= n
    σ = -k / b
    k = (k * n + prior_k * 0.5) / (n + prior_k)
    return k, σ
end

# Smooth the log importance ratios `lr` (one observation, S draws) by replacing
# the upper tail with order-statistic quantiles of a fitted GPD, truncating, and
# normalising. Returns `(log_weights, k̂)`; `k̂ > 0.7` flags an unreliable point.
function _psis_smooth(lr::AbstractVector{Float64})
    S = length(lr)
    lw = lr .- maximum(lr)                 # shift for numerical stability
    tail_len = min(ceil(Int, 0.2 * S), ceil(Int, 3 * sqrt(S)))
    khat = Inf
    if tail_len >= 5 && S - tail_len >= 1
        ord = sortperm(lw)
        tail_idx = @view ord[(S - tail_len + 1):S]
        lw_tail = lw[tail_idx]             # ascending
        if abs(lw_tail[end] - lw_tail[1]) >= eps()
            cutoff = lw[ord[S - tail_len]]
            exp_cutoff = exp(cutoff)
            exceed = exp.(lw_tail) .- exp_cutoff
            k, σ = _gpd_fit(exceed)
            khat = k
            if isfinite(k)
                @inbounds for l in 1:tail_len
                    p = (l - 0.5) / tail_len
                    q = σ * expm1(-k * log1p(-p)) / k + exp_cutoff
                    lw[tail_idx[l]] = log(q)
                end
            end
        end
    end
    @inbounds for s in 1:S
        lw[s] > 0.0 && (lw[s] = 0.0)        # truncate at the max raw weight
    end
    lw .-= _logsumexp(lw)
    return lw, khat
end
