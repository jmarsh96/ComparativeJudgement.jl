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
