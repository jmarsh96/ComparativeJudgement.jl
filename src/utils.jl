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
