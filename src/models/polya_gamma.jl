# Exact PG(1, ψ) sampler via Devroye / Polson-Scott-Windle (2013).
# J*(1, z) representation: PG(1, ψ) = J*(1, |ψ|/2) / 4.

const _PG_COEF = π^2 / 8

# Inverse Gaussian IG(μ, λ=1) sample via Michael-Schucany-Haas algorithm.
function _sample_ig(rng::AbstractRNG, μ::Float64)::Float64
    ν = randn(rng)
    y = ν^2
    x = μ + μ^2 * y / 2 - μ / 2 * sqrt(4μ * y + μ^2 * y^2)
    u = rand(rng)
    return u <= μ / (μ + x) ? x : μ^2 / x
end

# Sample from J*(1, z) using the alternating-series method.
function _sample_jstar(rng::AbstractRNG, z::Float64)::Float64
    # Proposal: truncated inverse Gaussian on [0, t] or exponential on (t, ∞)
    t = 0.64
    K = _PG_COEF + z^2 / 2
    p = (π / (2K)) * exp(-K * t)
    q = 2 * exp(-z) * _pigauss(t, z)   # cdf of IG(1/z, 1) at t

    while true
        u = rand(rng)
        if u < p / (p + q)
            # Sample from Exp(K) conditioned on > t: x = t + Exp(K)
            x = t + randexp(rng) / K
        else
            # Sample from IG(1/z, 1) truncated to [0, t]
            x = _sample_tig(rng, z, t)
        end

        # Evaluate alternating series bound S_n(x)
        s = _apgseries(x, z)
        v = rand(rng) * s
        n = 0
        go = true
        while go
            n += 1
            term = _apgterm(n, x)
            if isodd(n)
                s += term
                v <= s && (go = false; break)
            else
                s -= term
                v > s && (go = false; break)
            end
            n > 200 && (go = false; break)
        end
        isodd(n) && return x
    end
end

# Truncated IG(1/z, 1) on [0, t] via rejection from full IG.
function _sample_tig(rng::AbstractRNG, z::Float64, t::Float64)::Float64
    for _ in 1:10_000
        x = (z > 0) ? _sample_ig(rng, 1.0 / z) : abs(randn(rng))^(-2)
        x < t && return x
    end
    return t * rand(rng)
end

# CDF of IG(1/z, 1) at t: Φ(√(1/t)(zt-1)) + e^(2z) Φ(-√(1/t)(zt+1))
function _pigauss(t::Float64, z::Float64)::Float64
    if z < 1e-8
        return 2.0 * (1.0 - _normcdf(1.0 / sqrt(t)))
    end
    a = sqrt(1.0 / t) * (z * t - 1.0)
    b = sqrt(1.0 / t) * (z * t + 1.0)
    return _normcdf(a) + exp(2z) * _normcdf(-b)
end

# Normal CDF via Abramowitz & Stegun 26.2.17; |error| < 7.5e-8
function _normcdf(x::Float64)
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    p = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    φ = exp(-0.5 * x^2) * 0.3989422804014327  # 1/√(2π)
    prob = 1.0 - φ * p
    return x >= 0.0 ? prob : 1.0 - prob
end

# First term of alternating series for acceptance.
function _apgseries(x::Float64, z::Float64)::Float64
    return _apgterm(0, x)
end

function _apgterm(n::Int, x::Float64)::Float64
    # a_n(x) = π(2n+1) exp(-(2n+1)²π²x / 8) for n = 0, 1, 2, ...
    v = (2n + 1) * π
    return v * exp(-v^2 * x / 8.0)
end

# Sample PG(1, ψ).
function _sample_pg1(rng::AbstractRNG, ψ::Float64)::Float64
    z = abs(ψ) / 2.0
    return _sample_jstar(rng, z) / 4.0
end

# Moments of PG(1, ψ): mean and variance (Polson et al. 2013).
function _pg1_moments(ψ::Float64)
    z = abs(ψ)
    z < 1e-5 && return 0.25, 1.0 / 24.0
    hz  = z / 2
    th  = tanh(hz)
    μ   = th / (2z)
    # Guard against overflow in sinh/cosh for very large z
    z > 500.0 && return μ, 1.0 / (2z^3)
    ch  = cosh(hz)
    σ²  = (sinh(z) - z) / (4z^3 * ch^2)
    return μ, σ²
end

# Normal approximation threshold for PG(b, ψ).
# For b ≥ 13 the Gaussian tail mass below 0 is < 6 ppm at ψ = 0 (worst case).
const _PG_NORMAL_THRESH = 13

# Sample PG(b, ψ) for integer b ≥ 1.
# Uses the normal approximation for b ≥ _PG_NORMAL_THRESH (1 randn vs b J* draws).
function _sample_pg(rng::AbstractRNG, b::Int, ψ::Float64)::Float64
    if b >= _PG_NORMAL_THRESH
        μ1, σ²1 = _pg1_moments(ψ)
        μ = b * μ1
        σ = sqrt(b * σ²1)
        while true
            x = μ + σ * randn(rng)
            x > 0.0 && return x
        end
    end
    ω = 0.0
    for _ in 1:b
        ω += _sample_pg1(rng, ψ)
    end
    return ω
end
