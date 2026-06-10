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
