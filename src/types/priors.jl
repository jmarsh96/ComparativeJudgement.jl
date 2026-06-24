"""
    AbstractPrior

Supertype of prior specifications for [`Bayesian`](@ref) inference.
"""
abstract type AbstractPrior end

"""
    NormalPrior(μ, Σ)
    NormalPrior(K; scale=10.0)

Multivariate normal prior `N(μ, Σ)`. The convenience constructor gives a
`K`-variate `N(0, scale·I)`.
"""
struct NormalPrior <: AbstractPrior
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    function NormalPrior(μ::AbstractVector, Σ::AbstractMatrix)
        K = length(μ)
        size(Σ) == (K, K) || throw(DimensionMismatch(
            "Σ must be $(K)×$(K) to match μ of length $K, got $(size(Σ))"))
        new(Vector{Float64}(μ), Matrix{Float64}(Σ))
    end
end
NormalPrior(K::Int; scale::Float64=10.0) = NormalPrior(zeros(K), scale * Matrix{Float64}(I, K, K))

"""
    InverseGammaPrior(α, β)

Inverse-gamma prior with shape `α > 0` and scale `β > 0`, used for variance
parameters.
"""
struct InverseGammaPrior <: AbstractPrior
    α::Float64
    β::Float64
    function InverseGammaPrior(α::Real, β::Real)
        α > 0 || throw(ArgumentError("α must be positive, got $α"))
        β > 0 || throw(ArgumentError("β must be positive, got $β"))
        new(Float64(α), Float64(β))
    end
end

"""
    AnchoredPrior(; τ²=0.01, β_prior=NormalPrior(2), σ²_prior=InverseGammaPrior(2.0, 1.0))

Priors for an [`Anchored`](@ref) model: a ridge precision `τ²` on the latent
strengths, a bivariate [`NormalPrior`](@ref) on the calibration coefficients
`β = (a, b)`, and an [`InverseGammaPrior`](@ref) on the anchor noise
variance `σ²`.
"""
struct AnchoredPrior <: AbstractPrior
    τ²::Float64
    β_prior::NormalPrior
    σ²_prior::InverseGammaPrior
    function AnchoredPrior(τ²::Real, β_prior::NormalPrior, σ²_prior::InverseGammaPrior)
        τ² > 0 || throw(ArgumentError("τ² must be positive, got $τ²"))
        length(β_prior.μ) == 2 || throw(DimensionMismatch(
            "β_prior must be bivariate (intercept and slope), got K=$(length(β_prior.μ))"))
        new(Float64(τ²), β_prior, σ²_prior)
    end
end
function AnchoredPrior(; τ²::Real=0.01, β_prior::NormalPrior=NormalPrior(2),
                       σ²_prior::InverseGammaPrior=InverseGammaPrior(2.0, 1.0))
    return AnchoredPrior(τ², β_prior, σ²_prior)
end

"""
    HorseshoePrior(; τ₀=1.0)

Horseshoe (global-local) shrinkage prior on the covariate coefficients β for a
[`Bayesian`](@ref) [`Covariates`](@ref) fit. Each coefficient has its own local
scale and shares a global scale `τ` (hyperprior scale `τ₀`), strongly shrinking
small coefficients while leaving large ones almost unpenalised. Implemented with
the inverse-gamma auxiliary representation of Makalic & Schmidt (2016).
"""
struct HorseshoePrior <: AbstractPrior
    τ₀::Float64
    function HorseshoePrior(; τ₀::Real=1.0)
        τ₀ > 0 || throw(ArgumentError("τ₀ must be positive, got $τ₀"))
        new(Float64(τ₀))
    end
end

"""
    SpikeSlabPrior(; v_slab=10.0, v_spike=0.01, π₀=0.5)

Continuous spike-and-slab (SSVS) prior on the covariate coefficients β for a
[`Bayesian`](@ref) [`Covariates`](@ref) fit. Each coefficient is drawn from a
wide "slab" `N(0, v_slab)` when included or a narrow "spike" `N(0, v_spike)`
when excluded, with prior inclusion probability `π₀`. Yields posterior
inclusion probabilities per covariate ([`inclusion_probabilities`](@ref)).
"""
struct SpikeSlabPrior <: AbstractPrior
    v_slab::Float64
    v_spike::Float64
    π₀::Float64
    function SpikeSlabPrior(; v_slab::Real=10.0, v_spike::Real=0.01, π₀::Real=0.5)
        v_slab > 0 || throw(ArgumentError("v_slab must be positive, got $v_slab"))
        v_spike > 0 || throw(ArgumentError("v_spike must be positive, got $v_spike"))
        v_slab > v_spike || throw(ArgumentError(
            "v_slab ($v_slab) must exceed v_spike ($v_spike)"))
        0 < π₀ < 1 || throw(ArgumentError("π₀ must be in (0, 1), got $π₀"))
        new(Float64(v_slab), Float64(v_spike), Float64(π₀))
    end
end

"""
    BetaPrior(a=1, b=1)

Beta prior `Beta(a, b)` (`a, b > 0`) on a rater reliability `q_r ∈ [0, 1]` of a
[`RaterHeterogeneity`](@ref) fit. The default `Beta(1, 1)` is uniform.
"""
struct BetaPrior <: AbstractPrior
    a::Float64
    b::Float64
    function BetaPrior(a::Real=1.0, b::Real=1.0)
        a > 0 || throw(ArgumentError("a must be positive, got $a"))
        b > 0 || throw(ArgumentError("b must be positive, got $b"))
        new(Float64(a), Float64(b))
    end
end

"""
    RaterHeterogeneityPrior(; λ_prior=nothing, q_prior=BetaPrior())

Priors for a [`Bayesian`](@ref) [`RaterHeterogeneity`](@ref) fit: a
[`NormalPrior`](@ref) on the latent strengths λ (`λ_prior`, defaulting to
`NormalPrior(K)` when left as `nothing`) and a [`BetaPrior`](@ref) shared by the
rater reliabilities `q_r` (`q_prior`).
"""
struct RaterHeterogeneityPrior <: AbstractPrior
    λ_prior::Union{Nothing, NormalPrior}
    q_prior::BetaPrior
    function RaterHeterogeneityPrior(; λ_prior::Union{Nothing, NormalPrior}=nothing,
                                     q_prior::BetaPrior=BetaPrior())
        new(λ_prior, q_prior)
    end
end

"""
    IntransitivityPrior(; λ_prior=nothing, σ²γ_prior=InverseGammaPrior(2, 1))

Priors for a [`Bayesian`](@ref) [`Intransitive`](@ref) fit: a
[`NormalPrior`](@ref) on the latent strengths λ (`λ_prior`, defaulting to
`NormalPrior(K)` when left as `nothing`) and an [`InverseGammaPrior`](@ref) on
the variance `σ²_γ` of the skew-symmetric terms `γᵢⱼ ~ N(0, σ²_γ)` (`σ²γ_prior`).
Sampling `σ²_γ` lets the fit infer the overall amount of intransitivity.
"""
struct IntransitivityPrior <: AbstractPrior
    λ_prior::Union{Nothing, NormalPrior}
    σ²γ_prior::InverseGammaPrior
    function IntransitivityPrior(; λ_prior::Union{Nothing, NormalPrior}=nothing,
                                 σ²γ_prior::InverseGammaPrior=InverseGammaPrior(2.0, 1.0))
        new(λ_prior, σ²γ_prior)
    end
end
