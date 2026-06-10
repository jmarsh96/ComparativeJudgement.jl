abstract type AbstractComparativeModel end

abstract type PairwiseModel <: AbstractComparativeModel end
abstract type RankingModel <: AbstractComparativeModel end

struct BradleyTerry <: PairwiseModel end
struct PlackettLuce <: RankingModel end
struct ThurstoneCaseV <: PairwiseModel
    distribution::Symbol
end

# Wrapper composing any comparative model with anchor measurements that
# calibrate the latent scale via y = a + b·λ + ε.
struct Anchored{M <: AbstractComparativeModel} <: AbstractComparativeModel
    model::M
end

const BradleyTerryAnchored = Anchored{BradleyTerry}
BradleyTerryAnchored() = Anchored(BradleyTerry())

abstract type InferenceMethod end
struct MLE <: InferenceMethod end
struct Bayesian <: InferenceMethod
    n_samples::Int
    n_burnin::Int
    center::Bool
    thin::Int
    function Bayesian(; n_samples::Int=2000, n_burnin::Int=500, center::Bool=true, thin::Int=1)
        n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
        n_burnin >= 0 || throw(ArgumentError("n_burnin must be non-negative"))
        thin >= 1 || throw(ArgumentError("thin must be at least 1"))
        new(n_samples, n_burnin, center, thin)
    end
end

abstract type AbstractPrior end

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

struct InverseGammaPrior <: AbstractPrior
    α::Float64
    β::Float64
    function InverseGammaPrior(α::Real, β::Real)
        α > 0 || throw(ArgumentError("α must be positive, got $α"))
        β > 0 || throw(ArgumentError("β must be positive, got $β"))
        new(Float64(α), Float64(β))
    end
end

# Priors for an anchored model: a ridge precision τ² on the latent strengths,
# a bivariate normal prior on the calibration coefficients β = (a, b), and an
# inverse-gamma prior on the anchor noise variance σ².
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

struct BTMCMCSamples
    samples::Matrix{Float64}          # n_samples × K
    loglikelihoods::Vector{Float64}   # n_samples, one per post-burnin draw
    n_samples::Int
    n_burnin::Int
end

struct PairwiseData{L}
    wins::Matrix{Int}
    labels::Vector{L}
    function PairwiseData(wins::Matrix{Int}, labels::Vector{L}) where {L}
        n = length(labels)
        size(wins) == (n, n) || throw(
            DimensionMismatch("wins must be $(n)×$(n) to match $(n) labels, got $(size(wins))")
        )
        new{L}(wins, labels)
    end
end

# Comparison data of any kind augmented with anchor measurements y for the
# subset S of items given by anchor_idx (indices into the item labels).
struct AnchoredData{D, L}
    data::D
    anchor_idx::Vector{Int}
    anchor_values::Vector{Float64}
    function AnchoredData(data::PairwiseData{L}, anchor_labels::Vector{L},
                          anchor_values::Vector{<:Real}) where {L}
        r = length(anchor_labels)
        r >= 1 || throw(ArgumentError("Need at least 1 anchor, got none"))
        length(anchor_values) == r || throw(DimensionMismatch(
            "Got $r anchor labels but $(length(anchor_values)) anchor values"))
        allunique(anchor_labels) || throw(ArgumentError("Anchor labels must be unique"))
        anchor_idx = Vector{Int}(undef, r)
        for (k, lbl) in enumerate(anchor_labels)
            idx = findfirst(==(lbl), data.labels)
            idx === nothing && throw(ArgumentError("Anchor label $(lbl) not found in data labels"))
            anchor_idx[k] = idx
        end
        new{PairwiseData{L}, L}(data, anchor_idx, Vector{Float64}(anchor_values))
    end
end
function AnchoredData(data::PairwiseData{L}, anchors::AbstractDict{L, <:Real}) where {L}
    anchor_labels = collect(keys(anchors))
    anchor_values = [Float64(anchors[lbl]) for lbl in anchor_labels]
    return AnchoredData(data, anchor_labels, anchor_values)
end

# Posterior draws for an anchored model: latent strengths λ, calibration
# coefficients β = (a, b), and anchor noise variance σ².
struct AnchoredMCMCSamples
    λ_samples::Matrix{Float64}        # n_samples × n
    β_samples::Matrix{Float64}        # n_samples × 2 (columns a, b)
    σ²_samples::Vector{Float64}       # n_samples
    loglikelihoods::Vector{Float64}   # joint log p(c, y | λ, β, σ²) per draw
    n_samples::Int
    n_burnin::Int
    thin::Int
end

struct FittedComparativeModel{M <: AbstractComparativeModel, I <: InferenceMethod, R, L}
    model::M
    method::I
    result::R
    labels::Vector{L}
    converged::Bool
    iterations::Int
end