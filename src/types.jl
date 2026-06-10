abstract type AbstractComparativeModel end

abstract type PairwiseModel <: AbstractComparativeModel end
abstract type RankingModel <: AbstractComparativeModel end

struct BradleyTerry <: PairwiseModel end
struct PlackettLuce <: RankingModel end
struct ThurstoneCaseV <: PairwiseModel
    distribution::Symbol
end

abstract type InferenceMethod end
struct MLE <: InferenceMethod end
struct Bayesian <: InferenceMethod
    n_samples::Int
    n_burnin::Int
    center::Bool
    function Bayesian(; n_samples::Int=2000, n_burnin::Int=500, center::Bool=true)
        n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
        n_burnin >= 0 || throw(ArgumentError("n_burnin must be non-negative"))
        new(n_samples, n_burnin, center)
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

struct FittedComparativeModel{M <: AbstractComparativeModel, I <: InferenceMethod, R, L}
    model::M
    method::I
    result::R
    labels::Vector{L}
    converged::Bool
    iterations::Int
end