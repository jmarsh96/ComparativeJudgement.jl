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
struct Bayesian <: InferenceMethod end

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