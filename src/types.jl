"""
    AbstractComparativeModel

Supertype of all comparative judgement models.
"""
abstract type AbstractComparativeModel end

"""
    PairwiseModel <: AbstractComparativeModel

Supertype of models for pairwise comparison data (one winner per comparison).
"""
abstract type PairwiseModel <: AbstractComparativeModel end

"""
    RankingModel <: AbstractComparativeModel

Supertype of models for full or partial ranking data.
"""
abstract type RankingModel <: AbstractComparativeModel end

"""
    BradleyTerry()

The Bradley–Terry model for pairwise comparisons: each item has a latent
strength λ and `P(i beats j) = logistic(λᵢ − λⱼ)`. Fit with [`MLE`](@ref) or
[`Bayesian`](@ref) inference via [`fit`](@ref).
"""
struct BradleyTerry <: PairwiseModel end

"""
    PlackettLuce()

The Plackett–Luce ranking model. Placeholder — no inference implemented yet.
"""
struct PlackettLuce <: RankingModel end

"""
    ThurstoneCaseV(distribution)

Thurstone's Case V pairwise comparison model. Placeholder — no inference
implemented yet.
"""
struct ThurstoneCaseV <: PairwiseModel
    distribution::Symbol
end

"""
    Anchored(model)

Wrapper composing any comparative model with anchor measurements that
calibrate the latent scale via `y = a + b·λ + ε`. Fit with an
[`AnchoredData`](@ref) wrapping the comparison data, e.g.

```julia
fit(BradleyTerryAnchored(), AnchoredData(data, Dict("A" => 3.1, "C" => 4.5)))
```
"""
struct Anchored{M <: AbstractComparativeModel} <: AbstractComparativeModel
    model::M
end

"""
    BradleyTerryAnchored()

Alias for `Anchored(BradleyTerry())`: the joint Bradley–Terry + linear
calibration model. See [`Anchored`](@ref) and [`fit`](@ref).
"""
const BradleyTerryAnchored = Anchored{BradleyTerry}
BradleyTerryAnchored() = Anchored(BradleyTerry())

"""
    InferenceMethod

Supertype of inference methods accepted by [`fit`](@ref).
"""
abstract type InferenceMethod end

"""
    MLE()

Maximum-likelihood estimation.
"""
struct MLE <: InferenceMethod end

"""
    Bayesian(; n_samples=2000, n_burnin=500, center=true, thin=1)

MCMC (Gibbs sampling) inference. Runs `n_burnin` warm-up sweeps, then keeps
every `thin`-th of the following `thin × n_samples` sweeps, so the result
always holds `n_samples` posterior draws.

`center` re-centres the latent strengths to sum to zero after every sweep.
It defaults to `true` for anchored models too: the anchor likelihood only
constrains `a + b·λ`, so λ's location is shared with the intercept and pinned
down just by weak priors — re-centering removes that flat direction.
"""
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
    BTMCMCSamples

Posterior draws from a [`Bayesian`](@ref) Bradley–Terry fit: the
`n_samples × K` matrix of latent strength draws and the per-draw
log-likelihoods.
"""
struct BTMCMCSamples
    samples::Matrix{Float64}          # n_samples × K
    loglikelihoods::Vector{Float64}   # n_samples, one per post-burnin draw
    n_samples::Int
    n_burnin::Int
end

"""
    PairwiseData(wins, labels)

Pairwise comparison data: `wins[i, j]` counts how many times item `i` beat
item `j`, and `labels` names the items (any element type).
"""
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

"""
    AnchoredData(data, anchor_labels, anchor_values)
    AnchoredData(data, anchors::AbstractDict)

Comparison data augmented with anchor measurements `y` for a subset of
items, identified by label. Used to fit [`Anchored`](@ref) models, which
calibrate the latent scale via `y = a + b·λ + ε`.
"""
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

"""
    AnchoredMCMCSamples

Posterior draws from an [`Anchored`](@ref) model fit: latent strengths λ
(`n_samples × n`), calibration coefficients `β = (a, b)` (`n_samples × 2`),
anchor noise variances `σ²`, and per-draw joint log-likelihoods.
"""
struct AnchoredMCMCSamples
    λ_samples::Matrix{Float64}        # n_samples × n
    β_samples::Matrix{Float64}        # n_samples × 2 (columns a, b)
    σ²_samples::Vector{Float64}       # n_samples
    loglikelihoods::Vector{Float64}   # joint log p(c, y | λ, β, σ²) per draw
    n_samples::Int
    n_burnin::Int
    thin::Int
end

"""
    FittedComparativeModel

The result of [`fit`](@ref): bundles the model, the inference method, the
raw fitting result (an `Optim` result for [`MLE`](@ref), a posterior-draws
container for [`Bayesian`](@ref)), the item labels, and convergence info.
Query it with the accessor functions ([`strengths`](@ref),
[`probability`](@ref), [`posterior_mean`](@ref), [`predict`](@ref), …)
rather than via `result` directly.
"""
struct FittedComparativeModel{M <: AbstractComparativeModel, I <: InferenceMethod, R, L}
    model::M
    method::I
    result::R
    labels::Vector{L}
    converged::Bool
    iterations::Int
end
