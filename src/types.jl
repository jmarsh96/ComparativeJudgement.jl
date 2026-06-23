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
    ThurstoneCaseV(distribution=:normal)

Thurstone's Case V pairwise comparison model: each item has a latent strength λ
and `P(i beats j) = Φ(λᵢ − λⱼ)`, the equal-variance, uncorrelated
discriminal-process model with a probit link. Fit with [`MLE`](@ref) or
[`Bayesian`](@ref) inference via [`fit`](@ref).

`distribution` selects the discriminal-process distribution; only `:normal`
(the Case V default) is currently implemented.
"""
struct ThurstoneCaseV <: PairwiseModel
    distribution::Symbol
end

ThurstoneCaseV() = ThurstoneCaseV(:normal)

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
    ThurstoneCaseVAnchored()

Alias for `Anchored(ThurstoneCaseV())`: the joint Thurstone Case V + linear
calibration model. See [`Anchored`](@ref) and [`fit`](@ref).
"""
const ThurstoneCaseVAnchored = Anchored{ThurstoneCaseV}
ThurstoneCaseVAnchored() = Anchored(ThurstoneCaseV())

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
    AnchoredData(data, anchor_groups, anchor_values)
    AnchoredData(data, anchors::AbstractDict)
    AnchoredData(data, group => value, ...)

Comparison data augmented with anchor measurements `y`, used to fit
[`Anchored`](@ref) models, which calibrate the latent scale via `y = a + b·λ + ε`.

Each measurement targets either a **single item** (passed by label, as
`anchor_labels::Vector`) or a **group of items** (passed as `anchor_groups`, a
vector of label-vectors). A group anchor is modelled as the group *mean*,

```
y_g = a + b · mean_{i∈G_g}(λ_i) + ε_g,   ε_g ~ N(0, σ²/n_g),   n_g = |G_g|,
```

so a larger group is measured more precisely (variance `σ²/n_g`). Single-item
anchors are the special case `n_g = 1`, identical to the original model. Single
and group anchors may be mixed in one dataset, and an item may appear in more than
one group. Internally every anchor is stored as a group of item indices in
`anchor_groups::Vector{Vector{Int}}`.
"""
struct AnchoredData{D, L}
    data::D
    anchor_groups::Vector{Vector{Int}}
    anchor_values::Vector{Float64}
    function AnchoredData{D, L}(data::D, anchor_groups::Vector{Vector{Int}},
                               anchor_values::Vector{Float64}) where {D, L}
        new{D, L}(data, anchor_groups, anchor_values)
    end
end

# The labels against which anchor labels are resolved. `PairwiseData` carries
# them directly; the covariate wrapper keeps them on its inner `data` (the
# `CovariateData` method is defined after that type, below).
_anchor_target_labels(data::PairwiseData) = data.labels

# Resolve a label to its item index, or throw.
function _resolve_anchor_label(labels, lbl)
    idx = findfirst(==(lbl), labels)
    idx === nothing && throw(ArgumentError("Anchor label $(lbl) not found in data labels"))
    return idx
end

# Single-item anchors: each measurement targets one item (groups of size one).
function AnchoredData(data, anchor_labels::Vector{L},
                      anchor_values::Vector{<:Real}) where {L}
    labels = _anchor_target_labels(data)
    r = length(anchor_labels)
    r >= 1 || throw(ArgumentError("Need at least 1 anchor, got none"))
    length(anchor_values) == r || throw(DimensionMismatch(
        "Got $r anchor labels but $(length(anchor_values)) anchor values"))
    allunique(anchor_labels) || throw(ArgumentError("Anchor labels must be unique"))
    groups = [[_resolve_anchor_label(labels, lbl)] for lbl in anchor_labels]
    return AnchoredData{typeof(data), L}(data, groups, Vector{Float64}(anchor_values))
end

# Group anchors: each measurement targets a group of items, modelled as the mean.
function AnchoredData(data, anchor_groups::Vector{<:AbstractVector{L}},
                      anchor_values::Vector{<:Real}) where {L}
    labels = _anchor_target_labels(data)
    G = length(anchor_groups)
    G >= 1 || throw(ArgumentError("Need at least 1 anchor group, got none"))
    length(anchor_values) == G || throw(DimensionMismatch(
        "Got $G anchor groups but $(length(anchor_values)) anchor values"))
    groups = Vector{Vector{Int}}(undef, G)
    for (g, grp) in enumerate(anchor_groups)
        isempty(grp) && throw(ArgumentError("Anchor group $g is empty"))
        allunique(grp) || throw(ArgumentError("Items within anchor group $g must be unique"))
        groups[g] = [_resolve_anchor_label(labels, lbl) for lbl in grp]
    end
    return AnchoredData{typeof(data), L}(data, groups, Vector{Float64}(anchor_values))
end

# Dict convenience: keys are labels (single-item) or label-vectors (groups).
function AnchoredData(data, anchors::AbstractDict{L, <:Real}) where {L}
    anchor_labels = collect(keys(anchors))
    anchor_values = [Float64(anchors[lbl]) for lbl in anchor_labels]
    return AnchoredData(data, anchor_labels, anchor_values)
end

# Pairs convenience for group anchors: `AnchoredData(data, ["a","b"] => 3.0, ["c"] => 4.0)`.
function AnchoredData(data, anchors::Pair{<:AbstractVector, <:Real}...)
    groups = [collect(first(p)) for p in anchors]
    values = Float64[last(p) for p in anchors]
    return AnchoredData(data, groups, values)
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
    AnchoredMLEResult

Maximum-likelihood fit of an [`Anchored`](@ref) model: the centred latent
strengths λ, the calibration coefficients `a`, `b` and noise variance `σ²` of
`y = a + b·λ + ε`, and the maximised joint log-likelihood. Query via
[`strengths`](@ref), [`calibration`](@ref), [`predict`](@ref) and
[`loglikelihood`](@ref).
"""
struct AnchoredMLEResult
    λ::Vector{Float64}       # centred latent strengths
    a::Float64              # calibration intercept
    b::Float64              # calibration slope
    σ²::Float64             # calibration noise variance
    loglik::Float64         # maximised joint log-likelihood
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

# ─────────────────────────── Covariate models ────────────────────────────
#
# A covariate model parameterises the latent strengths as a linear combination
# of item-level covariates: λ_i = z_iᵀβ, so that
# `logit P(i beats j) = (z_i − z_j)ᵀβ`. Estimation targets the coefficient
# vector β ∈ ℝ^p instead of the K free strengths; λ is recovered as Z·β.

"""
    Covariates(model)

Wrapper parameterising a comparative `model`'s latent strengths as a linear
combination of item covariates, `λ_i = z_iᵀβ`. Fit with a [`CovariateData`](@ref)
wrapping the comparison data and the item covariate matrix, e.g.

```julia
fit(BradleyTerryCovariates(), MLE(), CovariateData(data, Z, [:age, :size]))
```

See also [`BradleyTerryCovariates`](@ref).
"""
struct Covariates{M <: AbstractComparativeModel} <: AbstractComparativeModel
    model::M
end

"""
    BradleyTerryCovariates()

Alias for `Covariates(BradleyTerry())`: the Bradley–Terry model whose latent
strengths are a linear function of item covariates, `λ_i = z_iᵀβ`, so that
`logit P(i beats j) = (z_i − z_j)ᵀβ`. See [`Covariates`](@ref) and [`fit`](@ref).
"""
const BradleyTerryCovariates = Covariates{BradleyTerry}
BradleyTerryCovariates() = Covariates(BradleyTerry())

"""
    ThurstoneCaseVCovariates()

Alias for `Covariates(ThurstoneCaseV())`: the Thurstone Case V model whose latent
strengths are a linear function of item covariates, `λ_i = z_iᵀβ`, so that
`probit P(i beats j) = Φ((z_i − z_j)ᵀβ)`. See [`Covariates`](@ref) and
[`fit`](@ref).
"""
const ThurstoneCaseVCovariates = Covariates{ThurstoneCaseV}
ThurstoneCaseVCovariates() = Covariates(ThurstoneCaseV())

"""
    StepwiseMLE(; direction=:both, criterion=:AIC)

Stepwise maximum-likelihood variable selection for a [`Covariates`](@ref)
model. `direction` is `:forward`, `:backward`, or `:both`; `criterion` is `:AIC`
or `:BIC`. Greedily adds/removes covariates to optimise the information
criterion, then refits the selected subset. The result records the selected
covariate indices and the selection trace; query it with the usual accessors
([`coefficients`](@ref), [`strengths`](@ref)).
"""
struct StepwiseMLE <: InferenceMethod
    direction::Symbol
    criterion::Symbol
    function StepwiseMLE(; direction::Symbol=:both, criterion::Symbol=:AIC)
        direction in (:forward, :backward, :both) || throw(ArgumentError(
            "direction must be :forward, :backward or :both, got $direction"))
        criterion in (:AIC, :BIC) || throw(ArgumentError(
            "criterion must be :AIC or :BIC, got $criterion"))
        new(direction, criterion)
    end
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
    CovariateData(data, Z, names)
    CovariateData(data, Z)
    CovariateData(data, name => values, ...)

Comparison `data` augmented with an item covariate matrix `Z` (`K × p`, one row
per item in the order of `data.labels`) and covariate `names`. Used to fit
[`Covariates`](@ref) models, where `λ_i = z_iᵀβ`.

An overall intercept (a covariate constant across items) is **not** identifiable:
it cancels in the differences `z_i − z_j`, so such columns are rejected.
"""
struct CovariateData{L}
    data::PairwiseData{L}
    Z::Matrix{Float64}
    names::Vector{Symbol}
    function CovariateData(data::PairwiseData{L}, Z::AbstractMatrix,
                           names::Vector{Symbol}) where {L}
        K = length(data.labels)
        size(Z, 1) == K || throw(DimensionMismatch(
            "Z must have $K rows to match $K items, got $(size(Z, 1))"))
        size(Z, 2) == length(names) || throw(DimensionMismatch(
            "Z has $(size(Z, 2)) columns but $(length(names)) names given"))
        Zf = Matrix{Float64}(Z)
        for c in 1:size(Zf, 2)
            col = @view Zf[:, c]
            all(==(col[1]), col) && throw(ArgumentError(
                "Covariate $(names[c]) is constant across items; it cancels in " *
                "the comparison differences and is not identifiable. Drop it."))
        end
        new{L}(data, Zf, names)
    end
end
function CovariateData(data::PairwiseData, Z::AbstractMatrix)
    names = [Symbol("x", c) for c in 1:size(Z, 2)]
    return CovariateData(data, Z, names)
end
function CovariateData(data::PairwiseData, cols::Pair{Symbol, <:AbstractVector}...)
    names = Symbol[c.first for c in cols]
    Z = reduce(hcat, [Vector{Float64}(c.second) for c in cols])
    return CovariateData(data, Z, names)
end

"""
    CovariateMLEResult

Result of an [`MLE`](@ref) or [`StepwiseMLE`](@ref) [`Covariates`](@ref) fit:
the coefficient estimates `β`, their covariance `vcov`, the log-likelihood, the
item covariate matrix `Z`, covariate `names`, the indices of covariates retained
in the model (`selected`), and the selection `trace` (empty for plain MLE).
"""
struct CovariateMLEResult
    β::Vector{Float64}
    vcov::Matrix{Float64}
    loglik::Float64
    Z::Matrix{Float64}
    names::Vector{Symbol}
    selected::Vector{Int}
    trace::Vector{NamedTuple}
end

"""
    CovariateMCMCSamples

Posterior draws from a [`Bayesian`](@ref) [`Covariates`](@ref) fit: the
`n_samples × p` matrix of coefficient draws `β_samples`, the per-draw
log-likelihoods, the item covariate matrix `Z`, covariate `names`, and (for
[`SpikeSlabPrior`](@ref) only) the `inclusion` indicator draws.
"""
struct CovariateMCMCSamples
    β_samples::Matrix{Float64}        # n_samples × p
    loglikelihoods::Vector{Float64}
    inclusion::Union{Nothing, Matrix{Float64}}  # n_samples × p, spike-slab only
    Z::Matrix{Float64}
    names::Vector{Symbol}
    n_samples::Int
    n_burnin::Int
    thin::Int
end

# ──────────────────────── Rater-heterogeneity models ─────────────────────────
#
# A mixture in which rater r follows Bradley–Terry with probability q_r and
# guesses at random otherwise:
# `P(rater r judges i ≻ j) = q_r·σ(λᵢ − λⱼ) + (1 − q_r)/2`. The reliabilities
# q_r down-weight inattentive assessors and are themselves a quality signal.

"""
    RaterHeterogeneity(model)

Wrapper turning a comparative `model` into a rater-heterogeneity mixture: rater
`r` follows the base model with reliability `q_r` and guesses at random
otherwise, so `P(rater r judges i ≻ j) = q_r·σ(λᵢ − λⱼ) + (1 − q_r)/2`. Fit with
a [`RaterData`](@ref) holding per-rater comparisons. See also
[`BradleyTerryRaterHeterogeneity`](@ref).
"""
struct RaterHeterogeneity{M <: AbstractComparativeModel} <: AbstractComparativeModel
    model::M
end

"""
    BradleyTerryRaterHeterogeneity()

Alias for `RaterHeterogeneity(BradleyTerry())`: the Bradley–Terry rater
mixture `P(rater r judges i ≻ j) = q_r·σ(λᵢ − λⱼ) + (1 − q_r)/2`, with a
rater-specific reliability `q_r ∈ [0, 1]`. See [`RaterHeterogeneity`](@ref) and
[`fit`](@ref).
"""
const BradleyTerryRaterHeterogeneity = RaterHeterogeneity{BradleyTerry}
BradleyTerryRaterHeterogeneity() = RaterHeterogeneity(BradleyTerry())

"""
    Intransitive(model)

Wrapper adding a skew-symmetric per-pair term `γᵢⱼ = −γⱼᵢ` to a comparative
`model`'s linear predictor, `logit P(i ≻ j) = (λᵢ − λⱼ) + γᵢⱼ`, capturing
preference structure the unidimensional scale cannot explain. Fit with an
ordinary [`PairwiseData`](@ref). See also [`BradleyTerryIntransitive`](@ref).
"""
struct Intransitive{M <: AbstractComparativeModel} <: AbstractComparativeModel
    model::M
end

"""
    BradleyTerryIntransitive()

Alias for `Intransitive(BradleyTerry())`: the Bradley–Terry model with a
skew-symmetric intransitivity term, `logit P(i ≻ j) = (λᵢ − λⱼ) + γᵢⱼ`,
`γᵢⱼ = −γⱼᵢ`. See [`Intransitive`](@ref) and [`fit`](@ref).
"""
const BradleyTerryIntransitive = Intransitive{BradleyTerry}
BradleyTerryIntransitive() = Intransitive(BradleyTerry())

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

"""
    RaterData(winners, losers, raters; item_labels=nothing, rater_labels=nothing)

Per-rater pairwise comparison data for a [`RaterHeterogeneity`](@ref) fit. Each
comparison `c` records the `winners[c]` item that beat the `losers[c]` item, as
judged by rater `raters[c]` (all given by label). Item and rater labels are
inferred in order of first appearance unless `item_labels` / `rater_labels` are
supplied to fix the ordering.
"""
struct RaterData{L, R}
    winner::Vector{Int}        # item index of the winner of each comparison
    loser::Vector{Int}         # item index of the loser of each comparison
    rater::Vector{Int}         # rater index of each comparison
    labels::Vector{L}          # K item labels
    raters::Vector{R}          # M rater labels
end

function RaterData(winners::AbstractVector, losers::AbstractVector,
                   raters::AbstractVector; item_labels=nothing, rater_labels=nothing)
    n = length(winners)
    (length(losers) == n && length(raters) == n) || throw(DimensionMismatch(
        "winners, losers and raters must have equal length, got " *
        "$(length(winners)), $(length(losers)), $(length(raters))"))
    n >= 1 || throw(ArgumentError("Need at least 1 comparison, got none"))
    ilabels = item_labels === nothing ? unique(vcat(collect(winners), collect(losers))) :
              collect(item_labels)
    rlabels = rater_labels === nothing ? unique(collect(raters)) : collect(rater_labels)
    length(ilabels) >= 2 || throw(ArgumentError(
        "Need at least 2 distinct items, got $(length(ilabels))"))
    iidx = Dict(l => i for (i, l) in enumerate(ilabels))
    ridx = Dict(r => i for (i, r) in enumerate(rlabels))
    w = Vector{Int}(undef, n); l = Vector{Int}(undef, n); r = Vector{Int}(undef, n)
    for c in 1:n
        haskey(iidx, winners[c]) || throw(ArgumentError("Unknown item label $(winners[c])"))
        haskey(iidx, losers[c])  || throw(ArgumentError("Unknown item label $(losers[c])"))
        haskey(ridx, raters[c])  || throw(ArgumentError("Unknown rater label $(raters[c])"))
        w[c] = iidx[winners[c]]; l[c] = iidx[losers[c]]; r[c] = ridx[raters[c]]
        w[c] == l[c] && throw(ArgumentError("Comparison $c pits item $(winners[c]) against itself"))
    end
    return RaterData{eltype(ilabels), eltype(rlabels)}(w, l, r, ilabels, rlabels)
end

"""
    RaterMLEResult

Maximum-likelihood fit of a [`RaterHeterogeneity`](@ref) model: the centred
latent strengths λ, the rater reliabilities `q` (one per rater label), and the
maximised mixture log-likelihood. Query with [`strengths`](@ref),
[`rater_reliabilities`](@ref) and [`loglikelihood`](@ref).
"""
struct RaterMLEResult{R}
    λ::Vector{Float64}
    q::Vector{Float64}
    rater_labels::Vector{R}
    loglik::Float64
end

"""
    RaterMCMCSamples

Posterior draws from a [`Bayesian`](@ref) [`RaterHeterogeneity`](@ref) fit: the
`n_samples × K` matrix of latent-strength draws, the `n_samples × M` matrix of
rater-reliability draws `q`, the rater labels, and per-draw log-likelihoods.
"""
struct RaterMCMCSamples{R}
    λ_samples::Matrix{Float64}        # n_samples × K
    q_samples::Matrix{Float64}        # n_samples × M
    rater_labels::Vector{R}
    loglikelihoods::Vector{Float64}
    n_samples::Int
    n_burnin::Int
    thin::Int
end

"""
    IntransitiveMLEResult

Penalised maximum-likelihood fit of an [`Intransitive`](@ref) model: the centred
latent strengths λ, the skew-symmetric terms `γ` for each observed pair (`pairs`
holds the `(i, j)`, `i < j` indices), the ridge penalty scale `σ²γ`, and the
maximised penalised log-likelihood. Query with [`strengths`](@ref),
[`intransitivity`](@ref) and [`loglikelihood`](@ref).
"""
struct IntransitiveMLEResult
    λ::Vector{Float64}
    pairs::Vector{Tuple{Int, Int}}
    γ::Vector{Float64}
    σ²γ::Float64
    loglik::Float64
end

"""
    IntransitiveMCMCSamples

Posterior draws from a [`Bayesian`](@ref) [`Intransitive`](@ref) fit: the
`n_samples × K` matrix of latent-strength draws, the `n_samples × P` matrix of
skew-symmetric-term draws `γ` (`pairs` holds the `(i, j)`, `i < j` indices), the
sampled variances `σ²γ`, and per-draw log-likelihoods.
"""
struct IntransitiveMCMCSamples
    λ_samples::Matrix{Float64}        # n_samples × K
    γ_samples::Matrix{Float64}        # n_samples × P
    pairs::Vector{Tuple{Int, Int}}
    σ²γ_samples::Vector{Float64}
    loglikelihoods::Vector{Float64}
    n_samples::Int
    n_burnin::Int
    thin::Int
end
