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
