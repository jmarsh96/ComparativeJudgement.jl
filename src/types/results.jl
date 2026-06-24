"""
    FittedComparativeModel <: StatsAPI.StatisticalModel

The result of [`fit`](@ref): bundles the model, the inference method, the raw
fitting result (an `Optim` result for [`MLE`](@ref), a posterior-draws container
for [`Bayesian`](@ref)), the item labels, the `data` the model was fit to, and
convergence info. As a `StatisticalModel` it supports the StatsAPI interface
([`coef`](@ref), [`vcov`](@ref), [`stderror`](@ref), [`confint`](@ref),
[`loglikelihood`](@ref), [`dof`](@ref), [`nobs`](@ref), [`aic`](@ref),
[`bic`](@ref), …) alongside the domain accessors ([`strengths`](@ref),
[`probability`](@ref), [`posterior_mean`](@ref), [`predict`](@ref), …). Query it
through these accessors rather than via `result` directly.
"""
struct FittedComparativeModel{M <: AbstractComparativeModel, I <: InferenceMethod, R, L, D} <: StatisticalModel
    model::M
    method::I
    result::R
    labels::Vector{L}
    data::D
    converged::Bool
    iterations::Int
end

# ─── Bradley–Terry / Thurstone (plain) ───────────────────────────────────────

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

# ─── Anchored ────────────────────────────────────────────────────────────────

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

# ─── Covariates ──────────────────────────────────────────────────────────────

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

# ─── Rater heterogeneity ─────────────────────────────────────────────────────

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

# ─── Intransitivity ──────────────────────────────────────────────────────────

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
