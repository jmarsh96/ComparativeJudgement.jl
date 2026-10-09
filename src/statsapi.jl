# ─── StatsAPI interface ───────────────────────────────────────────────────────
#
# `FittedComparativeModel <: StatsAPI.StatisticalModel`, so implementing
# `loglikelihood`/`dof`/`nobs`/`vcov` here gives `aic`/`aicc`/`bic`/`stderror`
# for free from StatsAPI's default methods (all model-only — the fit carries its
# data). `coef` returns the regression coefficients β for a covariate model and
# the latent strengths λ for the others; `coefnames` the covariate names or item
# labels accordingly. The domain accessors (`strengths`, `calibration`,
# `rater_reliabilities`, `intransitivity`, …) remain the CJ-specific interface.

# ─── Counts ───────────────────────────────────────────────────────────────────

_nobs(d::PairwiseData) = sum(d.wins)
_nobs(d::CovariateData) = sum(d.data.wins)
_nobs(d::AnchoredData) = _nobs(d.data)
_nobs(d::RaterData) = length(d.winner)

"""
    nobs(fitted)

Number of pairwise comparisons the model was fit to (the total trial count), used
as the sample size in the BIC penalty.
"""
nobs(f::FittedComparativeModel) = _nobs(f.data)

"""
    dof(fitted)

Number of free parameters of `fitted`: `K-1` for plain Bradley–Terry/Thurstone,
the number of selected covariates for a covariate model, `(K-1)+3` for an anchored
model (intercept, slope, noise variance), `K+M` for a rater-heterogeneity model
(`M` raters), and `(K-1)+P` for an intransitive model (`P` observed pairs).
"""
dof(f::FittedComparativeModel{<:Union{BradleyTerry, ThurstoneCaseV}}) = length(f.labels) - 1
dof(f::FittedComparativeModel{<:Anchored}) = (length(f.labels) - 1) + 3
dof(f::FittedComparativeModel{<:RaterHeterogeneity}) = length(f.labels) + length(f.data.raters)
dof(f::FittedComparativeModel{<:Intransitive}) = (length(f.labels) - 1) + length(f.result.pairs)
dof(f::FittedComparativeModel{<:Covariates, I, CovariateMLEResult}) where {I} = length(f.result.selected)
dof(f::FittedComparativeModel{<:Covariates, Bayesian, CovariateMCMCSamples}) = size(f.result.β_samples, 2)

# ─── Log-likelihood (scalar + pointwise) ──────────────────────────────────────

"""
    loglikelihood(fitted)
    loglikelihood(fitted, :)

Log-likelihood of the fitted model. `loglikelihood(fitted)` is the scalar
log-likelihood at the point estimate ([`MLE`](@ref)) or posterior mean
([`Bayesian`](@ref)); `loglikelihood(fitted, :)` is the vector of
per-observation contributions (one per observed pair, or per rater-pair cell for
the rater model), with `sum(loglikelihood(fitted, :)) == loglikelihood(fitted)`.
For anchored fits only the comparison terms enter. See
[`mcmc_loglikelihoods`](@ref) for the per-draw MCMC trace.
"""
loglikelihood(f::FittedComparativeModel, ::Colon) = _pointwise_at_point(f)
loglikelihood(f::FittedComparativeModel) = sum(_pointwise_at_point(f))

mcmc_loglikelihoods(f::FittedComparativeModel{M, Bayesian}) where {M <: AbstractComparativeModel} =
    f.result.loglikelihoods
mcmc_loglikelihoods(::FittedComparativeModel{M, I}) where {M, I} = throw(ArgumentError(
    "mcmc_loglikelihoods is only defined for Bayesian fits"))

"""
    deviance(fitted)

Residual deviance `-2·loglikelihood(fitted)`.
"""
deviance(f::FittedComparativeModel) = -2.0 * loglikelihood(f)

# Bayesian fits: the likelihood-penalty information criteria are not appropriate
# (the effective number of parameters differs from the nominal `dof`); redirect
# to the predictive criteria.
for fn in (:aic, :aicc, :bic)
    @eval $fn(::FittedComparativeModel{M, Bayesian}) where {M <: AbstractComparativeModel} =
        throw(ArgumentError(string($(QuoteNode(fn))) * " is defined for maximum-likelihood " *
            "fits; use `waic` or `loo` for a Bayesian fit."))
end

"""
    aic(fitted)

Akaike information criterion `-2·loglikelihood(fitted) + 2·dof(fitted)` of an
[`MLE`](@ref) fit (lower is better). Defined for maximum-likelihood fits only —
use [`waic`](@ref)/[`loo`](@ref) for a [`Bayesian`](@ref) fit. Comes from
StatsAPI's default via [`loglikelihood`](@ref) and [`dof`](@ref).
"""
aic

"""
    bic(fitted)

Bayesian information criterion `-2·loglikelihood(fitted) + dof(fitted)·log(nobs(fitted))`
of an [`MLE`](@ref) fit (lower is better). Maximum-likelihood fits only — use
[`waic`](@ref)/[`loo`](@ref) for a [`Bayesian`](@ref) fit.
"""
bic

"""
    aicc(fitted)

Small-sample corrected AIC of an [`MLE`](@ref) fit (lower is better). Maximum-
likelihood fits only — use [`waic`](@ref)/[`loo`](@ref) for a [`Bayesian`](@ref) fit.
"""
aicc

"""
    stderror(fitted)

Standard errors of [`coef`](@ref), `sqrt.(diag(vcov(fitted)))`. Available wherever
[`vcov`](@ref) is.
"""
stderror

# ─── Coefficients and names ────────────────────────────────────────────────────

"""
    coef(fitted)

Model coefficients: the covariate coefficients β for a [`Covariates`](@ref) fit,
and the latent strengths λ (see [`strengths`](@ref)) for the plain, anchored,
rater-heterogeneity and intransitive models. Names are given by
[`coefnames`](@ref).
"""
coef(f::FittedComparativeModel{<:Covariates, I, CovariateMLEResult}) where {I} = copy(f.result.β)
coef(f::FittedComparativeModel{<:Covariates, Bayesian, CovariateMCMCSamples}) =
    vec(mean(f.result.β_samples, dims=1))
coef(f::FittedComparativeModel) = strengths(f)

"""
    coefnames(fitted)

Names of the coefficients returned by [`coef`](@ref): covariate names for a
[`Covariates`](@ref) fit, item labels (as strings) otherwise.
"""
coefnames(f::FittedComparativeModel{<:Covariates, I, CovariateMLEResult}) where {I} =
    string.(f.result.names[f.result.selected])
coefnames(f::FittedComparativeModel{<:Covariates, Bayesian, CovariateMCMCSamples}) =
    string.(f.result.names)
coefnames(f::FittedComparativeModel) = string.(f.labels)

# ─── Covariance, standard errors, confidence/credible intervals ───────────────

# Centred latent-strength posterior draws (S×K) for the non-covariate models.
_strength_draws(f::FittedComparativeModel{<:Union{BradleyTerry, ThurstoneCaseV}, Bayesian}) =
    (S = f.result.samples; S .- mean(S, dims=2))
_strength_draws(f::FittedComparativeModel{<:Anchored, Bayesian}) =
    (S = f.result.λ_samples; S .- mean(S, dims=2))
_strength_draws(f::FittedComparativeModel{<:RaterHeterogeneity, Bayesian}) =
    (S = f.result.λ_samples; S .- mean(S, dims=2))
_strength_draws(f::FittedComparativeModel{<:Intransitive, Bayesian}) =
    (S = f.result.λ_samples; S .- mean(S, dims=2))

"""
    SingularInformationError(msg)

Thrown by [`vcov`](@ref)/[`stderror`](@ref)/[`confint`](@ref) for a plain
Bradley–Terry/Thurstone [`MLE`](@ref) fit whose observed-information matrix is
singular: the comparison design does not identify all the strengths (see
[`design_connectivity`](@ref)), so frequentist standard errors do not exist. Fit
the model with [`Bayesian`](@ref) inference for regularised uncertainty instead.
"""
struct SingularInformationError <: Exception
    msg::String
end

Base.showerror(io::IO, e::SingularInformationError) = print(io, "SingularInformationError: ", e.msg)

_singular_information(cause::AbstractString) = SingularInformationError(
    "observed-information covariance is singular ($cause): the comparison " *
    "design does not identify all Bradley–Terry/Thurstone strengths. Fit " *
    "with Bayesian inference for regularised vcov/stderror/confint.")

# Observed-information covariance of the centred strengths for a plain MLE fit.
# The information matrix is singular whenever the design fails to identify all
# strengths (a disconnected win-graph, or undefeated/winless items). The fitted
# strengths then diverge and the Hessian at the optimiser's stopping point is only
# near-singular, so the design is checked up front rather than relying on the
# inversion to fail; either way the cause is surfaced as a domain-specific
# `SingularInformationError` instead of a bare `LinearAlgebra.SingularException`.
# A ridge-penalised fit is always finite, so it skips the design check and inverts
# the penalised information instead.
function _strength_vcov(f::FittedComparativeModel{<:Union{BradleyTerry, ThurstoneCaseV}, MLE})
    K = length(f.labels)
    ridge = f.method.ridge
    if iszero(ridge)
        detail = _degeneracy_detail(f.data)
        detail === nothing || throw(_singular_information(detail))
    end
    θ̂ = collect(float.(Optim.minimizer(f.result)))
    agg = _aggregate_pairs(_pairwise(f.data).wins, K)
    H = f.model isa BradleyTerry ? _bt_neg_hessian(θ̂, agg) : _tcv_neg_hessian(θ̂, agg)
    _ridge_hessian!(H, ridge)
    Σfree = try
        inv(Symmetric(H))                                 # covariance of the free strengths
    catch err
        err isa LinearAlgebra.SingularException || rethrow()
        throw(_singular_information("the information matrix is numerically rank-deficient"))
    end
    Σfull = zeros(K, K)
    Σfull[2:K, 2:K] .= Σfree
    C = Matrix{Float64}(I, K, K) .- fill(1.0 / K, K, K)    # centring projection
    return C * Σfull * C'
end

"""
    vcov(fitted)

Variance–covariance matrix of [`coef`](@ref): the coefficient covariance for a
covariate fit, and the strength covariance for the others (observed information
for a plain Bradley–Terry/Thurstone [`MLE`](@ref) fit, the posterior covariance
for a [`Bayesian`](@ref) fit). Not available for an MLE fit of the anchored,
rater-heterogeneity or intransitive models. [`stderror`](@ref) and
[`confint`](@ref) build on it.
"""
vcov(f::FittedComparativeModel{<:Covariates, I, CovariateMLEResult}) where {I} = copy(f.result.vcov)
vcov(f::FittedComparativeModel{<:Covariates, Bayesian, CovariateMCMCSamples}) = cov(f.result.β_samples)
vcov(f::FittedComparativeModel{<:Union{BradleyTerry, ThurstoneCaseV}, MLE}) = _strength_vcov(f)
vcov(f::FittedComparativeModel{M, Bayesian}) where {M} = cov(_strength_draws(f))
vcov(::FittedComparativeModel{<:Union{Anchored, RaterHeterogeneity, Intransitive}, MLE}) = throw(ArgumentError(
    "vcov is not available for an MLE fit of this model; use a Bayesian fit for a " *
    "posterior covariance, or `coef` for point estimates."))

# Posterior draws of the coefficients (β for covariate models, λ otherwise).
_coef_draws(f::FittedComparativeModel{<:Covariates, Bayesian, CovariateMCMCSamples}) = f.result.β_samples
_coef_draws(f::FittedComparativeModel) = _strength_draws(f)

"""
    confint(fitted; level=0.95)

Confidence/credible intervals for [`coef`](@ref), as a `k × 2` matrix of
`(lower, upper)` rows. Wald intervals `coef ± z·stderror` for [`MLE`](@ref) fits,
posterior quantile intervals for [`Bayesian`](@ref) fits.
"""
function confint(f::FittedComparativeModel; level::Real=0.95)
    0.0 < level < 1.0 || throw(ArgumentError("level must be in (0, 1), got $level"))
    α = (1.0 - level) / 2.0
    if f.method isa Bayesian
        D = _coef_draws(f)
        lo = [quantile(view(D, :, k), α) for k in 1:size(D, 2)]
        hi = [quantile(view(D, :, k), 1.0 - α) for k in 1:size(D, 2)]
        return hcat(lo, hi)
    end
    c = coef(f); se = stderror(f); z = _norm_quantile(1.0 - α)
    return hcat(c .- z .* se, c .+ z .* se)
end
