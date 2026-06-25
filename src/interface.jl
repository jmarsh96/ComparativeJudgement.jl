# Generic functions for the domain-specific accessors (declared here with
# docstrings; each model file adds the methods). The StatsAPI generics — `fit`,
# `loglikelihood`, `predict`, `coef`, `coefnames`, `vcov`, `stderror`, `confint`,
# `dof`, `nobs`, `deviance`, `aic`, `bic`, `aicc` — are imported from StatsAPI and
# extended in `src/statsapi.jl`.

"""
    probability(fitted, i, j)
    probability(fitted, label_i, label_j)

Probability that item `i` beats item `j` under the fitted model. Items may be
given by index or by label. For Bayesian fits this is the posterior mean of
the win probability.
"""
function probability end

"""
    strengths(fitted)

Estimated latent strengths λ, one per item, in the order of `fitted.labels`.
For [`MLE`](@ref) fits these are the point estimates (centred to sum to zero);
for [`Bayesian`](@ref) fits this is the posterior mean, equivalent to
[`posterior_mean`](@ref). See also [`coef`](@ref), which returns these strengths
for the plain/anchored/rater/intransitive models and the coefficients β for a
covariate model.
"""
function strengths end

"""
    inclusion_probabilities(fitted)

Posterior inclusion probabilities per covariate from a [`Bayesian`](@ref)
[`Covariates`](@ref) fit with a [`SpikeSlabPrior`](@ref), keyed by covariate
name.
"""
function inclusion_probabilities end

"""
    posterior_mean(fitted)

Posterior mean of the latent strengths λ of a [`Bayesian`](@ref) fit, one per
item, in the order of `fitted.labels`.
"""
function posterior_mean end

"""
    posterior_std(fitted)

Posterior standard deviation of the latent strengths λ of a
[`Bayesian`](@ref) fit, one per item, in the order of `fitted.labels`.
"""
function posterior_std end

"""
    credible_interval(fitted, k; prob=0.95)

Symmetric posterior credible interval `(lo, hi)` for the latent strength of
item `k` from a [`Bayesian`](@ref) fit.
"""
function credible_interval end

"""
    mcmc_loglikelihoods(fitted)

The per-draw total log-likelihood trace of a [`Bayesian`](@ref) fit, one value
per retained posterior draw — a sampler diagnostic. (Distinct from
[`loglikelihood`](@ref)`(fitted)`, the scalar log-likelihood at the posterior
mean, and [`loglikelihood`](@ref)`(fitted, :)`, the pointwise-over-observations
vector.)
"""
function mcmc_loglikelihoods end

"""
    calibration(fitted)

Posterior means of the calibration parameters of an [`Anchored`](@ref) fit,
as a named tuple `(a = ..., b = ..., σ² = ...)` for the anchor model
`y = a + b·λ + ε`, `ε ~ N(0, σ²)`.
"""
function calibration end

"""
    rater_reliabilities(fitted)

Estimated rater reliabilities `q_r ∈ [0, 1]` of a [`RaterHeterogeneity`](@ref)
fit, as a named tuple keyed by rater label. Point estimates for [`MLE`](@ref)
fits, posterior means for [`Bayesian`](@ref) fits. A low `q_r` flags a rater
whose judgements are close to random.
"""
function rater_reliabilities end

"""
    intransitivity(fitted)

Estimated skew-symmetric intransitivity terms `γᵢⱼ = −γⱼᵢ` of an
[`Intransitive`](@ref) fit, as a `K × K` matrix (zero on the diagonal and on
unobserved pairs). Point estimates for [`MLE`](@ref) fits, posterior means for
[`Bayesian`](@ref) fits. Entries far from zero mark pairs the unidimensional
scale cannot explain.
"""
function intransitivity end
