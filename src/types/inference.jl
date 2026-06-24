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
