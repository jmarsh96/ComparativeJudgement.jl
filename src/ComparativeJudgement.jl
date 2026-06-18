module ComparativeJudgement

using LinearAlgebra: LinearAlgebra, Diagonal, Symmetric, cholesky, cholesky!, ldiv!, mul!, I, dot, inv, isdiag, diag
using Statistics: mean, std, quantile
using Random: AbstractRNG, randn, randn!, rand, randexp, Random
using Optim: Optim, optimize, LBFGS

export AbstractComparativeModel, PairwiseModel, RankingModel
export BradleyTerry, PlackettLuce, ThurstoneCaseV
export Anchored, BradleyTerryAnchored
export Covariates, BradleyTerryCovariates
export InferenceMethod, MLE, Bayesian, StepwiseMLE
export PairwiseData, AnchoredData, CovariateData, FittedComparativeModel
export AbstractPrior, NormalPrior, InverseGammaPrior, AnchoredPrior
export HorseshoePrior, SpikeSlabPrior
export BTMCMCSamples, AnchoredMCMCSamples, CovariateMLEResult, CovariateMCMCSamples
export fit, loglikelihood, probability, predict, calibration, strengths
export coefficients, coefficient_std, coefficient_intervals, inclusion_probabilities
export posterior_mean, posterior_std, credible_interval

include("types.jl")
include("interface.jl")
include("utils.jl")
include("polya_gamma.jl")
include("models/bradley_terry.jl")
include("models/bradley_terry_covariates.jl")

end
