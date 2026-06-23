module ComparativeJudgement

using LinearAlgebra: LinearAlgebra, Diagonal, Symmetric, cholesky, cholesky!, ldiv!, mul!, I, dot, inv, isdiag, diag
using Statistics: mean, std, quantile
using Random: AbstractRNG, randn, randn!, rand, randexp, Random
using Optim: Optim, optimize, LBFGS

export AbstractComparativeModel, PairwiseModel, RankingModel
export BradleyTerry, PlackettLuce, ThurstoneCaseV
export Anchored, BradleyTerryAnchored, ThurstoneCaseVAnchored
export Covariates, BradleyTerryCovariates, ThurstoneCaseVCovariates
export InferenceMethod, MLE, Bayesian, StepwiseMLE
export PairwiseData, AnchoredData, CovariateData, FittedComparativeModel
export AbstractPrior, NormalPrior, InverseGammaPrior, AnchoredPrior
export HorseshoePrior, SpikeSlabPrior
export BTMCMCSamples, AnchoredMCMCSamples, AnchoredMLEResult, CovariateMLEResult, CovariateMCMCSamples
export fit, loglikelihood, probability, predict, calibration, strengths
export coefficients, coefficient_std, coefficient_intervals, inclusion_probabilities
export posterior_mean, posterior_std, credible_interval

include("types.jl")
include("interface.jl")
include("utils.jl")
include("polya_gamma.jl")
include("models/bradley_terry.jl")
include("models/bradley_terry_anchored.jl")
include("models/bradley_terry_covariates.jl")
include("models/thurstone_case_v.jl")
include("models/thurstone_case_v_anchored.jl")
include("models/thurstone_case_v_covariates.jl")

end
