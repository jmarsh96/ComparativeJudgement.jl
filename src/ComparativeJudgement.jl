module ComparativeJudgement

using LinearAlgebra: LinearAlgebra, Diagonal, Symmetric, cholesky, cholesky!, ldiv!, mul!, I, dot, inv, isdiag, diag
using Statistics: mean, std, quantile
using Random: AbstractRNG, randn, randn!, rand, randexp, Random
using Optim: Optim, optimize, LBFGS

export AbstractComparativeModel, PairwiseModel, RankingModel
export BradleyTerry, PlackettLuce, ThurstoneCaseV
export Anchored, BradleyTerryAnchored, ThurstoneCaseVAnchored
export Covariates, BradleyTerryCovariates, ThurstoneCaseVCovariates
export RaterHeterogeneity, BradleyTerryRaterHeterogeneity
export Intransitive, BradleyTerryIntransitive
export InferenceMethod, MLE, Bayesian, StepwiseMLE
export PairwiseData, AnchoredData, CovariateData, RaterData, FittedComparativeModel
export AbstractPrior, NormalPrior, InverseGammaPrior, AnchoredPrior
export HorseshoePrior, SpikeSlabPrior, BetaPrior, RaterHeterogeneityPrior, IntransitivityPrior
export BTMCMCSamples, AnchoredMCMCSamples, AnchoredMLEResult, CovariateMLEResult, CovariateMCMCSamples
export RaterMLEResult, RaterMCMCSamples, IntransitiveMLEResult, IntransitiveMCMCSamples
export fit, loglikelihood, probability, predict, calibration, strengths
export coefficients, coefficient_std, coefficient_intervals, inclusion_probabilities
export posterior_mean, posterior_std, credible_interval
export rater_reliabilities, intransitivity

include("types.jl")
include("interface.jl")
include("utils.jl")
include("polya_gamma.jl")
include("models/bradley_terry.jl")
include("models/bradley_terry_anchored.jl")
include("models/bradley_terry_covariates.jl")
include("models/bradley_terry_rater_heterogeneity.jl")
include("models/bradley_terry_intransitivity.jl")
include("models/thurstone_case_v.jl")
include("models/thurstone_case_v_anchored.jl")
include("models/thurstone_case_v_covariates.jl")

end
