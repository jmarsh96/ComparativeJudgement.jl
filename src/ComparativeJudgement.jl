module ComparativeJudgement

using LinearAlgebra: LinearAlgebra, Diagonal, Symmetric, cholesky, cholesky!, ldiv!, mul!, I, dot, inv, isdiag, diag
using Statistics: mean, std, var, cor, quantile
using Random: AbstractRNG, randn, randn!, rand, randexp, randperm, Random
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
# Model checking — diagnostics (single model)
export pointwise_loglikelihood, nparams, nobs, deviance, aic, bic
export waic, loo, ssr, split_half_reliability
export WAICResult, LOOResult, ReliabilityResult
# Model checking — comparison (between models)
export train_test_split, kfold, log_loss, crossvalidate
export lrtest, rank_correlation, top_k_agreement, boundary_agreement, compare
export CVResult, LRTResult, ModelComparisonTable

include("types/models.jl")
include("types/inference.jl")
include("types/priors.jl")
include("types/data.jl")
include("types/results.jl")
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

include("diagnostics/pointwise.jl")
include("diagnostics/information_criteria.jl")
include("diagnostics/waic.jl")
include("diagnostics/loo.jl")
include("diagnostics/reliability.jl")
include("comparison/predictive.jl")
include("comparison/likelihood_ratio.jl")
include("comparison/robustness.jl")
include("comparison/compare.jl")

end
