module ComparativeJudgement

using Optim: Optim, optimize, LBFGS

export AbstractComparativeModel, PairwiseModel, RankingModel
export BradleyTerry, PlackettLuce, ThurstoneCaseV
export InferenceMethod, MLE, Bayesian
export PairwiseData, FittedComparativeModel
export fit, loglikelihood, probability

include("types.jl")
include("interface.jl")
include("models/bradley_terry.jl")
include("utils.jl")

end
