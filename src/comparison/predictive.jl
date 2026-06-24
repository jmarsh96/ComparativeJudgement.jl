# ─── Out-of-sample predictive scoring ────────────────────────────────────────
#
# Splits the comparisons into training and test sets by partitioning the
# *individual* comparisons (each directed win is one comparison), fits a model to
# the training set, and scores the held-out comparisons by the mean logarithmic
# loss. This compares models on exactly the quantity of interest — how well they
# predict unseen comparisons — and assumes no candidate is the true model.

# Expand a wins matrix to a list of individual (winner, loser) index comparisons.
function _pairwise_records(wins::Matrix{Int})
    K = size(wins, 1)
    W = Int[]; L = Int[]
    @inbounds for i in 1:K, j in 1:K
        i == j && continue
        for _ in 1:wins[i, j]
            push!(W, i); push!(L, j)
        end
    end
    return W, L
end

function _pairwise_from_records(W, L, K)
    wins = zeros(Int, K, K)
    @inbounds for c in eachindex(W)
        wins[W[c], L[c]] += 1
    end
    return wins
end

# Rebuild a RaterData from a subset of its comparison records, preserving the
# full item/rater label ordering so item and rater indices stay consistent.
function _rater_subset(rd::RaterData, idx)
    W = [rd.labels[rd.winner[c]] for c in idx]
    L = [rd.labels[rd.loser[c]] for c in idx]
    R = [rd.raters[rd.rater[c]] for c in idx]
    return RaterData(W, L, R; item_labels=rd.labels, rater_labels=rd.raters)
end

function _train_test_indices(n::Int, frac::Float64, rng)
    0.0 < frac < 1.0 || throw(ArgumentError("frac must be in (0, 1), got $frac"))
    p = randperm(rng, n)
    ntr = clamp(round(Int, frac * n), 1, n - 1)
    return p[1:ntr], p[(ntr + 1):n]
end

# Interleaved (balanced) k-fold index partition.
_index_folds(n::Int, k::Int, rng) = (p = randperm(rng, n); [p[f:k:n] for f in 1:k])

"""
    train_test_split(data; frac=0.8, rng=Random.default_rng())

Randomly partition the individual comparisons in `data` into a training set
(fraction `frac`) and a complementary test set, returning `(train, test)` of the
same container type ([`PairwiseData`](@ref), [`CovariateData`](@ref) or
[`RaterData`](@ref)). Used for out-of-sample scoring and, with `frac=0.5`, for
[`split_half_reliability`](@ref).
"""
function train_test_split(data::PairwiseData; frac::Float64=0.8,
                          rng::AbstractRNG=Random.default_rng())
    W, L = _pairwise_records(data.wins)
    length(W) >= 2 || throw(ArgumentError("need at least 2 comparisons to split"))
    tr, te = _train_test_indices(length(W), frac, rng)
    K = length(data.labels)
    return PairwiseData(_pairwise_from_records(W[tr], L[tr], K), data.labels),
           PairwiseData(_pairwise_from_records(W[te], L[te], K), data.labels)
end

function train_test_split(cd::CovariateData; frac::Float64=0.8,
                          rng::AbstractRNG=Random.default_rng())
    tr, te = train_test_split(cd.data; frac=frac, rng=rng)
    return CovariateData(tr, cd.Z, cd.names), CovariateData(te, cd.Z, cd.names)
end

function train_test_split(rd::RaterData; frac::Float64=0.8,
                          rng::AbstractRNG=Random.default_rng())
    n = length(rd.winner)
    n >= 2 || throw(ArgumentError("need at least 2 comparisons to split"))
    tr, te = _train_test_indices(n, frac, rng)
    return _rater_subset(rd, tr), _rater_subset(rd, te)
end

"""
    kfold(data; k=5, rng=Random.default_rng())

Partition the individual comparisons in `data` into `k` folds, returning a vector
of `(train, test)` pairs in which each fold serves once as the test set. Used by
[`crossvalidate`](@ref).
"""
function kfold(data::PairwiseData; k::Int=5, rng::AbstractRNG=Random.default_rng())
    W, L = _pairwise_records(data.wins)
    n = length(W)
    n >= k || throw(ArgumentError("need at least k=$k comparisons, got $n"))
    folds = _index_folds(n, k, rng)
    K = length(data.labels)
    out = Tuple{PairwiseData, PairwiseData}[]
    for f in 1:k
        te = folds[f]
        tr = reduce(vcat, folds[setdiff(1:k, f)])
        push!(out, (PairwiseData(_pairwise_from_records(W[tr], L[tr], K), data.labels),
                    PairwiseData(_pairwise_from_records(W[te], L[te], K), data.labels)))
    end
    return out
end

function kfold(cd::CovariateData; k::Int=5, rng::AbstractRNG=Random.default_rng())
    return [(CovariateData(tr, cd.Z, cd.names), CovariateData(te, cd.Z, cd.names))
            for (tr, te) in kfold(cd.data; k=k, rng=rng)]
end

function kfold(rd::RaterData; k::Int=5, rng::AbstractRNG=Random.default_rng())
    n = length(rd.winner)
    n >= k || throw(ArgumentError("need at least k=$k comparisons, got $n"))
    folds = _index_folds(n, k, rng)
    out = Tuple{RaterData, RaterData}[]
    for f in 1:k
        te = folds[f]
        tr = reduce(vcat, folds[setdiff(1:k, f)])
        push!(out, (_rater_subset(rd, tr), _rater_subset(rd, te)))
    end
    return out
end

"""
    log_loss(fitted, test_data)

Mean logarithmic loss of `fitted` on held-out `test_data`: the average of
`-log p̂` over the test comparisons, where `p̂` is the probability the model
assigns to the observed winner. Lower is better; the logarithmic loss rewards
well-calibrated probabilities and penalises confident errors.
"""
function log_loss(fitted::FittedComparativeModel, test::PairwiseData)
    K = length(test.labels)
    tot = 0.0; n = 0
    @inbounds for i in 1:K, j in 1:K
        i == j && continue
        c = test.wins[i, j]
        c == 0 && continue
        p = clamp(probability(fitted, i, j), 1e-12, 1.0 - 1e-12)
        tot += -c * log(p)
        n += c
    end
    n == 0 && throw(ArgumentError("test set has no comparisons"))
    return tot / n
end

log_loss(fitted::FittedComparativeModel, test::CovariateData) = log_loss(fitted, test.data)

"""
    CVResult

Result of [`crossvalidate`](@ref): the `mean_logloss` across folds, the
`per_fold` losses, and the number of folds `k`.
"""
struct CVResult
    mean_logloss::Float64
    per_fold::Vector{Float64}
    k::Int
end

function Base.show(io::IO, r::CVResult)
    println(io, "CVResult (", r.k, "-fold)")
    print(io, "  mean log loss = ", round(r.mean_logloss, digits=4),
          " (per-fold sd ", round(std(r.per_fold), digits=4), ")")
end

"""
    crossvalidate(model, method, data; k=5, rng=Random.default_rng(), prior=nothing)

`k`-fold cross-validated predictive log loss: refit `model`/`method` on each
training fold and score the held-out fold with [`log_loss`](@ref). Returns a
[`CVResult`](@ref); lower mean log loss indicates better out-of-sample
prediction. `prior` is forwarded to [`fit`](@ref) for [`Bayesian`](@ref) fits.
"""
function crossvalidate(model, method, data; k::Int=5,
                       rng::AbstractRNG=Random.default_rng(), prior=nothing)
    losses = Float64[]
    for (train, test) in kfold(data; k=k, rng=rng)
        f = _refit(model, method, train, prior; rng=rng)
        push!(losses, log_loss(f, test))
    end
    return CVResult(mean(losses), losses, k)
end
