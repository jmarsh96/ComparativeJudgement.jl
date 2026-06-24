# ─── Rank-correlation and decision-level robustness ──────────────────────────
#
# The technique that bears most directly on whether a CJ study's conclusions
# survive a change of model: fit competing models to the same comparisons and
# compare the resulting scales, both by rank correlation over the whole scale and
# — more importantly — by agreement on the operational decisions (the top k, a
# grade boundary), which can disagree even when the rank correlation is high.

# Align two fits' strengths by item label; errors on a different item set.
function _aligned_strengths(f1::FittedComparativeModel, f2::FittedComparativeModel)
    Set(f1.labels) == Set(f2.labels) || throw(ArgumentError(
        "the two fits must cover the same items"))
    s1 = strengths(f1)
    s2 = strengths(f2)
    pos = Dict(l => i for (i, l) in enumerate(f2.labels))
    return s1, [s2[pos[l]] for l in f1.labels]
end

"""
    rank_correlation(fit1, fit2; method=:spearman)

Correlation between the latent strengths of two fits over the common items
(aligned by label). `method` is `:spearman` (default), `:kendall`, or
`:pearson`. A high value indicates the rank order is stable across the two
models; note that even ≈ 0.98 can mask disagreement at decision boundaries, so
pair this with [`top_k_agreement`](@ref) / [`boundary_agreement`](@ref).
"""
function rank_correlation(fit1::FittedComparativeModel, fit2::FittedComparativeModel;
                          method::Symbol=:spearman)
    s1, s2 = _aligned_strengths(fit1, fit2)
    method === :spearman && return _corspearman(s1, s2)
    method === :kendall && return _corkendall(s1, s2)
    method === :pearson && return cor(s1, s2)
    throw(ArgumentError("method must be :spearman, :kendall or :pearson, got $method"))
end

# Labels of the top-`k` items by strength.
_top_labels(f::FittedComparativeModel, k::Int) =
    Set(f.labels[partialsortperm(strengths(f), 1:k, rev=true)])

"""
    top_k_agreement(fit1, fit2, k)

Fraction of the top-`k` items (by strength) shared between two fits: the size of
the intersection of the two top-`k` sets divided by `k`. Measures whether the
two models agree on the most important items.
"""
function top_k_agreement(fit1::FittedComparativeModel, fit2::FittedComparativeModel, k::Integer)
    Set(fit1.labels) == Set(fit2.labels) || throw(ArgumentError(
        "the two fits must cover the same items"))
    n = length(fit1.labels)
    1 <= k <= n || throw(ArgumentError("k must be in 1:$n, got $k"))
    return length(intersect(_top_labels(fit1, k), _top_labels(fit2, k))) / k
end

"""
    boundary_agreement(fit1, fit2, threshold)

Agreement of two fits on a decision boundary: each item is classified by whether
its strength is at or above `threshold`, and the result reports the fraction of
items the two models place on the same side, together with the 2×2 counts
(`both_above`, `both_below`, `disagree`). Two scales can correlate highly yet
disagree on a material fraction of borderline cases.
"""
function boundary_agreement(fit1::FittedComparativeModel, fit2::FittedComparativeModel,
                            threshold::Real)
    s1, s2 = _aligned_strengths(fit1, fit2)
    a1 = s1 .>= threshold
    a2 = s2 .>= threshold
    both_above = count(a1 .& a2)
    both_below = count((.!a1) .& (.!a2))
    disagree = length(s1) - both_above - both_below
    return (agreement = (both_above + both_below) / length(s1),
            both_above = both_above, both_below = both_below, disagree = disagree)
end
