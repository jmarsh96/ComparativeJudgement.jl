"""
    PairwiseData(wins, labels)

Pairwise comparison data: `wins[i, j]` counts how many times item `i` beat
item `j`, and `labels` names the items (any element type).
"""
struct PairwiseData{L}
    wins::Matrix{Int}
    labels::Vector{L}
    function PairwiseData(wins::Matrix{Int}, labels::Vector{L}) where {L}
        n = length(labels)
        size(wins) == (n, n) || throw(
            DimensionMismatch("wins must be $(n)×$(n) to match $(n) labels, got $(size(wins))")
        )
        new{L}(wins, labels)
    end
end

"""
    AnchoredData(data, anchor_labels, anchor_values)
    AnchoredData(data, anchor_groups, anchor_values)
    AnchoredData(data, anchors::AbstractDict)
    AnchoredData(data, group => value, ...)

Comparison data augmented with anchor measurements `y`, used to fit
[`Anchored`](@ref) models, which calibrate the latent scale via `y = a + b·λ + ε`.

Each measurement targets either a **single item** (passed by label, as
`anchor_labels::Vector`) or a **group of items** (passed as `anchor_groups`, a
vector of label-vectors). A group anchor is modelled as the group *mean*,

```
y_g = a + b · mean_{i∈G_g}(λ_i) + ε_g,   ε_g ~ N(0, σ²/n_g),   n_g = |G_g|,
```

so a larger group is measured more precisely (variance `σ²/n_g`). Single-item
anchors are the special case `n_g = 1`, identical to the original model. Single
and group anchors may be mixed in one dataset, and an item may appear in more than
one group. Internally every anchor is stored as a group of item indices in
`anchor_groups::Vector{Vector{Int}}`.
"""
struct AnchoredData{D, L}
    data::D
    anchor_groups::Vector{Vector{Int}}
    anchor_values::Vector{Float64}
    function AnchoredData{D, L}(data::D, anchor_groups::Vector{Vector{Int}},
                               anchor_values::Vector{Float64}) where {D, L}
        new{D, L}(data, anchor_groups, anchor_values)
    end
end

# The labels against which anchor labels are resolved. `PairwiseData` carries
# them directly.
_anchor_target_labels(data::PairwiseData) = data.labels

# Resolve a label to its item index, or throw.
function _resolve_anchor_label(labels, lbl)
    idx = findfirst(==(lbl), labels)
    idx === nothing && throw(ArgumentError("Anchor label $(lbl) not found in data labels"))
    return idx
end

# Single-item anchors: each measurement targets one item (groups of size one).
function AnchoredData(data, anchor_labels::Vector{L},
                      anchor_values::Vector{<:Real}) where {L}
    labels = _anchor_target_labels(data)
    r = length(anchor_labels)
    r >= 1 || throw(ArgumentError("Need at least 1 anchor, got none"))
    length(anchor_values) == r || throw(DimensionMismatch(
        "Got $r anchor labels but $(length(anchor_values)) anchor values"))
    allunique(anchor_labels) || throw(ArgumentError("Anchor labels must be unique"))
    groups = [[_resolve_anchor_label(labels, lbl)] for lbl in anchor_labels]
    return AnchoredData{typeof(data), L}(data, groups, Vector{Float64}(anchor_values))
end

# Group anchors: each measurement targets a group of items, modelled as the mean.
function AnchoredData(data, anchor_groups::Vector{<:AbstractVector{L}},
                      anchor_values::Vector{<:Real}) where {L}
    labels = _anchor_target_labels(data)
    G = length(anchor_groups)
    G >= 1 || throw(ArgumentError("Need at least 1 anchor group, got none"))
    length(anchor_values) == G || throw(DimensionMismatch(
        "Got $G anchor groups but $(length(anchor_values)) anchor values"))
    groups = Vector{Vector{Int}}(undef, G)
    for (g, grp) in enumerate(anchor_groups)
        isempty(grp) && throw(ArgumentError("Anchor group $g is empty"))
        allunique(grp) || throw(ArgumentError("Items within anchor group $g must be unique"))
        groups[g] = [_resolve_anchor_label(labels, lbl) for lbl in grp]
    end
    return AnchoredData{typeof(data), L}(data, groups, Vector{Float64}(anchor_values))
end

# Dict convenience: keys are labels (single-item) or label-vectors (groups).
function AnchoredData(data, anchors::AbstractDict{L, <:Real}) where {L}
    anchor_labels = collect(keys(anchors))
    anchor_values = [Float64(anchors[lbl]) for lbl in anchor_labels]
    return AnchoredData(data, anchor_labels, anchor_values)
end

# Pairs convenience for group anchors: `AnchoredData(data, ["a","b"] => 3.0, ["c"] => 4.0)`.
function AnchoredData(data, anchors::Pair{<:AbstractVector, <:Real}...)
    groups = [collect(first(p)) for p in anchors]
    values = Float64[last(p) for p in anchors]
    return AnchoredData(data, groups, values)
end

"""
    CovariateData(data, Z, names)
    CovariateData(data, Z)
    CovariateData(data, name => values, ...)

Comparison `data` augmented with an item covariate matrix `Z` (`K × p`, one row
per item in the order of `data.labels`) and covariate `names`. Used to fit
[`Covariates`](@ref) models, where `λ_i = z_iᵀβ`.

An overall intercept (a covariate constant across items) is **not** identifiable:
it cancels in the differences `z_i − z_j`, so such columns are rejected.
"""
struct CovariateData{L}
    data::PairwiseData{L}
    Z::Matrix{Float64}
    names::Vector{Symbol}
    function CovariateData(data::PairwiseData{L}, Z::AbstractMatrix,
                           names::Vector{Symbol}) where {L}
        K = length(data.labels)
        size(Z, 1) == K || throw(DimensionMismatch(
            "Z must have $K rows to match $K items, got $(size(Z, 1))"))
        size(Z, 2) == length(names) || throw(DimensionMismatch(
            "Z has $(size(Z, 2)) columns but $(length(names)) names given"))
        Zf = Matrix{Float64}(Z)
        for c in 1:size(Zf, 2)
            col = @view Zf[:, c]
            all(==(col[1]), col) && throw(ArgumentError(
                "Covariate $(names[c]) is constant across items; it cancels in " *
                "the comparison differences and is not identifiable. Drop it."))
        end
        new{L}(data, Zf, names)
    end
end
function CovariateData(data::PairwiseData, Z::AbstractMatrix)
    names = [Symbol("x", c) for c in 1:size(Z, 2)]
    return CovariateData(data, Z, names)
end
function CovariateData(data::PairwiseData, cols::Pair{Symbol, <:AbstractVector}...)
    names = Symbol[c.first for c in cols]
    Z = reduce(hcat, [Vector{Float64}(c.second) for c in cols])
    return CovariateData(data, Z, names)
end

"""
    RaterData(winners, losers, raters; item_labels=nothing, rater_labels=nothing)

Per-rater pairwise comparison data for a [`RaterHeterogeneity`](@ref) fit. Each
comparison `c` records the `winners[c]` item that beat the `losers[c]` item, as
judged by rater `raters[c]` (all given by label). Item and rater labels are
inferred in order of first appearance unless `item_labels` / `rater_labels` are
supplied to fix the ordering.
"""
struct RaterData{L, R}
    winner::Vector{Int}        # item index of the winner of each comparison
    loser::Vector{Int}         # item index of the loser of each comparison
    rater::Vector{Int}         # rater index of each comparison
    labels::Vector{L}          # K item labels
    raters::Vector{R}          # M rater labels
end

function RaterData(winners::AbstractVector, losers::AbstractVector,
                   raters::AbstractVector; item_labels=nothing, rater_labels=nothing)
    n = length(winners)
    (length(losers) == n && length(raters) == n) || throw(DimensionMismatch(
        "winners, losers and raters must have equal length, got " *
        "$(length(winners)), $(length(losers)), $(length(raters))"))
    n >= 1 || throw(ArgumentError("Need at least 1 comparison, got none"))
    ilabels = item_labels === nothing ? unique(vcat(collect(winners), collect(losers))) :
              collect(item_labels)
    rlabels = rater_labels === nothing ? unique(collect(raters)) : collect(rater_labels)
    length(ilabels) >= 2 || throw(ArgumentError(
        "Need at least 2 distinct items, got $(length(ilabels))"))
    iidx = Dict(l => i for (i, l) in enumerate(ilabels))
    ridx = Dict(r => i for (i, r) in enumerate(rlabels))
    w = Vector{Int}(undef, n); l = Vector{Int}(undef, n); r = Vector{Int}(undef, n)
    for c in 1:n
        haskey(iidx, winners[c]) || throw(ArgumentError("Unknown item label $(winners[c])"))
        haskey(iidx, losers[c])  || throw(ArgumentError("Unknown item label $(losers[c])"))
        haskey(ridx, raters[c])  || throw(ArgumentError("Unknown rater label $(raters[c])"))
        w[c] = iidx[winners[c]]; l[c] = iidx[losers[c]]; r[c] = ridx[raters[c]]
        w[c] == l[c] && throw(ArgumentError("Comparison $c pits item $(winners[c]) against itself"))
    end
    return RaterData{eltype(ilabels), eltype(rlabels)}(w, l, r, ilabels, rlabels)
end
