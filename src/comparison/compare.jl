# ─── Multi-model comparison table ────────────────────────────────────────────
#
# Convenience orchestration over the per-model information criteria: score a slate
# of competing fits on the same data by a single criterion and tabulate them by
# the difference from the best model. AIC/BIC for maximum-likelihood fits;
# WAIC/LOO for Bayesian fits.

"""
    ModelComparisonTable

Result of [`compare`](@ref): the model `names`, the `criterion` used, the per-model
`values` (lower is better), and `Δ`, the difference of each from the best model.
Rows are ordered best-first.
"""
struct ModelComparisonTable
    names::Vector{String}
    criterion::Symbol
    values::Vector{Float64}
    Δ::Vector{Float64}
end

function Base.show(io::IO, t::ModelComparisonTable)
    println(io, "ModelComparisonTable (", t.criterion, ", lower is better)")
    w = maximum(length, t.names)
    println(io, "  ", rpad("model", w), "   ", lpad(string(t.criterion), 10), "   ", lpad("Δ", 8))
    for i in eachindex(t.names)
        println(io, "  ", rpad(t.names[i], w), "   ",
                lpad(string(round(t.values[i], digits=2)), 10), "   ",
                lpad(string(round(t.Δ[i], digits=2)), 8))
    end
end

_default_name(f::FittedComparativeModel) = string(nameof(typeof(f.model)))

_ic_value(f, data, ::Val{:aic}) = aic(f, data)
_ic_value(f, data, ::Val{:bic}) = bic(f, data)
_ic_value(f, data, ::Val{:waic}) = waic(f, data).waic
_ic_value(f, data, ::Val{:loo}) = loo(f, data).looic

"""
    compare(fits...; data, criterion=:loo, names=nothing)

Tabulate competing `fits` (all fit to the same `data`) by an information
criterion: `:aic` or `:bic` for [`MLE`](@ref) fits, `:waic` or `:loo` for
[`Bayesian`](@ref) fits. Returns a [`ModelComparisonTable`](@ref) ordered
best-first, with `Δ` the gap from the best model. `names` optionally labels the
rows (defaults to the model type names).
"""
function compare(fits::FittedComparativeModel...; data, criterion::Symbol=:loo,
                 names::Union{Nothing, Vector{String}}=nothing)
    criterion in (:aic, :bic, :waic, :loo) || throw(ArgumentError(
        "criterion must be :aic, :bic, :waic or :loo, got $criterion"))
    isempty(fits) && throw(ArgumentError("need at least one fit to compare"))
    nm = names === nothing ? [_default_name(f) for f in fits] : names
    length(nm) == length(fits) || throw(DimensionMismatch(
        "got $(length(fits)) fits but $(length(nm)) names"))
    vals = [_ic_value(f, data, Val(criterion)) for f in fits]
    order = sortperm(vals)
    v = vals[order]
    return ModelComparisonTable(nm[order], criterion, v, v .- minimum(v))
end
