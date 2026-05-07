function _full_theta(θ_free::AbstractVector{T}) where {T}
    return vcat(zero(T), θ_free)
end

function _bt_neg_loglik(θ_free::AbstractVector, wins::Matrix{Int})
    θ = _full_theta(θ_free)
    λ = exp.(θ)
    n = length(λ)
    ll = zero(eltype(θ))
    for i in 1:n, j in 1:n
        i == j && continue
        w = wins[i, j]
        iszero(w) && continue
        ll += w * log(λ[i] / (λ[i] + λ[j]))
    end
    return -ll
end

function _bt_neg_grad!(G::AbstractVector, θ_free::AbstractVector, wins::Matrix{Int})
    θ = _full_theta(θ_free)
    λ = exp.(θ)
    n = length(λ)
    for k in 2:n
        expected = zero(eltype(θ))
        for j in 1:n
            j == k && continue
            n_kj = wins[k, j] + wins[j, k]
            iszero(n_kj) && continue
            expected += n_kj * λ[k] / (λ[k] + λ[j])
        end
        G[k - 1] = -(sum(wins[k, :]) - expected)
    end
    return G
end

function fit(model::BradleyTerry, method::MLE, data::PairwiseData{L}) where {L}
    wins = data.wins
    n = length(data.labels)
    n >= 2 || throw(ArgumentError("Need at least 2 items to fit BradleyTerry, got $n"))
    θ₀ = zeros(n - 1)
    f(θ_free) = _bt_neg_loglik(θ_free, wins)
    g!(G, θ_free) = _bt_neg_grad!(G, θ_free, wins)
    result = optimize(f, g!, θ₀, LBFGS())
    return FittedComparativeModel(
        model, 
        method, 
        result, 
        data.labels,
        Optim.converged(result), 
        Optim.iterations(result)
    )
end

function fit(model::BradleyTerry, method::MLE, wins::Matrix{Int}, labels::Vector{L}) where {L}
    return fit(model, method, PairwiseData(wins, labels))
end

function fit(model::BradleyTerry, data::PairwiseData)
    return fit(model, MLE(), data)
end

function fit(model::BradleyTerry, wins::Matrix{Int}, labels::Vector{L}) where {L}
    return fit(model, MLE(), PairwiseData(wins, labels))
end

function loglikelihood(fitted::FittedComparativeModel{BradleyTerry, MLE})
    return -Optim.minimum(fitted.result)
end

function probability(fitted::FittedComparativeModel{BradleyTerry, MLE}, i::Integer, j::Integer)
    θ = _full_theta(Optim.minimizer(fitted.result))
    λᵢ = exp(θ[i])
    λⱼ = exp(θ[j])
    return λᵢ / (λᵢ + λⱼ)
end

function probability(fitted::FittedComparativeModel{BradleyTerry, MLE, R, L},
                     item_i::L, item_j::L) where {R, L}
    idx_i = findfirst(==(item_i), fitted.labels)
    idx_j = findfirst(==(item_j), fitted.labels)
    idx_i === nothing && throw(ArgumentError("Label $(item_i) not found in fitted model"))
    idx_j === nothing && throw(ArgumentError("Label $(item_j) not found in fitted model"))
    return probability(fitted, idx_i, idx_j)
end
