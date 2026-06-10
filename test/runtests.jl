using ComparativeJudgement
using Test
using Random: MersenneTwister, rand
using Statistics: mean, var

@testset "ComparativeJudgement.jl" begin

    @testset "PairwiseData" begin
        wins = [0 3; 1 0]
        data = PairwiseData(wins, ["A", "B"])
        @test data.wins == wins
        @test data.labels == ["A", "B"]
        @test_throws DimensionMismatch PairwiseData([0 1 0; 1 0 0], ["A", "B"])
    end

    @testset "BradleyTerry MLE — 2 items" begin
        wins = [0 3; 1 0]
        data = PairwiseData(wins, ["A", "B"])
        fitted = fit(BradleyTerry(), MLE(), data)

        @test fitted.converged
        @test fitted.iterations >= 1
        @test fitted.labels == ["A", "B"]
        @test fitted.model isa BradleyTerry
        @test fitted.method isa MLE

        p_AB = probability(fitted, "A", "B")
        @test p_AB > 0.5
        @test probability(fitted, "B", "A") ≈ 1 - p_AB

        ll = loglikelihood(fitted)
        @test isfinite(ll)
        @test ll <= 0
    end

    @testset "BradleyTerry MLE — convenience overloads" begin
        wins = [0 3; 1 0]
        # matrix + labels overload
        fitted1 = fit(BradleyTerry(), MLE(), wins, ["A", "B"])
        @test fitted1.converged
        # default MLE overload
        fitted2 = fit(BradleyTerry(), PairwiseData(wins, ["A", "B"]))
        @test fitted2.converged
        @test probability(fitted1, "A", "B") ≈ probability(fitted2, "A", "B")
    end

    @testset "BradleyTerry MLE — equal wins" begin
        wins = [0 5; 5 0]
        fitted = fit(BradleyTerry(), wins, [:x, :y])
        @test probability(fitted, :x, :y) ≈ 0.5 atol=1e-6
    end

    @testset "BradleyTerry MLE — 3 items ordinal consistency" begin
        # item 3 dominates item 2 dominates item 1
        wins = [0 5 2; 15 0 5; 18 15 0]
        fitted = fit(BradleyTerry(), wins, [1, 2, 3])
        @test probability(fitted, 1, 2) < 0.5
        @test probability(fitted, 1, 3) < 0.5
        @test probability(fitted, 2, 3) < 0.5
        @test probability(fitted, 1, 3) < probability(fitted, 1, 2)
    end

    @testset "BradleyTerry MLE — integer index probability" begin
        wins = [0 3; 1 0]
        fitted = fit(BradleyTerry(), wins, ["A", "B"])
        @test probability(fitted, 1, 2) ≈ probability(fitted, "A", "B")
    end

    @testset "BradleyTerry MLE — bad label" begin
        fitted = fit(BradleyTerry(), [0 3; 1 0], ["A", "B"])
        @test_throws ArgumentError probability(fitted, "A", "C")
    end

    @testset "NormalPrior" begin
        p = NormalPrior(3)
        @test length(p.μ) == 3
        @test size(p.Σ) == (3, 3)
        @test p.μ == zeros(3)

        p2 = NormalPrior([1.0, 0.0], [2.0 0.0; 0.0 2.0])
        @test p2.μ == [1.0, 0.0]

        @test_throws DimensionMismatch NormalPrior([1.0, 0.0], [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
    end

    @testset "Bayesian constructor" begin
        m = Bayesian()
        @test m.n_samples == 2000
        @test m.n_burnin  == 500
        @test m.center    == true

        m2 = Bayesian(n_samples=100, n_burnin=50, center=false)
        @test m2.n_samples == 100

        @test_throws ArgumentError Bayesian(n_samples=0)
        @test_throws ArgumentError Bayesian(n_burnin=-1)
    end

    @testset "PG sampler — PG(1,0) mean ≈ 0.25" begin
        rng = MersenneTwister(42)
        n = 2000
        samples = [ComparativeJudgement._sample_pg(rng, 1, 0.0) for _ in 1:n]
        @test abs(mean(samples) - 0.25) < 0.02
        @test all(isfinite, samples)
        @test all(>(0), samples)
    end

    @testset "BradleyTerry Bayesian — 2 items" begin
        rng = MersenneTwister(1)
        wins = [0 3; 1 0]
        data = PairwiseData(wins, ["A", "B"])
        method = Bayesian(n_samples=500, n_burnin=200)
        fitted = fit(BradleyTerry(), method, data; rng=rng)

        @test fitted.converged
        @test fitted.iterations == 700
        @test fitted.labels == ["A", "B"]

        p_AB = probability(fitted, "A", "B")
        @test p_AB > 0.5
        @test probability(fitted, "B", "A") ≈ 1 - p_AB atol=1e-10

        pm = posterior_mean(fitted)
        @test length(pm) == 2
        @test all(isfinite, pm)

        ps = posterior_std(fitted)
        @test length(ps) == 2
        @test all(>(0), ps)

        lb, ub = credible_interval(fitted, 1)
        @test lb < ub

        ll = loglikelihood(fitted)
        @test ll isa Vector{Float64}
        @test length(ll) == 500
        @test all(isfinite, ll)
    end

    @testset "BradleyTerry Bayesian — default prior" begin
        rng = MersenneTwister(2)
        wins = [0 3; 1 0]
        fitted = fit(BradleyTerry(), Bayesian(n_samples=200, n_burnin=100),
                     PairwiseData(wins, ["A", "B"]); rng=rng)
        @test probability(fitted, "A", "B") > 0.5
    end

    @testset "BradleyTerry Bayesian — 3 items ordinal consistency" begin
        rng = MersenneTwister(3)
        wins = [0 5 2; 15 0 5; 18 15 0]
        fitted = fit(BradleyTerry(), Bayesian(n_samples=1000, n_burnin=300),
                     wins, [1, 2, 3], NormalPrior(3); rng=rng)
        @test probability(fitted, 1, 2) < 0.5
        @test probability(fitted, 1, 3) < 0.5
        @test probability(fitted, 2, 3) < 0.5
        @test probability(fitted, 1, 3) < probability(fitted, 1, 2)
    end

    @testset "BradleyTerry Bayesian — bad label" begin
        rng = MersenneTwister(4)
        fitted = fit(BradleyTerry(), Bayesian(n_samples=100, n_burnin=50),
                     PairwiseData([0 3; 1 0], ["A", "B"]); rng=rng)
        @test_throws ArgumentError probability(fitted, "A", "C")
    end

    @testset "Bayesian thin" begin
        @test Bayesian().thin == 1
        @test Bayesian(thin=5).thin == 5
        @test_throws ArgumentError Bayesian(thin=0)
    end

    @testset "InverseGammaPrior" begin
        p = InverseGammaPrior(2.0, 1.0)
        @test p.α == 2.0
        @test p.β == 1.0
        @test_throws ArgumentError InverseGammaPrior(0.0, 1.0)
        @test_throws ArgumentError InverseGammaPrior(2.0, -1.0)
    end

    @testset "AnchoredPrior" begin
        p = AnchoredPrior()
        @test p.τ² == 0.01
        @test length(p.β_prior.μ) == 2
        @test p.σ²_prior isa InverseGammaPrior

        p2 = AnchoredPrior(τ²=0.5)
        @test p2.τ² == 0.5

        @test_throws ArgumentError AnchoredPrior(τ²=0.0)
        @test_throws DimensionMismatch AnchoredPrior(β_prior=NormalPrior(3))
    end

    @testset "AnchoredData" begin
        wins = [0 3 1; 1 0 2; 2 1 0]
        pdata = PairwiseData(wins, ["A", "B", "C"])

        data = AnchoredData(pdata, ["A", "C"], [1.0, 5.0])
        @test data.anchor_idx == [1, 3]
        @test data.anchor_values == [1.0, 5.0]

        ddata = AnchoredData(pdata, Dict("B" => 3))
        @test ddata.anchor_idx == [2]
        @test ddata.anchor_values == [3.0]

        @test_throws ArgumentError AnchoredData(pdata, ["A", "D"], [1.0, 2.0])
        @test_throws DimensionMismatch AnchoredData(pdata, ["A"], [1.0, 2.0])
        @test_throws ArgumentError AnchoredData(pdata, ["A", "A"], [1.0, 2.0])
        @test_throws ArgumentError AnchoredData(pdata, String[], Float64[])
    end

    @testset "Gamma sampler — moments" begin
        rng = MersenneTwister(11)
        for α in (0.5, 1.0, 4.5)
            samples = [ComparativeJudgement._sample_gamma(rng, α) for _ in 1:20_000]
            @test all(isfinite, samples)
            @test all(>(0), samples)
            @test abs(mean(samples) - α) < 0.1 * max(α, 1.0)
            @test abs(var(samples) - α) < 0.2 * max(α, 1.0)
        end
        ig = [ComparativeJudgement._sample_inv_gamma(rng, 3.0, 2.0) for _ in 1:20_000]
        @test abs(mean(ig) - 1.0) < 0.05   # mean = β/(α−1)
    end

    @testset "BradleyTerryAnchored — construction" begin
        m = BradleyTerryAnchored()
        @test m isa Anchored{BradleyTerry}
        @test m.model isa BradleyTerry
        @test Anchored(BradleyTerry()) isa BradleyTerryAnchored
    end

    @testset "BradleyTerryAnchored — recovery" begin
        rng = MersenneTwister(7)
        n = 8
        λ_true = collect(range(-1.5, 1.5, length=n))
        a_true, b_true, σ_true = 3.0, 2.0, 0.1
        wins = zeros(Int, n, n)
        for i in 1:n, j in 1:n
            i == j && continue
            p = 1 / (1 + exp(-(λ_true[i] - λ_true[j])))
            for _ in 1:15
                rand(rng) < p && (wins[i, j] += 1)
            end
        end
        anchor_labels = [1, 3, 6, 8]
        y = [a_true + b_true * λ_true[i] + σ_true * randn(rng) for i in anchor_labels]
        data = AnchoredData(PairwiseData(wins, collect(1:n)), anchor_labels, y)

        method = Bayesian(n_samples=600, n_burnin=300, thin=2)
        fitted = fit(BradleyTerryAnchored(), method, data; rng=rng)

        @test fitted.converged
        @test fitted.iterations == 300 + 2 * 600
        @test fitted.result isa AnchoredMCMCSamples
        @test size(fitted.result.λ_samples) == (600, n)
        @test size(fitted.result.β_samples) == (600, 2)
        @test length(fitted.result.σ²_samples) == 600

        # λ ordering matches the truth
        pm = posterior_mean(fitted)
        @test sortperm(pm) == sortperm(λ_true)
        @test all(>(0), posterior_std(fitted))

        # calibration recovers (a, b) within loose tolerance
        cal = calibration(fitted)
        @test abs(cal.a - a_true) < 0.5
        @test abs(cal.b - b_true) < 0.6
        @test cal.σ² > 0

        # predictions: anchored item near its measurement, all items ordered
        preds = predict(fitted)
        @test length(preds) == n
        @test abs(preds[8] - y[end]) < 0.5
        @test issorted(preds)

        # posterior-predictive draws and interval for an unanchored item
        draws = predict(fitted, 5; rng=rng)
        @test length(draws) == 600
        @test all(isfinite, draws)
        lo, hi = predict(fitted, 5; prob=0.95, rng=rng)
        @test lo < preds[5] < hi
        lo2, hi2 = predict(fitted, 2; prob=0.5, rng=rng)
        @test lo2 < hi2

        # label-based access
        @test predict(fitted, 5) isa Vector{Float64}

        # probabilities behave like BT
        p81 = probability(fitted, 8, 1)
        @test p81 > 0.5
        @test probability(fitted, 1, 8) ≈ 1 - p81 atol=1e-10

        lb, ub = credible_interval(fitted, 1)
        @test lb < ub

        ll = loglikelihood(fitted)
        @test length(ll) == 600
        @test all(isfinite, ll)
    end

    @testset "BradleyTerryAnchored — convenience and labels" begin
        rng = MersenneTwister(9)
        wins = [0 8 2; 4 0 7; 9 3 0]
        pdata = PairwiseData(wins, ["A", "B", "C"])
        data = AnchoredData(pdata, ["A", "C"], [1.0, 2.0])

        # default-method overload
        fitted = fit(BradleyTerryAnchored(),
                     Bayesian(n_samples=200, n_burnin=100), data; rng=rng)
        @test fitted.labels == ["A", "B", "C"]

        # label-based predict and probability
        @test predict(fitted, "B"; rng=rng) isa Vector{Float64}
        @test probability(fitted, "A", "C") ≈ probability(fitted, 1, 3)
        @test_throws ArgumentError predict(fitted, "D")
        @test_throws ArgumentError probability(fitted, "A", "D")

        # explicit prior
        prior = AnchoredPrior(τ²=0.1, β_prior=NormalPrior(2, scale=5.0),
                              σ²_prior=InverseGammaPrior(3.0, 2.0))
        fitted2 = fit(BradleyTerryAnchored(), Bayesian(n_samples=100, n_burnin=50),
                      data, prior; rng=rng)
        @test fitted2.converged
    end

end
