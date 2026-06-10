using ComparativeJudgement
using Test
using Random: MersenneTwister
using Statistics: mean

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

end
