using ComparativeJudgement
using Test

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

end
