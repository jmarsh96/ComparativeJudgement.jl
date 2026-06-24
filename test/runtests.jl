using ComparativeJudgement
using Test
using Random: MersenneTwister, rand
using Statistics: mean, var, cor
using LinearAlgebra: diag, dot

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

    @testset "strengths" begin
        # item 3 dominates item 2 dominates item 1
        wins = [0 5 2; 15 0 5; 18 15 0]
        data = PairwiseData(wins, ["a", "b", "c"])

        mle = fit(BradleyTerry(), MLE(), data)
        λ̂ = strengths(mle)
        @test length(λ̂) == 3
        @test abs(sum(λ̂)) < 1e-10           # centred
        @test issorted(λ̂)                    # matches the dominance ordering

        rng = MersenneTwister(11)
        bayes = fit(BradleyTerry(), Bayesian(n_samples=200, n_burnin=100), data; rng=rng)
        @test strengths(bayes) == posterior_mean(bayes)

        adata = AnchoredData(data, ["a", "c"], [1.0, 3.0])
        anchored = fit(BradleyTerryAnchored(), Bayesian(n_samples=200, n_burnin=100),
                       adata; rng=rng)
        @test strengths(anchored) == posterior_mean(anchored)
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
        @test m2.center    == false

        @test_throws ArgumentError Bayesian(n_samples=0)
        @test_throws ArgumentError Bayesian(n_burnin=-1)
    end

    @testset "Centering — anchored model" begin
        # Centering stays on by default for anchored models: the anchor
        # likelihood only constrains a + b·λ, so λ's location trades off
        # against the intercept and is otherwise pinned only by weak priors.
        wins = [0 8 2; 4 0 7; 9 3 0]
        data = AnchoredData(PairwiseData(wins, ["A", "B", "C"]), ["A", "C"], [1.0, 2.0])
        f_default = fit(BradleyTerryAnchored(), Bayesian(n_samples=100, n_burnin=50),
                        data; rng=MersenneTwister(5))
        f_off     = fit(BradleyTerryAnchored(),
                        Bayesian(n_samples=100, n_burnin=50, center=false),
                        data; rng=MersenneTwister(5))
        row_means(f) = vec(mean(f.result.λ_samples, dims=2))
        @test all(abs.(row_means(f_default)) .< 1e-10)
        @test any(abs.(row_means(f_off)) .> 1e-6)
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
        @test data.anchor_groups == [[1], [3]]      # single-item anchors → size-1 groups
        @test data.anchor_values == [1.0, 5.0]

        ddata = AnchoredData(pdata, Dict("B" => 3))
        @test ddata.anchor_groups == [[2]]
        @test ddata.anchor_values == [3.0]

        @test_throws ArgumentError AnchoredData(pdata, ["A", "D"], [1.0, 2.0])
        @test_throws DimensionMismatch AnchoredData(pdata, ["A"], [1.0, 2.0])
        @test_throws ArgumentError AnchoredData(pdata, ["A", "A"], [1.0, 2.0])
        @test_throws ArgumentError AnchoredData(pdata, String[], Float64[])
    end

    @testset "AnchoredData — group anchors" begin
        wins = [0 3 1 2; 1 0 2 1; 2 1 0 3; 1 2 1 0]
        pdata = PairwiseData(wins, ["A", "B", "C", "D"])

        # Group anchors: each measurement averages a group of items.
        gdata = AnchoredData(pdata, [["A", "B"], ["C", "D"]], [1.0, 5.0])
        @test gdata isa AnchoredData{PairwiseData{String}, String}
        @test gdata.anchor_groups == [[1, 2], [3, 4]]
        @test gdata.anchor_values == [1.0, 5.0]

        # Mixed single + group anchors, and an item shared across groups.
        mdata = AnchoredData(pdata, [["A"], ["B", "C", "D"], ["A", "D"]], [1.0, 2.0, 3.0])
        @test mdata.anchor_groups == [[1], [2, 3, 4], [1, 4]]

        # Pairs convenience constructor.
        pdata2 = AnchoredData(pdata, ["A", "B"] => 1.0, ["C"] => 4.0)
        @test pdata2.anchor_groups == [[1, 2], [3]]
        @test pdata2.anchor_values == [1.0, 4.0]

        # Single-label and group constructors dispatch as expected.
        @test AnchoredData(pdata, ["A", "C"], [1.0, 2.0]).anchor_groups == [[1], [3]]

        # Validation.
        @test_throws ArgumentError AnchoredData(pdata, [["A", "Z"]], [1.0])      # unknown label
        @test_throws ArgumentError AnchoredData(pdata, [["A", "A"]], [1.0])      # dup within group
        @test_throws ArgumentError AnchoredData(pdata, [String[]], [1.0])        # empty group
        @test_throws DimensionMismatch AnchoredData(pdata, [["A"], ["B"]], [1.0])
        @test_throws ArgumentError AnchoredData(pdata, Vector{String}[], Float64[])
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

    # Simulate covariate Bradley-Terry data: λ_i = z_iᵀβ.
    function _simulate_covariate_data(rng, K, βtrue; n_per_pair=6)
        p = length(βtrue)
        Z = randn(rng, K, p)
        λ = Z * βtrue
        wins = zeros(Int, K, K)
        for i in 1:K, j in (i + 1):K
            pij = 1.0 / (1.0 + exp(-(λ[i] - λ[j])))
            for _ in 1:n_per_pair
                rand(rng) < pij ? (wins[i, j] += 1) : (wins[j, i] += 1)
            end
        end
        labels = ["item$i" for i in 1:K]
        return CovariateData(PairwiseData(wins, labels), Z), λ
    end

    @testset "CovariateData construction" begin
        wins = [0 3 1; 2 0 4; 5 1 0]
        data = PairwiseData(wins, ["A", "B", "C"])
        Z = [1.0 0.0; 0.5 1.0; -1.0 2.0]

        cd = CovariateData(data, Z, [:x, :y])
        @test size(cd.Z) == (3, 2)
        @test cd.names == [:x, :y]
        @test CovariateData(data, Z).names == [:x1, :x2]          # auto names
        @test CovariateData(data, :x => Z[:, 1], :y => Z[:, 2]).Z == Z  # pairs ctor

        @test_throws DimensionMismatch CovariateData(data, Z, [:only_one])
        @test_throws DimensionMismatch CovariateData(data, randn(2, 2), [:a, :b])
        # constant column is not identifiable
        @test_throws ArgumentError CovariateData(data, [1.0 1.0; 1.0 0.0; 1.0 2.0], [:const, :ok])
    end

    @testset "Covariate BradleyTerry MLE recovery" begin
        rng = MersenneTwister(2024)
        βtrue = [1.5, -1.0, 0.0]
        cd, λtrue = _simulate_covariate_data(rng, 40, βtrue; n_per_pair=10)

        f = fit(BradleyTerryCovariates(), MLE(), cd)
        @test f.converged
        β̂ = collect(values(coefficients(f)))
        @test isapprox(β̂, βtrue; atol=0.4)

        # strengths are λ = Zβ, centred to sum to zero
        λ̂ = strengths(f)
        @test sum(λ̂) ≈ 0.0 atol=1e-8
        @test cor(λ̂, λtrue) > 0.95

        # default-method overload + label/index probability agree
        @test fit(BradleyTerryCovariates(), cd).converged
        @test probability(f, "item1", "item2") ≈ probability(f, 1, 2)
        @test 0.0 < probability(f, 1, 2) < 1.0
        @test_throws ArgumentError probability(f, "item1", "nope")

        # coefficient covariance is a sensible p×p PSD matrix
        @test size(f.result.vcov) == (3, 3)
        @test all(diag(f.result.vcov) .> 0)
    end

    @testset "Covariate one-hot reproduces plain BradleyTerry" begin
        rng = MersenneTwister(7)
        wins = [0 9 6; 3 0 8; 4 5 0]
        data = PairwiseData(wins, ["A", "B", "C"])
        # One-hot covariates for items 2,3 (item 1 is the reference / dropped level)
        Z = [0.0 0.0; 1.0 0.0; 0.0 1.0]
        cd = CovariateData(data, Z, [:I2, :I3])

        fcov = fit(BradleyTerryCovariates(), MLE(), cd)
        fbt = fit(BradleyTerry(), MLE(), data)
        @test isapprox(strengths(fcov), strengths(fbt); atol=1e-5)
    end

    @testset "Covariate BradleyTerry Bayesian (Normal)" begin
        rng = MersenneTwister(99)
        βtrue = [1.2, -0.8]
        cd, _ = _simulate_covariate_data(rng, 35, βtrue; n_per_pair=10)

        f = fit(BradleyTerryCovariates(), Bayesian(n_samples=800, n_burnin=300),
                cd; rng=MersenneTwister(1))
        β̂ = collect(values(coefficients(f)))
        @test isapprox(β̂, βtrue; atol=0.4)
        @test length(strengths(f)) == 35
        @test length(posterior_std(f)) == 35
        lo, hi = credible_interval(f, 1)
        @test lo < hi
        @test loglikelihood(f) isa Vector{Float64}
        @test 0.0 < probability(f, 1, 2) < 1.0
    end

    @testset "Coefficient uncertainty" begin
        # normal quantile helper agrees with known z-values
        @test ComparativeJudgement._norm_quantile(0.975) ≈ 1.959963985 atol=1e-6
        @test ComparativeJudgement._norm_quantile(0.5) ≈ 0.0 atol=1e-9
        @test ComparativeJudgement._norm_quantile(0.95) ≈ 1.644853627 atol=1e-6

        rng = MersenneTwister(2718)
        βtrue = [1.6, -1.2, 0.0]
        cd, _ = _simulate_covariate_data(rng, 45, βtrue; n_per_pair=12)

        # MLE: Wald confidence intervals from the Fisher information
        f = fit(BradleyTerryCovariates(), MLE(), cd)
        se = coefficient_std(f)
        ci = coefficient_intervals(f; level=0.95)
        @test keys(se) == (:x1, :x2, :x3)
        @test all(values(se) .> 0)
        β̂ = coefficients(f)
        for k in keys(ci)
            lo, hi = ci[k]
            @test lo < β̂[k] < hi                       # estimate inside its interval
        end
        @test ci.x1[1] > 0 && ci.x2[2] < 0             # signal CIs exclude zero
        @test ci.x3[1] < 0 < ci.x3[2]                  # null CI covers zero
        # wider level ⇒ wider interval
        ci99 = coefficient_intervals(f; level=0.99)
        @test (ci99.x1[2] - ci99.x1[1]) > (ci.x1[2] - ci.x1[1])
        @test_throws ArgumentError coefficient_intervals(f; level=1.5)

        # Bayesian: posterior credible intervals
        fb = fit(BradleyTerryCovariates(), Bayesian(n_samples=800, n_burnin=300),
                 cd; rng=MersenneTwister(11))
        cib = coefficient_intervals(fb; level=0.9)
        sdb = coefficient_std(fb)
        @test all(values(sdb) .> 0)
        for k in keys(cib)
            lo, hi = cib[k]
            @test lo < coefficients(fb)[k] < hi
        end
        @test cib.x1[1] > 0 && cib.x2[2] < 0           # signal credible intervals exclude zero

        # selection carries through: intervals only for retained covariates
        fs = fit(BradleyTerryCovariates(), StepwiseMLE(criterion=:BIC), cd)
        @test keys(coefficient_intervals(fs)) == keys(coefficients(fs))
    end

    @testset "Horseshoe shrinks null coefficients" begin
        rng = MersenneTwister(123)
        βtrue = [2.0, 0.0, 0.0, 0.0]    # one signal, three null
        cd, _ = _simulate_covariate_data(rng, 45, βtrue; n_per_pair=10)

        f = fit(BradleyTerryCovariates(), Bayesian(n_samples=800, n_burnin=400),
                cd, HorseshoePrior(); rng=MersenneTwister(5))
        β̂ = collect(values(coefficients(f)))
        @test abs(β̂[1]) > 1.0                       # signal survives
        @test all(abs.(β̂[2:4]) .< 0.4)              # nulls shrunk toward zero
        @test abs(β̂[1]) > maximum(abs.(β̂[2:4]))
    end

    @testset "Spike-and-slab inclusion probabilities" begin
        rng = MersenneTwister(321)
        βtrue = [2.0, -2.0, 0.0, 0.0]
        cd, _ = _simulate_covariate_data(rng, 45, βtrue; n_per_pair=10)

        f = fit(BradleyTerryCovariates(), Bayesian(n_samples=800, n_burnin=400),
                cd, SpikeSlabPrior(); rng=MersenneTwister(6))
        pip = collect(values(inclusion_probabilities(f)))
        @test pip[1] > 0.8 && pip[2] > 0.8          # true covariates included
        @test pip[3] < 0.5 && pip[4] < 0.5          # null covariates mostly excluded

        # inclusion probabilities require a spike-slab fit
        fn = fit(BradleyTerryCovariates(), Bayesian(n_samples=100, n_burnin=50),
                 cd; rng=MersenneTwister(6))
        @test_throws ArgumentError inclusion_probabilities(fn)
    end

    @testset "Stepwise MLE selection" begin
        rng = MersenneTwister(555)
        βtrue = [1.8, -1.5, 0.0, 0.0, 0.0]
        cd, _ = _simulate_covariate_data(rng, 50, βtrue; n_per_pair=12)

        f = fit(BradleyTerryCovariates(), StepwiseMLE(direction=:both, criterion=:BIC), cd)
        @test sort(f.result.selected) == [1, 2]     # picks the two real covariates
        @test length(coefficients(f)) == 2
        @test !isempty(f.result.trace)

        # forward and backward also reach the right subset here
        ff = fit(BradleyTerryCovariates(), StepwiseMLE(direction=:forward, criterion=:BIC), cd)
        fb = fit(BradleyTerryCovariates(), StepwiseMLE(direction=:backward, criterion=:BIC), cd)
        @test sort(ff.result.selected) == [1, 2]
        @test sort(fb.result.selected) == [1, 2]

        @test_throws ArgumentError StepwiseMLE(direction=:sideways)
        @test_throws ArgumentError StepwiseMLE(criterion=:WAIC)
    end

    # ─────────────────────────── Thurstone Case V ────────────────────────────

    # Probit comparison data with known latent strengths λ: P(i beats j) = Φ(λᵢ−λⱼ).
    _Φ(x) = ComparativeJudgement._normcdf(x)
    function _simulate_thurstone_data(rng, λ; n_per_pair=20)
        K = length(λ)
        wins = zeros(Int, K, K)
        for i in 1:K, j in (i + 1):K
            p = _Φ(λ[i] - λ[j])
            for _ in 1:n_per_pair
                rand(rng) < p ? (wins[i, j] += 1) : (wins[j, i] += 1)
            end
        end
        return wins
    end
    function _simulate_thurstone_covariate_data(rng, K, βtrue; n_per_pair=14)
        Z = randn(rng, K, length(βtrue))
        λ = Z * βtrue
        wins = _simulate_thurstone_data(rng, λ; n_per_pair=n_per_pair)
        labels = ["item$i" for i in 1:K]
        return CovariateData(PairwiseData(wins, labels), Z), λ
    end

    @testset "ThurstoneCaseV — construction" begin
        @test ThurstoneCaseV() isa ThurstoneCaseV
        @test ThurstoneCaseV().distribution === :normal
        @test ThurstoneCaseV(:normal).distribution === :normal
        @test ThurstoneCaseVAnchored() isa Anchored{ThurstoneCaseV}
        @test ThurstoneCaseVCovariates() isa Covariates{ThurstoneCaseV}
        # unimplemented distribution is rejected at fit time
        @test_throws ArgumentError fit(ThurstoneCaseV(:logistic), MLE(),
                                       PairwiseData([0 3; 1 0], ["A", "B"]))
    end

    @testset "ThurstoneCaseV MLE — 2 items" begin
        wins = [0 3; 1 0]
        data = PairwiseData(wins, ["A", "B"])
        fitted = fit(ThurstoneCaseV(), MLE(), data)
        @test fitted.converged
        @test fitted.model isa ThurstoneCaseV
        p_AB = probability(fitted, "A", "B")
        @test p_AB > 0.5
        @test probability(fitted, "B", "A") ≈ 1 - p_AB atol=1e-9
        @test probability(fitted, 1, 2) ≈ p_AB
        ll = loglikelihood(fitted)
        @test isfinite(ll) && ll <= 0
    end

    @testset "ThurstoneCaseV MLE — overloads, equal wins, ordinal, bad label" begin
        wins = [0 3; 1 0]
        f1 = fit(ThurstoneCaseV(), MLE(), wins, ["A", "B"])
        f2 = fit(ThurstoneCaseV(), PairwiseData(wins, ["A", "B"]))
        f3 = fit(ThurstoneCaseV(), wins, ["A", "B"])
        @test probability(f1, "A", "B") ≈ probability(f2, "A", "B") ≈ probability(f3, "A", "B")

        feq = fit(ThurstoneCaseV(), [0 5; 5 0], [:x, :y])
        @test probability(feq, :x, :y) ≈ 0.5 atol=1e-6

        ford = fit(ThurstoneCaseV(), [0 5 2; 15 0 5; 18 15 0], [1, 2, 3])
        @test probability(ford, 1, 2) < 0.5
        @test probability(ford, 1, 3) < probability(ford, 1, 2)

        @test_throws ArgumentError probability(f1, "A", "C")
    end

    @testset "ThurstoneCaseV MLE — strengths centred & ordered" begin
        data = PairwiseData([0 5 2; 15 0 5; 18 15 0], ["a", "b", "c"])
        λ̂ = strengths(fit(ThurstoneCaseV(), MLE(), data))
        @test length(λ̂) == 3
        @test abs(sum(λ̂)) < 1e-10
        @test issorted(λ̂)
    end

    @testset "ThurstoneCaseV MLE — recovers known strengths" begin
        rng = MersenneTwister(202)
        λ_true = collect(range(-2.0, 2.0, length=10)); λ_true .-= sum(λ_true) / 10
        wins = _simulate_thurstone_data(rng, λ_true; n_per_pair=40)
        f = fit(ThurstoneCaseV(), MLE(), PairwiseData(wins, collect(1:10)))
        @test cor(strengths(f), λ_true) > 0.95
        @test sortperm(strengths(f)) == sortperm(λ_true)
    end

    @testset "ThurstoneCaseV Bayesian — 2 items" begin
        rng = MersenneTwister(1)
        data = PairwiseData([0 3; 1 0], ["A", "B"])
        fitted = fit(ThurstoneCaseV(), Bayesian(n_samples=500, n_burnin=200), data; rng=rng)
        @test fitted.iterations == 700
        @test fitted.result isa BTMCMCSamples
        p_AB = probability(fitted, "A", "B")
        @test p_AB > 0.5
        @test probability(fitted, "B", "A") ≈ 1 - p_AB atol=1e-10
        @test length(posterior_mean(fitted)) == 2
        @test all(>(0), posterior_std(fitted))
        lb, ub = credible_interval(fitted, 1)
        @test lb < ub
        ll = loglikelihood(fitted)
        @test ll isa Vector{Float64} && length(ll) == 500 && all(isfinite, ll)
        @test strengths(fitted) == posterior_mean(fitted)
    end

    @testset "ThurstoneCaseV Bayesian — default prior, ordinal, recovery, bad label" begin
        rng = MersenneTwister(3)
        f0 = fit(ThurstoneCaseV(), Bayesian(n_samples=200, n_burnin=100),
                 PairwiseData([0 3; 1 0], ["A", "B"]); rng=rng)
        @test probability(f0, "A", "B") > 0.5

        wins = _simulate_thurstone_data(MersenneTwister(8),
                                        collect(range(-2, 2, length=8)); n_per_pair=30)
        fb = fit(ThurstoneCaseV(), Bayesian(n_samples=600, n_burnin=300),
                 wins, collect(1:8), NormalPrior(8); rng=MersenneTwister(9))
        @test sortperm(posterior_mean(fb)) == collect(1:8)

        @test_throws ArgumentError probability(f0, "A", "C")
    end

    @testset "ThurstoneCaseV MLE ≈ Bayesian" begin
        rng = MersenneTwister(77)
        wins = _simulate_thurstone_data(rng, collect(range(-1.5, 1.5, length=6)); n_per_pair=40)
        data = PairwiseData(wins, collect(1:6))
        fm = fit(ThurstoneCaseV(), MLE(), data)
        fb = fit(ThurstoneCaseV(), Bayesian(n_samples=1500, n_burnin=500), data;
                 rng=MersenneTwister(1))
        @test isapprox(strengths(fm), posterior_mean(fb); atol=0.25)
    end

    @testset "ThurstoneCaseVAnchored — MLE recovery" begin
        rng = MersenneTwister(31)
        n = 10
        λ_true = collect(range(-1.5, 1.5, length=n)); λ_true .-= sum(λ_true) / n
        a_true, b_true = 5.0, 3.0
        wins = _simulate_thurstone_data(rng, λ_true; n_per_pair=40)
        anchor_labels = collect(1:n)
        y = [a_true + b_true * λ_true[i] + 0.1 * randn(rng) for i in anchor_labels]
        data = AnchoredData(PairwiseData(wins, anchor_labels), anchor_labels, y)

        f = fit(ThurstoneCaseVAnchored(), MLE(), data)
        @test f.result isa AnchoredMLEResult
        @test abs(sum(strengths(f))) < 1e-8
        cal = calibration(f)
        @test abs(cal.a - a_true) < 0.5
        @test abs(cal.b - b_true) < 0.6
        @test cal.σ² > 0
        preds = predict(f)
        @test length(preds) == n
        @test issorted(preds)
        lo, hi = predict(f, 5; prob=0.95)
        @test lo < preds[5] < hi
        @test predict(f, 5) ≈ preds[5]
        p = probability(f, n, 1)
        @test p > 0.5
        @test probability(f, 1, n) ≈ 1 - p atol=1e-10
        @test isfinite(loglikelihood(f))
    end

    @testset "ThurstoneCaseVAnchored — Bayesian recovery" begin
        rng = MersenneTwister(41)
        n = 10
        λ_true = collect(range(-1.5, 1.5, length=n)); λ_true .-= sum(λ_true) / n
        a_true, b_true = 5.0, 3.0
        wins = _simulate_thurstone_data(rng, λ_true; n_per_pair=40)
        y = [a_true + b_true * λ_true[i] + 0.1 * randn(rng) for i in 1:n]
        data = AnchoredData(PairwiseData(wins, collect(1:n)), collect(1:n), y)

        f = fit(ThurstoneCaseVAnchored(), Bayesian(n_samples=600, n_burnin=300, thin=2),
                data; rng=rng)
        @test f.result isa AnchoredMCMCSamples
        @test size(f.result.λ_samples) == (600, n)
        @test sortperm(posterior_mean(f)) == sortperm(λ_true)
        cal = calibration(f)
        @test abs(cal.a - a_true) < 0.6
        @test abs(cal.b - b_true) < 0.7
        draws = predict(f, 5; rng=rng)
        @test length(draws) == 600 && all(isfinite, draws)
        lo, hi = predict(f, 5; prob=0.95, rng=rng)
        @test lo < hi
        p = probability(f, n, 1)
        @test p > 0.5
        @test probability(f, 1, n) ≈ 1 - p atol=1e-10
        @test length(loglikelihood(f)) == 600
    end

    @testset "BradleyTerryAnchored — MLE recovery" begin
        rng = MersenneTwister(17)
        n = 10
        λ_true = collect(range(-1.5, 1.5, length=n)); λ_true .-= sum(λ_true) / n
        a_true, b_true = 4.0, 2.0
        wins = zeros(Int, n, n)
        for i in 1:n, j in 1:n
            i == j && continue
            p = 1 / (1 + exp(-(λ_true[i] - λ_true[j])))
            for _ in 1:25
                rand(rng) < p && (wins[i, j] += 1)
            end
        end
        y = [a_true + b_true * λ_true[i] + 0.1 * randn(rng) for i in 1:n]
        data = AnchoredData(PairwiseData(wins, collect(1:n)), collect(1:n), y)

        f = fit(BradleyTerryAnchored(), MLE(), data)
        @test f.result isa AnchoredMLEResult
        @test abs(sum(strengths(f))) < 1e-8
        cal = calibration(f)
        @test abs(cal.a - a_true) < 0.5
        @test abs(cal.b - b_true) < 0.6
        @test issorted(predict(f))
        p = probability(f, n, 1)
        @test p > 0.5
        @test probability(f, 1, n) ≈ 1 - p atol=1e-10
        @test probability(f, 1, 2) ≈ 1 / (1 + exp(-(strengths(f)[1] - strengths(f)[2])))
    end

    @testset "Covariate ThurstoneCaseV MLE recovery" begin
        rng = MersenneTwister(2025)
        βtrue = [1.5, -1.0, 0.0]
        cd, λtrue = _simulate_thurstone_covariate_data(rng, 45, βtrue; n_per_pair=16)
        f = fit(ThurstoneCaseVCovariates(), MLE(), cd)
        @test f.converged
        @test isapprox(collect(values(coefficients(f))), βtrue; atol=0.4)
        λ̂ = strengths(f)
        @test sum(λ̂) ≈ 0.0 atol=1e-8
        @test cor(λ̂, λtrue) > 0.95
        @test fit(ThurstoneCaseVCovariates(), cd).converged
        @test probability(f, "item1", "item2") ≈ probability(f, 1, 2)
        @test 0.0 < probability(f, 1, 2) < 1.0
        @test size(f.result.vcov) == (3, 3)
        @test all(diag(f.result.vcov) .> 0)
    end

    @testset "Covariate ThurstoneCaseV one-hot reproduces plain Thurstone" begin
        wins = [0 9 6; 3 0 8; 4 5 0]
        data = PairwiseData(wins, ["A", "B", "C"])
        Z = [0.0 0.0; 1.0 0.0; 0.0 1.0]
        cd = CovariateData(data, Z, [:I2, :I3])
        fcov = fit(ThurstoneCaseVCovariates(), MLE(), cd)
        ftcv = fit(ThurstoneCaseV(), MLE(), data)
        @test isapprox(strengths(fcov), strengths(ftcv); atol=1e-4)
    end

    @testset "Covariate ThurstoneCaseV Bayesian (Normal)" begin
        rng = MersenneTwister(99)
        βtrue = [1.2, -0.8]
        cd, _ = _simulate_thurstone_covariate_data(rng, 40, βtrue; n_per_pair=16)
        f = fit(ThurstoneCaseVCovariates(), Bayesian(n_samples=800, n_burnin=300),
                cd; rng=MersenneTwister(1))
        @test isapprox(collect(values(coefficients(f))), βtrue; atol=0.4)
        @test length(strengths(f)) == 40
        lo, hi = credible_interval(f, 1)
        @test lo < hi
        @test loglikelihood(f) isa Vector{Float64}
        @test 0.0 < probability(f, 1, 2) < 1.0
        cib = coefficient_intervals(f; level=0.9)
        @test cib.x1[1] > 0 && cib.x2[2] < 0
    end

    @testset "Covariate ThurstoneCaseV Horseshoe & Spike-slab" begin
        rng = MersenneTwister(123)
        βtrue = [2.0, -2.0, 0.0, 0.0]
        cd, _ = _simulate_thurstone_covariate_data(rng, 45, βtrue; n_per_pair=16)

        fh = fit(ThurstoneCaseVCovariates(), Bayesian(n_samples=800, n_burnin=400),
                 cd, HorseshoePrior(); rng=MersenneTwister(5))
        β̂ = collect(values(coefficients(fh)))
        @test minimum(abs.(β̂[1:2])) > 1.0
        @test all(abs.(β̂[3:4]) .< 0.5)

        fs = fit(ThurstoneCaseVCovariates(), Bayesian(n_samples=800, n_burnin=400),
                 cd, SpikeSlabPrior(); rng=MersenneTwister(6))
        pip = collect(values(inclusion_probabilities(fs)))
        @test pip[1] > 0.8 && pip[2] > 0.8
        @test pip[3] < 0.5 && pip[4] < 0.5
    end

    @testset "Covariate ThurstoneCaseV Stepwise selection" begin
        rng = MersenneTwister(555)
        βtrue = [1.8, -1.5, 0.0, 0.0, 0.0]
        cd, _ = _simulate_thurstone_covariate_data(rng, 50, βtrue; n_per_pair=18)
        f = fit(ThurstoneCaseVCovariates(), StepwiseMLE(direction=:both, criterion=:BIC), cd)
        @test sort(f.result.selected) == [1, 2]
        @test length(coefficients(f)) == 2
        @test !isempty(f.result.trace)
    end

    # ─── Rater heterogeneity and intransitivity ──────────────────────────────

    # Simulate rater-tagged data: rater r follows BT with reliability qtrue[r]
    # and otherwise guesses at random.
    function _simulate_rater_data(rng, K, λtrue, qtrue; n_per=4)
        M = length(qtrue)
        items = ["item" * lpad(i, 2, '0') for i in 1:K]
        raters = ["r$r" for r in 1:M]
        ws = String[]; ls = String[]; wh = String[]
        for r in 1:M, i in 1:K, j in (i + 1):K
            for _ in 1:n_per
                p = qtrue[r] / (1 + exp(-(λtrue[i] - λtrue[j]))) + (1 - qtrue[r]) / 2
                if rand(rng) < p
                    push!(ws, items[i]); push!(ls, items[j])
                else
                    push!(ws, items[j]); push!(ls, items[i])
                end
                push!(wh, raters[r])
            end
        end
        return RaterData(ws, ls, wh)
    end

    # Simulate data with a planted skew-symmetric intransitivity Γ.
    function _simulate_intransitive_data(rng, K, λtrue, Γ; n_per_pair=25)
        wins = zeros(Int, K, K)
        for i in 1:K, j in (i + 1):K
            p = 1 / (1 + exp(-(λtrue[i] - λtrue[j] + Γ[i, j])))
            for _ in 1:n_per_pair
                rand(rng) < p ? (wins[i, j] += 1) : (wins[j, i] += 1)
            end
        end
        return PairwiseData(wins, ["item" * lpad(i, 2, '0') for i in 1:K])
    end

    @testset "RaterData construction" begin
        rd = RaterData(["A", "C", "B"], ["B", "A", "C"], ["r1", "r1", "r2"])
        @test length(rd.winner) == 3
        @test sort(rd.labels) == ["A", "B", "C"]
        @test length(rd.raters) == 2
        @test_throws DimensionMismatch RaterData(["A"], ["B", "C"], ["r1", "r1"])
        @test_throws ArgumentError RaterData(["A", "A"], ["A", "B"], ["r1", "r1"])  # self-comparison
        rd2 = RaterData(["A", "B"], ["B", "A"], ["r1", "r2"];
                        item_labels=["A", "B"], rater_labels=["r1", "r2"])
        @test rd2.labels == ["A", "B"]
        @test_throws ArgumentError RaterData(["A", "B"], ["B", "A"], ["r1", "x"];
                                             rater_labels=["r1", "r2"])
    end

    @testset "BetaPrior and bundled priors" begin
        @test BetaPrior().a == 1.0 && BetaPrior().b == 1.0
        @test_throws ArgumentError BetaPrior(-1.0, 1.0)
        @test_throws ArgumentError BetaPrior(1.0, 0.0)
        rp = RaterHeterogeneityPrior()
        @test rp.λ_prior === nothing && rp.q_prior isa BetaPrior
        ip = IntransitivityPrior()
        @test ip.λ_prior === nothing && ip.σ²γ_prior isa InverseGammaPrior
    end

    @testset "Rater-heterogeneity BradleyTerry MLE recovery" begin
        rng = MersenneTwister(3)
        K = 10
        λtrue = collect(range(2.5, -2.5; length=K))
        qtrue = [0.95, 0.9, 0.85, 0.25, 0.15, 0.05]
        rd = _simulate_rater_data(rng, K, λtrue, qtrue; n_per=4)

        f = fit(BradleyTerryRaterHeterogeneity(), MLE(), rd)
        @test f.converged
        λ̂ = strengths(f)
        @test sum(λ̂) ≈ 0.0 atol=1e-8
        @test cor(λ̂, λtrue) > 0.85

        q = rater_reliabilities(f)
        @test keys(q) == (:r1, :r2, :r3, :r4, :r5, :r6)
        qv = collect(values(q))
        @test all(0.0 .<= qv .<= 1.0)
        @test mean(qv[1:3]) > mean(qv[4:6]) + 0.3        # reliable raters stand out

        @test fit(BradleyTerryRaterHeterogeneity(), rd).converged    # default-method overload
        @test probability(f, "item01", "item02") ≈ probability(f, 1, 2)
        @test 0.0 < probability(f, 1, 2) < 1.0
        @test_throws ArgumentError probability(f, "item01", "nope")
        @test loglikelihood(f) <= 0.0
    end

    @testset "Rater-heterogeneity BradleyTerry Bayesian recovery" begin
        rng = MersenneTwister(8)
        K = 10
        λtrue = collect(range(2.5, -2.5; length=K))
        qtrue = [0.92, 0.88, 0.2, 0.1]
        rd = _simulate_rater_data(rng, K, λtrue, qtrue; n_per=5)

        f = fit(BradleyTerryRaterHeterogeneity(), Bayesian(n_samples=800, n_burnin=400),
                rd; rng=MersenneTwister(1))
        @test length(strengths(f)) == K
        @test length(posterior_std(f)) == K
        @test cor(posterior_mean(f), λtrue) > 0.8
        lo, hi = credible_interval(f, 1)
        @test lo < hi
        @test loglikelihood(f) isa Vector{Float64}
        qv = collect(values(rater_reliabilities(f)))
        @test all(0.0 .<= qv .<= 1.0)
        @test mean(qv[1:2]) > mean(qv[3:4]) + 0.2
        @test 0.0 < probability(f, 1, 2) < 1.0
    end

    @testset "Intransitive BradleyTerry MLE recovery" begin
        rng = MersenneTwister(11)
        K = 8
        λtrue = collect(range(2.0, -2.0; length=K))
        Γ = zeros(K, K); g = 3.0
        for (a, b) in ((6, 7), (7, 8), (8, 6)); Γ[a, b] += g; Γ[b, a] -= g; end
        data = _simulate_intransitive_data(rng, K, λtrue, Γ; n_per_pair=25)

        f = fit(BradleyTerryIntransitive(), MLE(), data)
        @test f.converged
        λ̂ = strengths(f)
        @test sum(λ̂) ≈ 0.0 atol=1e-8
        @test cor(λ̂, λtrue) > 0.9

        Γ̂ = intransitivity(f)
        @test maximum(abs.(Γ̂ .+ transpose(Γ̂))) < 1e-10      # skew-symmetric
        cyc = [abs(Γ̂[6, 7]), abs(Γ̂[7, 8]), abs(Γ̂[6, 8])]
        off = [abs(Γ̂[i, j]) for i in 1:K for j in (i + 1):K
               if !((i, j) in ((6, 7), (7, 8), (6, 8)))]
        @test minimum(cyc) > 1.0
        @test minimum(cyc) > maximum(off)                    # planted cycle stands out

        # γ flips the order: item 8 beats item 6 despite a lower strength
        @test λ̂[8] < λ̂[6]
        @test probability(f, "item08", "item06") > 0.5
        @test probability(f, 8, 6) ≈ probability(f, "item08", "item06")
        @test_throws ArgumentError probability(f, "item08", "nope")

        @test fit(BradleyTerryIntransitive(), data).converged   # default-method overload
        fsmall = fit(BradleyTerryIntransitive(), MLE(), data; σ²γ=0.05)
        @test maximum(abs, intransitivity(fsmall)) < maximum(abs, intransitivity(f))
        @test_throws ArgumentError fit(BradleyTerryIntransitive(), MLE(), data; σ²γ=-1.0)
    end

    @testset "Intransitive BradleyTerry Bayesian recovery" begin
        rng = MersenneTwister(11)
        K = 8
        λtrue = collect(range(2.0, -2.0; length=K))
        Γ = zeros(K, K); g = 3.0
        for (a, b) in ((6, 7), (7, 8), (8, 6)); Γ[a, b] += g; Γ[b, a] -= g; end
        data = _simulate_intransitive_data(rng, K, λtrue, Γ; n_per_pair=25)

        f = fit(BradleyTerryIntransitive(), Bayesian(n_samples=800, n_burnin=400),
                data; rng=MersenneTwister(2))
        @test length(strengths(f)) == K
        @test cor(posterior_mean(f), λtrue) > 0.9
        @test length(posterior_std(f)) == K
        lo, hi = credible_interval(f, 1)
        @test lo < hi
        @test loglikelihood(f) isa Vector{Float64}
        Γ̄ = intransitivity(f)
        @test maximum(abs.(Γ̄ .+ transpose(Γ̄))) < 1e-10
        cyc = [abs(Γ̄[6, 7]), abs(Γ̄[7, 8]), abs(Γ̄[6, 8])]
        @test minimum(cyc) > 1.0
        @test 0.0 < probability(f, 1, 2) < 1.0
        @test all(f.result.σ²γ_samples .> 0)
    end

    # ─── Model checking: diagnostics and comparison ──────────────────────────

    # Separable pairwise data on K items with `npp` comparisons per pair.
    function _simulate_separable(rng, K; npp=12, spread=2.5)
        λ = collect(range(spread, -spread, length=K))
        wins = zeros(Int, K, K)
        for i in 1:K, j in (i + 1):K
            p = 1.0 / (1.0 + exp(-(λ[i] - λ[j])))
            for _ in 1:npp
                rand(rng) < p ? (wins[i, j] += 1) : (wins[j, i] += 1)
            end
        end
        return PairwiseData(wins, ["it$i" for i in 1:K]), λ
    end

    @testset "nparams / nobs" begin
        rng = MersenneTwister(1)
        data, _ = _simulate_separable(rng, 6)
        bt = fit(BradleyTerry(), MLE(), data)
        @test nparams(bt) == 5            # K - 1
        @test nobs(data) == sum(data.wins)
        cd, _ = _simulate_covariate_data(rng, 6, [1.0, -1.0])
        cm = fit(BradleyTerryCovariates(), MLE(), cd)
        @test nparams(cm) == 2
        @test nobs(cd) == sum(cd.data.wins)
    end

    @testset "pointwise_loglikelihood" begin
        rng = MersenneTwister(2)
        data, _ = _simulate_separable(rng, 5)
        bt = fit(BradleyTerry(), MLE(), data)
        pw = pointwise_loglikelihood(bt, data)
        @test pw isa Vector
        @test sum(pw) ≈ loglikelihood(bt) atol = 1e-8     # data loglik
        bayes = fit(BradleyTerry(), Bayesian(n_samples=200, n_burnin=100), data; rng=rng)
        pwb = pointwise_loglikelihood(bayes, data)
        @test size(pwb, 1) == 200
        @test all(isfinite, pwb)
    end

    @testset "AIC / BIC" begin
        rng = MersenneTwister(3)
        data, _ = _simulate_separable(rng, 7)
        bt = fit(BradleyTerry(), MLE(), data)
        @test aic(bt, data) ≈ -2 * loglikelihood(bt) + 2 * nparams(bt) atol = 1e-6
        @test bic(bt, data) ≈ -2 * loglikelihood(bt) + log(nobs(data)) * nparams(bt) atol = 1e-6
        @test deviance(bt, data) ≈ -2 * loglikelihood(bt) atol = 1e-6
        # Bayesian fits are redirected to waic/loo.
        bayes = fit(BradleyTerry(), Bayesian(n_samples=100, n_burnin=50), data; rng=rng)
        @test_throws ArgumentError aic(bayes, data)
        @test_throws ArgumentError bic(bayes, data)
    end

    @testset "WAIC / LOO" begin
        rng = MersenneTwister(4)
        data, _ = _simulate_separable(rng, 8)
        bayes = fit(BradleyTerry(), Bayesian(n_samples=800, n_burnin=400), data; rng=rng)
        w = waic(bayes, data)
        l = loo(bayes, data)
        @test w isa WAICResult && l isa LOOResult
        @test isfinite(w.elpd_waic) && isfinite(l.elpd_loo)
        @test w.p_waic > 0 && l.p_loo > 0
        @test w.waic ≈ -2 * w.elpd_waic
        @test l.looic ≈ -2 * l.elpd_loo
        @test all(isfinite, l.pareto_k)
        @test abs(w.elpd_waic - l.elpd_loo) < 2.0       # close on a well-behaved fit
        # MLE fits have no posterior draws.
        mle = fit(BradleyTerry(), MLE(), data)
        @test_throws ArgumentError waic(mle, data)
        @test_throws ArgumentError loo(mle, data)
    end

    @testset "SSR" begin
        rng = MersenneTwister(5)
        data, _ = _simulate_separable(rng, 8; npp=16)
        bt = fit(BradleyTerry(), MLE(), data)
        s_mle = ssr(bt, data)
        @test 0.0 < s_mle < 1.0
        @test s_mle > 0.8                                  # strongly separated items
        bayes = fit(BradleyTerry(), Bayesian(n_samples=600, n_burnin=300), data; rng=rng)
        @test abs(ssr(bayes) - s_mle) < 0.05               # posterior SD ≈ observed-info SE
        th = fit(ThurstoneCaseV(), MLE(), data)
        @test 0.0 < ssr(th, data) < 1.0
    end

    @testset "split-half reliability" begin
        rng = MersenneTwister(6)
        data, _ = _simulate_separable(rng, 8; npp=20)
        r = split_half_reliability(BradleyTerry(), MLE(), data; n_splits=30, rng=rng)
        @test r isa ReliabilityResult
        @test r.n_splits == 30 && length(r.per_split) == 30
        @test r.mean > 0.7                                 # reproducible scale
        @test r.spearman_brown >= r.mean                   # step-up never lowers it
    end

    @testset "train/test split and k-fold" begin
        rng = MersenneTwister(7)
        data, _ = _simulate_separable(rng, 6)
        tr, te = train_test_split(data; frac=0.7, rng=rng)
        @test nobs(tr) + nobs(te) == nobs(data)            # partition conserves comparisons
        @test tr.labels == data.labels
        folds = kfold(data; k=5, rng=rng)
        @test length(folds) == 5
        @test sum(nobs(te) for (_, te) in folds) == nobs(data)
    end

    @testset "cross-validated log loss" begin
        rng = MersenneTwister(8)
        # Data generated from covariates: the true covariate should predict better
        # out of sample than a random one.
        K = 14; Ztrue = randn(rng, K, 1); λ = Ztrue[:, 1] .* 2.0
        wins = zeros(Int, K, K)
        for i in 1:K, j in (i + 1):K
            p = 1.0 / (1.0 + exp(-(λ[i] - λ[j])))
            for _ in 1:8
                rand(rng) < p ? (wins[i, j] += 1) : (wins[j, i] += 1)
            end
        end
        labels = ["i$i" for i in 1:K]
        cd_true = CovariateData(PairwiseData(wins, labels), Ztrue, [:x])
        cd_rand = CovariateData(PairwiseData(wins, labels), randn(rng, K, 1), [:x])
        cv_true = crossvalidate(BradleyTerryCovariates(), MLE(), cd_true; k=5, rng=rng)
        cv_rand = crossvalidate(BradleyTerryCovariates(), MLE(), cd_rand; k=5, rng=rng)
        @test cv_true isa CVResult
        @test cv_true.mean_logloss < cv_rand.mean_logloss
    end

    @testset "likelihood-ratio test" begin
        rng = MersenneTwister(9)
        K = 12; Z = randn(rng, K, 3); βt = [1.8, -1.2, 0.0]; λ = Z * βt
        wins = zeros(Int, K, K)
        for i in 1:K, j in (i + 1):K
            p = 1.0 / (1.0 + exp(-(λ[i] - λ[j])))
            for _ in 1:10
                rand(rng) < p ? (wins[i, j] += 1) : (wins[j, i] += 1)
            end
        end
        labels = ["i$i" for i in 1:K]
        full = fit(BradleyTerryCovariates(), MLE(),
                   CovariateData(PairwiseData(wins, labels), Z, [:x1, :x2, :x3]))
        drop_noise = fit(BradleyTerryCovariates(), MLE(),
                         CovariateData(PairwiseData(wins, labels), Z[:, 1:2], [:x1, :x2]))
        drop_real = fit(BradleyTerryCovariates(), MLE(),
                        CovariateData(PairwiseData(wins, labels), Z[:, 1:1], [:x1]))
        t1 = lrtest(drop_noise, full)                      # drop x3 (noise)
        @test t1 isa LRTResult
        @test t1.df == 1
        @test t1.statistic >= 0.0
        @test t1.pvalue > 0.05                             # not significant
        t2 = lrtest(drop_real, drop_noise)                 # drop x2 (real)
        @test t2.pvalue < 0.01                             # significant
        @test_throws ArgumentError lrtest(full, drop_noise)  # not nested (wrong order)
        bt = fit(BradleyTerry(), MLE(), PairwiseData(wins, labels))
        @test_throws ArgumentError lrtest(bt, bt)          # not covariate fits
    end

    @testset "rank correlation and decision agreement" begin
        rng = MersenneTwister(10)
        data, _ = _simulate_separable(rng, 10)
        bt = fit(BradleyTerry(), MLE(), data)
        th = fit(ThurstoneCaseV(), MLE(), data)
        @test rank_correlation(bt, bt) ≈ 1.0               # self-correlation
        @test rank_correlation(bt, th) > 0.95              # link invariance
        @test rank_correlation(bt, th; method=:kendall) > 0.9
        @test rank_correlation(bt, th; method=:pearson) > 0.95
        @test_throws ArgumentError rank_correlation(bt, th; method=:bogus)
        @test top_k_agreement(bt, bt, 3) == 1.0
        @test 0.0 <= top_k_agreement(bt, th, 5) <= 1.0
        ba = boundary_agreement(bt, bt, 0.0)
        @test ba.agreement == 1.0
        @test ba.both_above + ba.both_below == length(data.labels)
    end

    @testset "compare table" begin
        rng = MersenneTwister(11)
        data, _ = _simulate_separable(rng, 8)
        btb = fit(BradleyTerry(), Bayesian(n_samples=500, n_burnin=250), data; rng=rng)
        thb = fit(ThurstoneCaseV(), Bayesian(n_samples=500, n_burnin=250), data; rng=rng)
        tbl = compare(btb, thb; data=data, criterion=:loo, names=["BT", "TCV"])
        @test tbl isa ModelComparisonTable
        @test length(tbl.values) == 2
        @test tbl.Δ[1] == 0.0                              # best model first
        @test issorted(tbl.values)
        @test all(tbl.Δ .>= 0.0)
        @test_throws ArgumentError compare(btb, thb; data=data, criterion=:bogus)
    end

end
