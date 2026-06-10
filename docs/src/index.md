```@meta
CurrentModule = ComparativeJudgement
```

# ComparativeJudgement

[ComparativeJudgement](https://github.com/jmarsh96/ComparativeJudgement.jl)
fits comparative judgement models to pairwise comparison data: items are
placed on a latent strength scale from the outcomes of head-to-head
comparisons.

The package currently implements the Bradley–Terry model with three
workflows:

- **Maximum likelihood** — fast point estimates and rankings.
- **Bayesian** — Pólya-Gamma augmented Gibbs sampling for full posterior
  uncertainty.
- **Anchored** — a joint model in which known measurements for a few items
  calibrate the latent scale, so measurements can be predicted for all items.

See the [Bradley–Terry models](bradley_terry.md) tutorial for a worked
example of all three on simulated data, and the [API reference](api.md) for
the full interface.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/jmarsh96/ComparativeJudgement.jl")
```

## Quick start

```julia
using ComparativeJudgement

wins = [0 12 15;
        8  0 13;
        5  7  0]                      # wins[i, j]: times item i beat item j
data = PairwiseData(wins, ["A", "B", "C"])

fitted = fit(BradleyTerry(), MLE(), data)
strengths(fitted)                     # estimated latent strengths
probability(fitted, "A", "C")         # P(A beats C)
```
