using ComparativeJudgement
using Documenter

ENV["GKSwstype"] = "100"   # headless GR for the Plots-based tutorial figures

DocMeta.setdocmeta!(ComparativeJudgement, :DocTestSetup, :(using ComparativeJudgement); recursive=true)

makedocs(;
    modules=[ComparativeJudgement],
    authors="Joseph Marsh <joe.s.marsh@gmail.com> and contributors",
    sitename="ComparativeJudgement.jl",
    format=Documenter.HTML(;
        canonical="https://jmarsh96.github.io/ComparativeJudgement.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Bradley–Terry models" => "bradley_terry.md",
        "Anchored models" => "anchored_bt.md",
        "Covariate models" => "covariate_bt.md",
        "Anchored covariate models" => "covariate_anchored_bt.md",
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/jmarsh96/ComparativeJudgement.jl",
    devbranch="main",
)
