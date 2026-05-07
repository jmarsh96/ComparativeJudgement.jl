using ComparativeJudgement
using Documenter

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
    ],
)

deploydocs(;
    repo="github.com/jmarsh96/ComparativeJudgement.jl",
    devbranch="main",
)
