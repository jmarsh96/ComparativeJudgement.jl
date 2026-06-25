# Building the documentation

The documentation is built with [Documenter.jl](https://documenter.juliadocs.org/).
Its environment (`docs/Project.toml`) dev-depends on the package itself, so it
always documents your working copy.

All commands below are run from the **package root**.

## One-time setup

Instantiate the docs environment (and dev-link the package into it):

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

## Build once

```bash
julia --project=docs docs/make.jl
```

The rendered site is written to `docs/build/`. The `@example` blocks in the
tutorials are executed during the build (including the MCMC fits), so a clean
build also serves as an end-to-end check that the examples still run.

## Live preview with LiveServer

[LiveServer.jl](https://github.com/JuliaDocs/LiveServer.jl)'s `servedocs` rebuilds
the docs whenever a source file changes and serves them locally with automatic
browser reload — the fastest loop when editing pages.

Add LiveServer to the docs environment once:

```bash
julia --project=docs -e 'using Pkg; Pkg.add("LiveServer")'
```

Then start the live server:

```bash
julia --project=docs -e 'using ComparativeJudgement, LiveServer; servedocs()'
```

Open <http://localhost:8000> in a browser. `servedocs` watches `docs/src/`,
`docs/make.jl`, and the package `src/`, so edits to a page — or to a docstring —
trigger a rebuild and the open page refreshes itself. Stop the server with
`Ctrl-C`.

To instead serve an already-built site without watching for changes:

```bash
julia --project=docs -e 'using LiveServer; serve(dir="docs/build")'
```
