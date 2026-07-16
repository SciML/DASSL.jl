using Pkg

Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
Pkg.instantiate()

using DASSL
using Documenter

makedocs(;
    modules = [DASSL],
    authors = "Chris Rackauckas <accounts@chrisrackauckas.com>",
    sitename = "DASSL.jl",
    checkdocs = :exports,
    format = Documenter.HTML(;
        canonical = "https://docs.sciml.ai/DASSL/stable/",
        edit_link = "master",
        assets = String[],
    ),
    pages = [
        "Public API" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/SciML/DASSL.jl.git",
    push_preview = true,
)
