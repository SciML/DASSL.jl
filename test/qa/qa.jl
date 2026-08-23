using SciMLTesting, DASSL, Test

# The SciML common interface DASSL deliberately reexports so that `using DASSL` is enough
# to build a DAE problem, solve it, and inspect the result. Owned and documented upstream;
# kept in sync with the reexport `export` blocks in src/DASSL.jl.
const REEXPORTS = (
    :CheckInit, :DAEFunction, :DAEProblem, :DAESolution, :DEStats, :DefaultInit,
    :EnsembleAnalysis, :EnsembleDistributed, :EnsembleProblem, :EnsembleSerial,
    :EnsembleSolution, :EnsembleSplitThreads, :EnsembleSummary, :EnsembleThreads,
    :NoInit, :NullParameters, :OverrideInit, :ReturnCode, :remake, :solve,
    :successful_retcode,
)

run_qa(DASSL; reexports_allow = REEXPORTS)

@testset "Reexport surface" begin
    # Every approved reexport must actually be reachable from `using DASSL`, so the
    # allow-list cannot drift into approving names the package no longer provides.
    # `isdefined(@__MODULE__, ...)` tests the property directly: this file's
    # `using DASSL` is what has to bring the name into scope.
    @testset "$name" for name in REEXPORTS
        @test name in names(DASSL)
        @test isdefined(@__MODULE__, name)
    end
end

@testset "Type Stability" begin
    alg = dassl()
    @test typeof(alg.maxorder) === Int
    @test typeof(alg.factorize_jacobian) === Bool

    y0 = [1.0, 2.0]
    cache = DASSL.alg_cache(alg, y0, nothing, 0.0, Val(true))
    @test isconcretetype(typeof(cache.jac_factorized))
    @test !(cache.jac_factorized isa Any && typeof(cache.jac_factorized) === Any)
end

include(joinpath(@__DIR__, "..", "shared", "alloc_tests.jl"))
