using DASSL
using Test

@testset "QA" begin
    @testset "Explicit Imports" begin
        include(joinpath(@__DIR__, "..", "explicit_imports.jl"))
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

    include(joinpath(@__DIR__, "..", "alloc_tests.jl"))
end
