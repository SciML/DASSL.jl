using SciMLTesting, DASSL, Test

# DASSL historically exposes the DiffEqBase API as a solver-interface facade.
diffeqbase_reexports = Tuple(names(DASSL.DiffEqBase))

run_qa(
    DASSL;
    reexports_allow = diffeqbase_reexports,
    api_docs_kwargs = (; rendered_ignore = diffeqbase_reexports),
    ei_kwargs = (;
        all_explicit_imports_are_public = (;
            ignore = (
                :_process_verbose_param,    # DiffEqBase
            ),
        ),
    )
)

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
