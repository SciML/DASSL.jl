using SciMLTesting, DASSL, Test

# All flagged names are non-public in their *own* packages (SciMLBase/DiffEqBase/
# ArrayInterface internals DASSL legitimately depends on); ignore them here and drop
# each as its source package marks the name public.
run_qa(
    DASSL; explicit_imports = true,
    ei_kwargs = (;
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractDAEAlgorithm,      # SciMLBase
                :AbstractDAEProblem,        # SciMLBase
                :NoInit,                    # SciMLBase
                :OverrideInit,              # SciMLBase
                :build_solution,            # SciMLBase
                :get_initial_values,        # SciMLBase
                :has_initialization_data,   # SciMLBase
                :_process_verbose_param,    # DiffEqBase
                :fast_scalar_indexing,      # ArrayInterface
            ),
        ),
        # ReturnCode enum members are not marked public on Julia 1.10's view of
        # SciMLBase.ReturnCode (passes on 1.11+); ignore as SciMLBase-owned names.
        all_qualified_accesses_are_public = (;
            ignore = (
                :Success,                   # SciMLBase.ReturnCode
                :InitialFailure,            # SciMLBase.ReturnCode
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
