using DASSL, Test
using LinearAlgebra: diagm, I

const GROUP = get(ENV, "GROUP", "all")

if GROUP == "all" || GROUP == "core"
    @testset "Testing maxorder" begin
        F(t, y, dy) = (dy + y .^ 2)
        Fy(t, y, dy) = diagm(0 => 2y)
        Fdy(t, y, dy) = Matrix{Float64}(I, length(y), length(y))

        sol(t) = 1.0 ./ (1 .+ t)
        tspan = [0.0, 1.0]

        atol = 1.0e-5
        rtol = 1.0e-3

        # test the maxorder option
        for order in 1:6
            # scalar version
            (tn, yn, dyn) = DASSL.dasslSolve(F, sol(0.0), tspan, maxorder = order)
            aerror = maximum(abs.(yn - sol(tn)))
            rerror = maximum(abs.(yn - sol(tn)) ./ abs.(sol(tn)))
            nsteps = length(tn)

            @test aerror < (2 * nsteps * atol)
            @test rerror < (2 * nsteps * rtol)

            # vector version
            (tnV, ynV, dynV) = DASSL.dasslSolve(F, [sol(0.0)], tspan, maxorder = order)

            @test vcat(ynV...) == yn
            @test vcat(dynV...) == dyn

            # analytical jacobian version (vector)
            (
                tna, yna,
                dyna,
            ) = dasslSolve(
                F, [sol(0.0)], tspan, maxorder = order, Fy = Fy,
                Fdy = Fdy
            )
            aerror = maximum(abs.(map(first, yn) - sol(tn)))
            rerror = maximum(abs.(map(first, yn) - sol(tn)) ./ abs.(sol(tn)))
            nsteps = length(tn)

            @test aerror < (2 * nsteps * atol)
            @test rerror < (2 * nsteps * rtol)
        end
    end

    @testset "Testing common interface" begin
        include("common.jl")
    end

    @testset "DAE Initialization" begin
        include("initialization_tests.jl")
    end

    @testset "ModelingToolkit DAE Initialization" begin
        include("mtk_initialization_tests.jl")
    end

    @testset "Interface Compatibility" begin
        include("interface.jl")
    end

    include("convergence.jl")

    # In-place tests
    @testset "In-Place Operations" begin
        include("inplace_tests.jl")
    end
end

if GROUP == "all" || GROUP == "QA"
    @testset "QA" begin
        @testset "Explicit Imports" begin
            include("explicit_imports.jl")
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

        include("alloc_tests.jl")
    end
end
