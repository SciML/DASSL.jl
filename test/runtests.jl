using Test
using SafeTestsets

const GROUP = let g = get(ENV, "GROUP", "All")
    isempty(g) ? "All" : g
end

if GROUP == "All" || GROUP == "Core"
    @safetestset "Testing maxorder" begin
        using DASSL
        using LinearAlgebra: diagm, I
        using Test

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

    @safetestset "Testing common interface" begin
        include("common.jl")
    end

    @safetestset "DAE Initialization" begin
        include("initialization_tests.jl")
    end

    @safetestset "ModelingToolkit DAE Initialization" begin
        include("mtk_initialization_tests.jl")
    end

    @safetestset "Interface Compatibility" begin
        include("interface.jl")
    end

    @safetestset "Convergence tests" begin
        include("convergence.jl")
    end

    # In-place tests
    @safetestset "In-Place Operations" begin
        include("inplace_tests.jl")
    end
end

if GROUP == "QA"
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.develop(path = joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    include(joinpath(@__DIR__, "qa", "qa.jl"))
end
