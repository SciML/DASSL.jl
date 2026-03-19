# Tests for DAE initialization support
using DASSL
using Test
using SciMLBase: NoInit, CheckInit, OverrideInit

@testset "DAE Initialization" begin
    @testset "DefaultInit with consistent ICs" begin
        # Simple ODE as DAE: du + u = 0
        # Consistent IC: u0 = 1.0, du0 = -1.0 (satisfies du0 + u0 = 0)
        f_dae = (out, du, u, p, t) -> (out .= du .+ u)
        prob = DAEProblem(
            f_dae, [-1.0], [1.0], (0.0, 1.0), nothing;
            differential_vars = [true]
        )

        # DefaultInit should work (falls back to CheckInit)
        sol = solve(prob, dassl())
        @test SciMLBase.successful_retcode(sol)
        @test length(sol.t) > 1
    end

    @testset "NoInit" begin
        # With NoInit, even inconsistent ICs are accepted (no check)
        f_dae = (out, du, u, p, t) -> (out .= du .+ u)

        # Inconsistent IC: u0 = 1.0, du0 = 0.0 (du0 + u0 = 1 ≠ 0)
        prob = DAEProblem(
            f_dae, [0.0], [1.0], (0.0, 1.0), nothing;
            differential_vars = [true]
        )

        # NoInit skips the check - solver may still work due to DASSL's internal handling
        sol = solve(prob, dassl(); initializealg = NoInit())
        # We just verify it doesn't throw during initialization
        @test true
    end

    @testset "CheckInit with consistent ICs" begin
        # Consistent IC: du0 + u0 = 0
        f_dae = (out, du, u, p, t) -> (out .= du .+ u)
        prob = DAEProblem(
            f_dae, [-1.0], [1.0], (0.0, 1.0), nothing;
            differential_vars = [true]
        )

        sol = solve(prob, dassl(); initializealg = CheckInit())
        @test SciMLBase.successful_retcode(sol)
    end

    @testset "CheckInit with inconsistent ICs throws" begin
        # Inconsistent IC: du0 + u0 ≠ 0
        f_dae = (out, du, u, p, t) -> (out .= du .+ u)
        prob = DAEProblem(
            f_dae, [0.0], [1.0], (0.0, 1.0), nothing;
            differential_vars = [true]
        )

        @test_throws ErrorException solve(prob, dassl(); initializealg = CheckInit())
    end

    @testset "Vector system with initialization" begin
        # 2D system: du1 + u1 = 0, du2 + u2 = 0
        f_dae = (out, du, u, p, t) -> begin
            out[1] = du[1] + u[1]
            out[2] = du[2] + u[2]
        end

        # Consistent ICs
        prob = DAEProblem(
            f_dae, [-1.0, -2.0], [1.0, 2.0], (0.0, 1.0), nothing;
            differential_vars = [true, true]
        )

        sol = solve(prob, dassl(); initializealg = CheckInit())
        @test SciMLBase.successful_retcode(sol)
        @test length(sol.u[end]) == 2
    end

    @testset "OverrideInit without initialization_data" begin
        # OverrideInit without initialization_data should work like NoInit
        f_dae = (out, du, u, p, t) -> (out .= du .+ u)
        prob = DAEProblem(
            f_dae, [-1.0], [1.0], (0.0, 1.0), nothing;
            differential_vars = [true]
        )

        sol = solve(prob, dassl(); initializealg = OverrideInit())
        @test SciMLBase.successful_retcode(sol)
    end

    @testset "DefaultInit export" begin
        # DefaultInit comes from DiffEqBase (reexported by DASSL)
        @test DefaultInit <: SciMLBase.DAEInitializationAlgorithm
        @test DefaultInit() isa DefaultInit
    end
end
