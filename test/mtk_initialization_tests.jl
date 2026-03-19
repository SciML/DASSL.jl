# Tests for ModelingToolkit DAE initialization support
using DASSL
using Test
using ModelingToolkit, SciMLBase
using SymbolicIndexingInterface
using ModelingToolkit: t_nounits as t, D_nounits as D

@testset "ModelingToolkit DAE Initialization" begin
    @testset "DAE with @mtkcompile, initialization_eqs, and missing parameters" begin
        @variables x(t) [guess = 1.0] y(t) [guess = 1.0]
        @parameters p = missing [guess = 1.0] q = missing [guess = 1.0]
        @mtkcompile sys = System([D(x) ~ p * y + q * t, x^3 + y^3 ~ 5], t; initialization_eqs = [p^2 + q^2 ~ 3])

        @testset "DAEProblem{$iip}" for iip in [true, false]
            prob = DAEProblem(sys, [D(x) => cbrt(4), D(y) => -1 / cbrt(4), p => 1.0], (0.0, 0.4))
            
            @testset "DefaultInit (OverrideInit → CheckInit)" begin
                # DefaultInit first runs OverrideInit to solve for initial conditions,
                # then CheckInit to verify them (following Sundials v5 pattern)
                sol = solve(prob, dassl())
                @test SciMLBase.successful_retcode(sol)
                @test sol[x, 1] ≈ 1.0
                @test sol[y, 1] ≈ cbrt(4)
                @test sol.ps[p] ≈ 1.0
                @test sol.ps[q] ≈ sqrt(2)
            end
            
            @testset "CheckInit" begin
                prob = DAEProblem(sys, [D(x) => cbrt(4), D(y) => -1 / cbrt(4), p => 1.0], (0.0, 0.4))
                # CheckInit should fail because the initial conditions aren't consistent yet
                # (they need to be computed by OverrideInit)
                @test_throws Any solve(prob, dassl(); initializealg = SciMLBase.CheckInit())
                
                # Create a new problem with fully consistent initial values
                # D(x) = p*y = 1*cbrt(4) = cbrt(4)
                # D(y) = -x²/y²*D(x) = -1/cbrt(4)²*cbrt(4) = -1/cbrt(4)
                prob_correct = DAEProblem(
                    sys,
                    [D(x) => cbrt(4), D(y) => -1 / cbrt(4), p => 1.0, x => 1.0, y => cbrt(4), q => sqrt(2)],
                    (0.0, 0.4)
                )
                # Need to convert to IIP/OOP after creation to get proper numeric arrays
                if iip
                    prob_correct_typed = DAEProblem{true}(
                        prob_correct.f, prob_correct.du0, prob_correct.u0,
                        prob_correct.tspan, prob_correct.p
                    )
                else
                    prob_correct_typed = DAEProblem{false}(
                        prob_correct.f, prob_correct.du0, prob_correct.u0,
                        prob_correct.tspan, prob_correct.p
                    )
                end
                sol_correct = solve(prob_correct_typed, dassl(); initializealg = SciMLBase.CheckInit())
                @test SciMLBase.successful_retcode(sol_correct)
            end
        end
    end

    @testset "Simple DAE with algebraic constraint" begin
        # A simpler test case: x' = -k*x, y = x² (algebraic constraint)
        @variables x(t) [guess = 1.0] y(t) [guess = 1.0]
        @parameters k = missing [guess = 1.0]
        
        @mtkcompile sys = System(
            [D(x) ~ -k * x, y ~ x^2],
            t;
            initialization_eqs = [k ~ 2.0]
        )
        
        prob = DAEProblem(sys, [D(x) => -2.0], (0.0, 1.0))
        
        sol = solve(prob, dassl())
        @test SciMLBase.successful_retcode(sol)
        @test sol[x, 1] ≈ 1.0
        @test sol[y, 1] ≈ 1.0
        @test sol.ps[k] ≈ 2.0
        
        # Verify algebraic constraint holds throughout
        for i in eachindex(sol.t)
            @test sol[y, i] ≈ sol[x, i]^2 atol=1e-4
        end
    end
end
