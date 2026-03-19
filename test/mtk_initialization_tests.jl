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
            
            @testset "OverrideInit" begin
                integ = init(prob, dassl())
                @test integ.initializealg isa DASSL.DefaultInit
                @test integ[x] ≈ 1.0
                @test integ[y] ≈ cbrt(4)
                @test integ.ps[p] ≈ 1.0
                @test integ.ps[q] ≈ sqrt(2)
                sol = solve(prob, dassl())
                @test SciMLBase.successful_retcode(sol)
                @test sol[x, 1] ≈ 1.0
                @test sol[y, 1] ≈ cbrt(4)
                @test sol.ps[p] ≈ 1.0
                @test sol.ps[q] ≈ sqrt(2)
            end
            
            @testset "CheckInit" begin
                prob = DAEProblem(sys, [D(x) => cbrt(4), D(y) => -1 / cbrt(4), p => 1.0], (0.0, 0.4))
                @test_throws Any init(prob, dassl(); initializealg = SciMLBase.CheckInit())
                
                # Create a new problem with correct initial values
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
                @test_nowarn init(prob_correct_typed, dassl(); initializealg = SciMLBase.CheckInit())
            end
        end
    end

    @testset "Simple DAE with algebraic constraint" begin
        # A simpler test case: x' = -x, y = x² (algebraic constraint)
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
