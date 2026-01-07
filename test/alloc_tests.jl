# Allocation tests for DASSL.jl
# These tests verify that key internal functions don't allocate unnecessarily

using DASSL
using AllocCheck
using Test

@testset "Allocation Tests" begin
    @testset "dassl_norm - zero allocations" begin
        v = [1.0, 2.0, 3.0]
        wt = [0.5, 0.5, 0.5]
        # Warmup
        DASSL.dassl_norm(v, wt)
        # Test allocations
        allocs = @allocated DASSL.dassl_norm(v, wt)
        @test allocs == 0
    end

    @testset "_all_steps_equal - zero allocations" begin
        t = [0.0, 0.1, 0.2, 0.3, 0.4]
        # Warmup
        DASSL._all_steps_equal(t)
        # Test allocations
        allocs = @allocated DASSL._all_steps_equal(t)
        @test allocs == 0
    end

    @testset "Benchmark regression tests" begin
        # Simple ODE benchmark - verify allocations don't regress significantly
        F_exp(t, y, dy) = dy .+ y
        y0 = [1.0]
        tspan = [0.0, 1.0]

        # Warmup
        DASSL.dasslSolve(F_exp, y0, tspan)

        # Get allocation count
        allocs = @allocated DASSL.dasslSolve(F_exp, y0, tspan)

        # After optimizations, this should be around 100KB
        # Set a regression threshold at 150KB to catch any significant regression
        @test allocs < 150 * 1024  # Less than 150 KB
    end

    @testset "Larger system benchmark regression" begin
        # 5-equation system
        function F_large(t, y, dy)
            n = length(y)
            out = similar(y)
            for i in 1:n
                out[i] = dy[i] + y[i]
                if i > 1
                    out[i] += 0.1 * y[i - 1]
                end
                if i < n
                    out[i] += 0.1 * y[i + 1]
                end
            end
            return out
        end

        y0_large = ones(5)
        tspan = [0.0, 1.0]

        # Warmup
        DASSL.dasslSolve(F_large, y0_large, tspan)

        # Get allocation count
        allocs = @allocated DASSL.dasslSolve(F_large, y0_large, tspan)

        # After optimizations, this should be around 186KB
        # Set a regression threshold at 250KB
        @test allocs < 250 * 1024  # Less than 250 KB
    end
end
