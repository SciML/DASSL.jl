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

        # Set a regression threshold to catch major regressions
        # The exact value depends on Julia version and dependencies
        @test allocs < 2000 * 1024  # Less than 2 MB
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

    @testset "In-place interpolation - zero allocations" begin
        x = [0.0, 0.1, 0.2]
        y = [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]]
        p_out = zeros(2)

        # Warmup
        DASSL.interpolateAt!(p_out, x, y, 0.15)

        # Test allocations
        allocs = @allocated DASSL.interpolateAt!(p_out, x, y, 0.15)
        @test allocs == 0

        # Test derivative interpolation
        DASSL.interpolateDerivativeAt!(p_out, x, y, 0.15)
        allocs = @allocated DASSL.interpolateDerivativeAt!(p_out, x, y, 0.15)
        @test allocs == 0

        # Test highest derivative
        DASSL.interpolateHighestDerivative!(p_out, x, y)
        allocs = @allocated DASSL.interpolateHighestDerivative!(p_out, x, y)
        @test allocs == 0
    end

    @testset "In-place weights! - zero allocations" begin
        wt = zeros(3)
        y = [1.0, 2.0, 3.0]

        # Warmup
        DASSL.weights!(wt, y, 1.0e-3, 1.0e-5)

        # Test allocations
        allocs = @allocated DASSL.weights!(wt, y, 1.0e-3, 1.0e-5)
        @test allocs == 0
    end

    @testset "In-place newton_iteration! - minimal allocations" begin
        y0 = [1.0, 2.0]
        cache = DASSL.alg_cache(dassl(), y0, nothing, 0.0, Val(true))
        wt = [1.0, 1.0]

        # Simple linear function for Newton iteration
        function f_test!(out, y)
            out[1] = -0.1 * (y[1] - 1.0)
            out[2] = -0.1 * (y[2] - 2.0)
        end

        # Pre-define normy to avoid closure allocation in measurement
        normy(v) = DASSL.dassl_norm(v, wt)

        # Warmup
        copyto!(cache.yn, y0)
        DASSL.newton_iteration!(cache, f_test!, y0, normy)

        # Test allocations - allow small allocations for closure captures
        copyto!(cache.yn, y0)
        allocs = @allocated DASSL.newton_iteration!(cache, f_test!, y0, normy)
        @test allocs < 256  # Less than 256 bytes (minimal closure overhead)
    end

    @testset "In-place numerical_jacobian! - zero allocations" begin
        y0 = [1.0, 2.0]
        cache = DASSL.alg_cache(dassl(), y0, nothing, 0.0, Val(true))

        # Simple in-place residual function
        function F_test!(out, t, y, dy)
            out[1] = dy[1] + y[1]
            out[2] = dy[2] + y[2]
        end

        dy0 = zeros(2)
        DASSL.weights!(cache.wt, y0, 1.0e-3, 1.0e-5)

        # Warmup
        DASSL.numerical_jacobian!(cache.jac, cache, F_test!, 0.0, y0, dy0, 1.0, cache.wt)

        # Test allocations
        allocs = @allocated DASSL.numerical_jacobian!(cache.jac, cache, F_test!, 0.0, y0, dy0, 1.0, cache.wt)
        @test allocs == 0
    end

    @testset "In-place stepper! allocation check" begin
        using LinearAlgebra: factorize

        y0 = [1.0]
        cache = DASSL.alg_cache(dassl(), y0, nothing, 0.0, Val(true))

        # Simple in-place residual function
        function F_step!(out, t, y, dy)
            out[1] = dy[1] + y[1]
        end

        dy0 = [0.0]

        # Initialize cache with some history
        DASSL.clear_history!(cache)
        DASSL.push_history!(cache, 0.0, y0, dy0)
        DASSL.weights!(cache.wt, y0, 1.0e-3, 1.0e-5)

        # Initialize Jacobian
        DASSL.numerical_jacobian!(cache.jac, cache, F_step!, 0.0, y0, dy0, 10.0, cache.wt)
        cache.jac_factorized = factorize(cache.jac)
        cache.a = 10.0

        normy(v) = DASSL.dassl_norm(v, cache.wt)

        # Warmup - first step allocates due to closures being compiled
        DASSL.stepper!(cache, 1, 0.1, F_step!, normy, 6)

        # Add another history point for multi-step test
        DASSL.push_history!(cache, 0.1, cache.yn, cache.dyn)
        DASSL.weights!(cache.wt, cache.yn, 1.0e-3, 1.0e-5)

        # Second step should have minimal allocations
        # Note: Some small allocations may occur due to closure captures
        allocs = @allocated DASSL.stepper!(cache, 1, 0.1, F_step!, normy, 6)

        # Allow small allocations for closure captures but catch major regressions
        # The key is that we're not allocating large arrays per step
        @test allocs < 1024  # Less than 1 KB per step
    end

    @testset "In-place dasslSolve! reduced allocations" begin
        y0 = [1.0]
        cache = DASSL.alg_cache(dassl(), y0, nothing, 0.0, Val(true))

        # Simple in-place residual function
        function F_inplace!(out, t, y, dy)
            out[1] = dy[1] + y[1]
        end

        tspan = [0.0, 1.0]

        # Warmup
        DASSL.clear_history!(cache)
        DASSL.dasslSolve!(cache, F_inplace!, y0, tspan)

        # Get allocation count for in-place version
        DASSL.clear_history!(cache)
        allocs_inplace = @allocated DASSL.dasslSolve!(cache, F_inplace!, y0, tspan)

        # Compare with out-of-place version
        F_oop(t, y, dy) = dy .+ y
        allocs_oop = @allocated DASSL.dasslSolve(F_oop, y0, tspan)

        # In-place should allocate significantly less than out-of-place
        # (primarily just output arrays vs output arrays + per-step allocations)
        @test allocs_inplace < allocs_oop
    end
end
