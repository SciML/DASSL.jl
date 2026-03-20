using DASSL
using Test

@testset "In-Place Operations" begin
    @testset "Cache allocation" begin
        u0 = [1.0, 2.0]
        cache = DASSL.alg_cache(dassl(), u0, nothing, 0.0, Val(true))

        @test size(cache.yn) == size(u0)
        @test size(cache.dyn) == size(u0)
        @test size(cache.y0) == size(u0)
        @test size(cache.dy0) == size(u0)
        @test size(cache.delta) == size(u0)
        @test size(cache.residual) == size(u0)
        @test size(cache.jac) == (2, 2)
        @test length(cache.t_hist) == DASSL.MAXORDER + 3
        @test length(cache.y_hist) == DASSL.MAXORDER + 3
        @test length(cache.dy_hist) == DASSL.MAXORDER + 3
        @test cache.hist_len == 0
    end

    @testset "History buffer operations" begin
        u0 = [1.0, 2.0]
        cache = DASSL.alg_cache(dassl(), u0, nothing, 0.0, Val(true))

        # Push some entries
        DASSL.push_history!(cache, 0.0, [1.0, 2.0], [0.1, 0.2])
        @test cache.hist_len == 1
        @test DASSL.get_latest_t(cache) == 0.0
        @test DASSL.get_latest_y(cache) == [1.0, 2.0]

        DASSL.push_history!(cache, 0.1, [1.1, 2.1], [0.11, 0.21])
        @test cache.hist_len == 2
        @test DASSL.get_latest_t(cache) == 0.1
        @test DASSL.get_t_at(cache, 1) == 0.0
        @test DASSL.get_t_at(cache, 2) == 0.1

        # Test history retrieval
        t_hist = DASSL.get_history_t(cache, 2)
        @test t_hist == [0.0, 0.1]

        # Clear and verify
        DASSL.clear_history!(cache)
        @test cache.hist_len == 0
    end

    @testset "In-place interpolation" begin
        # Test interpolateAt! with quadratic Lagrange interpolation
        x = [0.0, 1.0, 2.0]
        y = [[1.0, 2.0], [2.0, 3.0], [5.0, 6.0]]
        p_out = zeros(2)

        DASSL.interpolateAt!(p_out, x, y, 0.5)
        # Verify against out-of-place version
        p_expected = DASSL.interpolateAt(x, y, 0.5)
        @test p_out ≈ p_expected

        # Test interpolateDerivativeAt!
        fill!(p_out, 0.0)
        DASSL.interpolateDerivativeAt!(p_out, x, y, 1.0)
        p_deriv_expected = DASSL.interpolateDerivativeAt(x, y, 1.0)
        @test p_out ≈ p_deriv_expected
    end

    @testset "dasslSolve! basic functionality" begin
        # Simple exponential decay: dy + y = 0
        F!(out, t, y, dy) = (out .= dy .+ y)
        y0 = [1.0]
        tspan = [0.0, 1.0]

        cache = DASSL.alg_cache(dassl(), y0, nothing, 0.0, Val(true))
        (tout, yout, dyout) = dasslSolve!(cache, F!, y0, tspan)

        @test length(tout) > 1
        @test tout[1] == 0.0
        @test tout[end] ≈ 1.0 atol = 1.0e-10
        # Solution should be approximately exp(-t)
        @test isapprox(yout[end][1], exp(-1.0), rtol = 1.0e-3)
    end

    @testset "In-place vs out-of-place consistency" begin
        # Same problem solved both ways should give approximately same answer
        # Note: Due to different code paths, exact match is not expected,
        # but the solutions should converge to the same analytical result
        F!(out, t, y, dy) = (out .= dy .+ y)
        F(t, y, dy) = dy .+ y

        y0 = [1.0]
        tspan = [0.0, 1.0]

        # Out-of-place
        (t1, y1, dy1) = dasslSolve(F, y0, tspan)

        # In-place
        cache = DASSL.alg_cache(dassl(), y0, nothing, 0.0, Val(true))
        (t2, y2, dy2) = dasslSolve!(cache, F!, y0, tspan)

        # Both should converge to the analytical solution exp(-1) ≈ 0.368
        analytical = exp(-1.0)
        @test isapprox(y1[end][1], analytical, rtol = 1.0e-3)
        @test isapprox(y2[end][1], analytical, rtol = 1.0e-3)
        # Check that the solutions are reasonably close to each other
        @test isapprox(y1[end][1], y2[end][1], rtol = 1.0e-2)
    end

    @testset "DiffEqBase interface in-place" begin
        # Test through standard interface with in-place problem
        f_dae = (out, du, u, p, t) -> (out .= du .+ u)
        prob = DAEProblem(
            f_dae, [-1.0], [1.0], (0.0, 1.0), nothing;
            differential_vars = [true]
        )
        sol = solve(prob, dassl())

        # Check solution quality (retcode may be Default in some versions)
        @test length(sol.t) > 1
        @test isapprox(sol.u[end][1], exp(-1.0), rtol = 1.0e-3)
    end

    @testset "Multi-dimensional in-place problem" begin
        # 2D system: coupled equations
        # dy1 + y1 + y2 = 0
        # dy2 + y1 - y2 = 0
        F!(out, t, y, dy) = begin
            out[1] = dy[1] + y[1] + y[2]
            out[2] = dy[2] + y[1] - y[2]
        end

        y0 = [1.0, 0.5]
        tspan = [0.0, 1.0]

        cache = DASSL.alg_cache(dassl(), y0, nothing, 0.0, Val(true))
        (tout, yout, dyout) = dasslSolve!(cache, F!, y0, tspan)

        @test length(tout) > 1
        @test tout[end] ≈ 1.0 atol = 1.0e-10
        @test length(yout[end]) == 2
    end

    @testset "Cache reuse across multiple solves" begin
        # Cache should be reusable for multiple solves with same dimensions
        F!(out, t, y, dy) = (out .= dy .+ y)
        y0 = [1.0]

        cache = DASSL.alg_cache(dassl(), y0, nothing, 0.0, Val(true))

        # First solve
        (t1, y1, _) = dasslSolve!(cache, F!, y0, [0.0, 0.5])
        @test isapprox(y1[end][1], exp(-0.5), rtol = 1.0e-3)

        # Clear cache and solve again with different initial condition
        DASSL.clear_history!(cache)
        (t2, y2, _) = dasslSolve!(cache, F!, [2.0], [0.0, 0.5])
        @test isapprox(y2[end][1], 2.0 * exp(-0.5), rtol = 1.0e-3)
    end
end
