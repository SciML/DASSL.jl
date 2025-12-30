# Convergence tests for DASSL.jl
# This file tests the convergence properties of the DASSL solver by comparing
# numerical solutions against analytical solutions at various tolerances.

using DASSL
using LinearAlgebra
using Test

# Test the convergence of dasslSolve. Returns a tuple of relative and absolute
# L^Inf norms of the difference between analytic and numerical solutions.
# The third element in the returned tuple is the time it took to obtain the
# numerical solution.
function dasslTestConvergence(F::Function,  # equation to solve
        y0::Vector{T}, # initial data
        tspan::Vector{T}, # time span of a solution
        sol::Function,  # analytic solution, for comparison with numerical solution
        rtol_range::Vector{T}, # vector of relative tolerances
        atol_range::Vector{T}; # vector of absolute tolerances
        dy0::Vector{T} = zero(y0)) where {T <: Number}
    if length(rtol_range) != length(atol_range)
        error("The table of relative errors and absolute errors should be of the same size.")
    end

    n = length(rtol_range)

    norms_abs = zeros(T, n)
    norms_rel = zeros(T, n)
    times = zeros(T, n)

    for i in 1:n
        times[i] = @elapsed begin
            (tn, yout, dyout) = dasslSolve(F, y0, tspan;
                dy0 = dy0,
                reltol = rtol_range[i],
                abstol = atol_range[i])
        end
        k = length(tn)
        delta_rel = zeros(T, k)
        delta_abs = zeros(T, k)

        for j in 1:k
            t = tn[j]
            y_exact = sol(t)
            y_num = yout[j]
            delta_abs[j] = norm(y_exact - y_num, Inf)
            delta_rel[j] = norm(y_exact - y_num, Inf) / norm(y_exact, Inf)
        end

        norms_abs[i] = norm(delta_abs, Inf)
        norms_rel[i] = norm(delta_rel, Inf)
    end

    return norms_rel, norms_abs, times
end

#------------------------------------------------------------
# Simple ODE example: dy/dt = -y, y(0) = 1
# Analytical solution: y(t) = exp(-t)
#------------------------------------------------------------

function F_exp(t::T, y::Vector{T}, dy::Vector{T}) where {T <: Number}
    dy .+ y
end

function sol_exp(t::T) where {T <: Number}
    [exp(-t)]
end

function dy_sol_exp(t::T) where {T <: Number}
    [-exp(-t)]
end

#------------------------------------------------------------
# Index-1 DAE example from Ascher & Petzold
# This is a linear index-1 DAE with algebraic constraint on y[3]
#------------------------------------------------------------

function F_dae(t::T, y::Vector{T}, dy::Vector{T}) where {T <: Number}
    a = 10.0
    [-dy[1] + (a - 1 / (2 - t)) * y[1] + (2 - t) * a * y[3] + exp(t) * (3 - t) / (2 - t),
        -dy[2] + (1 - a) / (t - 2) * y[1] - y[2] + (a - 1) * y[3] + 2 * exp(t),
        (t + 2) * y[1] + (t^2 - 4) * y[2] - (t^2 + t - 2) * exp(t)]
end

function sol_dae(t::T) where {T <: Number}
    [exp(t), exp(t), -exp(t) / (2 - t)]
end

function dy_sol_dae(t::T) where {T <: Number}
    [exp(t), exp(t), -exp(t) * (3 - t) / (2 - t)^2]
end

@testset "Convergence tests" begin
    # Test tolerances - use a moderate range that works for both ODE and DAE
    # Note: Very tight tolerances (< 1e-6) can cause issues with the DAE problem
    rtol_range = 10.0 .^ collect(-3.0:-1.0:-6.0)
    atol_range = 0.01 * rtol_range

    @testset "Simple ODE convergence" begin
        y0 = sol_exp(0.0)
        dy0 = dy_sol_exp(0.0)
        tspan = [0.0, 1.0]

        (rel_errors, abs_errors, times) = dasslTestConvergence(
            F_exp, y0, tspan, sol_exp, rtol_range, atol_range; dy0 = dy0)

        # Check that errors decrease as tolerances decrease
        # The errors should roughly follow the tolerance (within some factor)
        for i in 1:length(rtol_range)
            # Allow some margin - errors should be within 100x the tolerance
            @test abs_errors[i] < 100 * atol_range[i]
            @test rel_errors[i] < 100 * rtol_range[i]
        end

        # Check that errors generally decrease (not strictly monotonic due to noise)
        @test abs_errors[1] > abs_errors[end]
        @test rel_errors[1] > rel_errors[end]
    end

    @testset "Index-1 DAE convergence" begin
        y0 = sol_dae(0.0)
        dy0 = dy_sol_dae(0.0)
        tspan = [0.0, 1.0]

        (rel_errors, abs_errors, times) = dasslTestConvergence(
            F_dae, y0, tspan, sol_dae, rtol_range, atol_range; dy0 = dy0)

        # Check that errors decrease as tolerances decrease
        for i in 1:length(rtol_range)
            # DAE problems may have larger errors than pure ODEs
            # Allow 1000x margin for the algebraic constraint
            @test abs_errors[i] < 1000 * atol_range[i]
            @test rel_errors[i] < 1000 * rtol_range[i]
        end

        # Check that errors generally decrease
        @test abs_errors[1] > abs_errors[end]
        @test rel_errors[1] > rel_errors[end]
    end
end
