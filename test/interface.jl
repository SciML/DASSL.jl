# Interface compatibility tests for DASSL.jl
# Tests compliance with Julia's standard interfaces and SciML's array/number interface

using DASSL
using Test
using JLArrays

@testset "Interface Compatibility" begin
    @testset "BigFloat support" begin
        # Test that DASSL works with arbitrary precision arithmetic
        F(t, y, dy) = dy .+ y  # Simple ODE: dy/dt = -y

        # Test with BigFloat arrays
        y0 = [BigFloat(1.0)]
        tspan = [BigFloat(0.0), BigFloat(1.0)]

        (tn, yn, dyn) = dasslSolve(F, y0, tspan)

        @test eltype(yn[1]) == BigFloat
        @test eltype(dyn[1]) == BigFloat
        @test length(tn) > 1
        # Check that the solution is approximately exp(-t) at t=1
        @test isapprox(yn[end][1], exp(BigFloat(-1.0)), rtol = 1e-3)
    end

    @testset "BigFloat with Float64 time" begin
        # Test mixed precision: Float64 time, BigFloat state
        F(t, y, dy) = dy .+ y

        y0 = [BigFloat(1.0)]
        tspan = [0.0, 1.0]  # Float64 time

        (tn, yn, dyn) = dasslSolve(F, y0, tspan)

        @test eltype(yn[1]) == BigFloat
        @test length(tn) > 1
    end

    @testset "Vector BigFloat" begin
        # Test with multi-dimensional BigFloat arrays
        F(t, y, dy) = [dy[1] + y[1], dy[2] + y[2]]

        y0 = [BigFloat(1.0), BigFloat(2.0)]
        tspan = [0.0, 1.0]

        (tn, yn, dyn) = dasslSolve(F, y0, tspan)

        @test eltype(yn[1]) == BigFloat
        @test length(yn[1]) == 2
    end

    @testset "GPU-like array rejection" begin
        # Test that DASSL properly rejects arrays without fast scalar indexing
        F(t, y, dy) = dy .+ y

        y0_jl = JLArray([1.0])
        tspan = [0.0, 1.0]

        # The error is thrown inside a Task/Channel, so we get TaskFailedException
        # wrapping the actual ArgumentError
        error_thrown = false
        error_msg_correct = false
        try
            dasslSolve(F, y0_jl, tspan)
        catch e
            error_thrown = true
            # Handle TaskFailedException wrapping ArgumentError
            if e isa TaskFailedException
                inner = e.task.exception
                if inner isa ArgumentError
                    error_msg_correct = occursin("scalar indexing", inner.msg)
                end
            elseif e isa ArgumentError
                error_msg_correct = occursin("scalar indexing", e.msg)
            end
        end
        @test error_thrown
        @test error_msg_correct
    end

    @testset "Type preservation" begin
        # Test that eltype is properly preserved through computation
        F(t, y, dy) = dy .+ y

        # Float32
        y0_f32 = Float32[1.0]
        (tn, yn, dyn) = dasslSolve(F, y0_f32, [0.0, 1.0])
        @test eltype(yn[1]) == Float32

        # Float64
        y0_f64 = Float64[1.0]
        (tn, yn, dyn) = dasslSolve(F, y0_f64, [0.0, 1.0])
        @test eltype(yn[1]) == Float64
    end
end
