using DASSL, Test

@testset "Testing maxorder" begin

    F(t,y,dy)=(dy+y.^2)
    Fy(t,y,dy)=diagm(2y)
    Fdy(t,y,dy)=eye(length(y))

    sol(t)=1.0./(1 .+ t)
    tspan=[0.0, 1.0]

    atol = 1.0e-5
    rtol = 1.0e-3


    # test the maxorder option
    for order=1:6
        # scalar version
        (tn,yn,dyn)=DASSL.dasslSolve(F, sol(0.0), tspan, maxorder = order)
        aerror = maximum(abs.(yn-sol(tn)))
        rerror = maximum(abs.(yn-sol(tn))./abs.(sol(tn)))
        nsteps = length(tn)

        @test aerror < (2*nsteps*atol)
        @test rerror < (2*nsteps*rtol)

        # vector version
        (tnV,ynV,dynV)=DASSL.dasslSolve(F,[sol(0.0)], tspan, maxorder = order)

        @test  vcat(ynV...) == yn
        @test  vcat(dynV...) == dyn

        # analytical jacobian version (vector)
        (tna,yna,dyna)=dasslSolve(F, [sol(0.0)], tspan, maxorder = order, Fy = Fy, Fdy = Fdy)
        aerror = maximum(abs.(map(first,yn)-sol(tn)))
        rerror = maximum(abs.(map(first,yn)-sol(tn))./abs.(sol(tn)))
        nsteps = length(tn)

        @test aerror < (2*nsteps*atol)
        @test rerror < (2*nsteps*rtol)
    end
end

@testset "Testing common interface" begin
  include("common.jl")
end
