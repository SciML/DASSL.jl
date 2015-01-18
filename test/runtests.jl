using DASSL
using Base.Test

F(t,y,dy)=(dy+y.^2)
Fy(t,y,dy)=diagm(2y)
Fdy(t,y,dy)=eye(length(y))

sol(t)=1./(1 .+ t)
const tspan=[0.0, 1.0]

atol = 1.0e-5
rtol = 1.0e-3


# test the maxorder option
for order=1:6
    # scalar version
    (tn,yn,dyn)=DASSL.dasslSolve(F, sol(0.0), tspan, maxorder = order)
    aerror = maximum(abs(yn-sol(tn)))
    rerror = maximum(abs(yn-sol(tn))/abs(sol(tn)))
    nsteps = length(tn)

    @test aerror < nsteps*atol
    @test rerror < nsteps*rtol

    # vector version
    (tnV,ynV,dynV)=DASSL.dasslSolve(F,[sol(0.0)], tspan, maxorder = order)

    @test  vcat(ynV...)==yn
    @test vcat(dynV...)==dyn

    # analytical jacobian version (vector)
    (tna,yna,dyna)=DASSL.dasslSolve(F, [sol(0.0)], tspan, maxorder = order, Fy = Fy, Fdy = Fdy)
    aerror = maximum(abs(map(first,yn)-sol(tn)))
    rerror = maximum(abs(map(first,yn)-sol(tn))/abs(sol(tn)))
    nsteps = length(tn)

    @test aerror < nsteps*atol
    @test rerror < nsteps*rtol
end

# test the initial derivatives y0 using the van der Pol equation

# van der Pol equation
Fvdp(t,y,dy)=([dy[1]+y[2],
               eps*dy[2]-y[1]+(y[2]^3/3-y[2])])

eps = 1e-8
y0  = [ 0.0,       1]
dy0 = [-1.0, 2/3/eps]

# this should result in a warning and fail to start the intergration
# (tn,yn)=DASSL.dasslSolve(Fvdp, y0, tspan, rtol = rtol, atol = atol, h0 = 1.0e-4)
# @test length(tn)==1

# it should work after specifying the initial derivative
(tn,yn)=DASSL.dasslSolve(Fvdp, y0, tspan, dy0 = dy0)
@test length(tn)>1
