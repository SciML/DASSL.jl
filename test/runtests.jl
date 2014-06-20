using DASSL
using Base.Test

F(t,y,dy)=(dy+y)
sol(t)=exp(-t)
const tspan=[0.0, 10.0]

atol = 1.0e-5
rtol = 1.0e-3


for order=1:6
    # scalar version
    (tn,yn)=DASSL.dasslSolve(F, sol(0.0), tspan, rtol = rtol, atol = atol, h0 = 1.0e-4, maxorder = order)
    aerror = maximum(abs(yn-sol(tn)))
    rerror = maximum(abs(yn-sol(tn))/abs(sol(tn)))

    @test aerror < 10*atol
    @test rerror < 10*rtol

    # vector version
    (tnV,ynV)=DASSL.dasslSolve(F,[sol(0.0)],tspan, rtol = rtol, atol = atol, h0 = 1.0e-4, maxorder = order)

    @test vcat(ynV...)==yn

end
