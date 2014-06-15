using DASSL
using Base.Test

F(t,y,dy)=(dy+y)
sol(t)=exp(-t)
const tspan=[0.0, 10.0]

# vector version
(tn,yn)=DASSL.dasslSolve(F,[sol(0.0)],tspan, rtol = 1.0e-3, atol = 1.0e-5, h0 = 1.0e-4)
@test norm(yn[end]-[sol(tn[end])])/sol(tn[end]) < 0.002

# scalar version
(tn,yn)=DASSL.dasslSolve(F,sol(0.0),tspan, rtol = 1.0e-3, atol = 1.0e-5, h0 = 1.0e-4)
@test norm(yn[end]-sol(tn[end]))/sol(tn[end]) < 0.002
