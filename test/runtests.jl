using DASSL
using Base.Test

F(t,y,dy)=(dy+y)
sol(t)=exp(-t)

const y0=[sol(0.0)]
const tspan=[0.0, 10.0]

(tn,yn)=dassl.dasslSolve(F,y0,tspan, rtol = 1.0e-3, atol = 1.0e-5, h0 = 1.0e-4)

@test abs(yn[1,end]-sol(tn[end]))/sol(tn[end]) < 0.002
