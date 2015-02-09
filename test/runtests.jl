using DASSL
using Base.Test
using FactCheck

facts("Testing maxorder") do

    F(t,y,dy)=(dy+y.^2)
    Fy(t,y,dy)=diagm(2y)
    Fdy(t,y,dy)=eye(length(y))

    sol(t)=1./(1 .+ t)
    tspan=[0.0, 1.0]

    atol = 1.0e-5
    rtol = 1.0e-3


    # test the maxorder option
    for order=1:6
        # scalar version
        (tn,yn,dyn)=DASSL.dasslSolve(F, sol(0.0), tspan, maxorder = order)
        aerror = maximum(abs(yn-sol(tn)))
        rerror = maximum(abs(yn-sol(tn))/abs(sol(tn)))
        nsteps = length(tn)

        @fact aerror => less_than(2*nsteps*atol)
        @fact rerror => less_than(2*nsteps*rtol)

        # vector version
        (tnV,ynV,dynV)=DASSL.dasslSolve(F,[sol(0.0)], tspan, maxorder = order)

        @fact  vcat(ynV...) => yn
        @fact vcat(dynV...) => dyn

        # analytical jacobian version (vector)
        (tna,yna,dyna)=dasslSolve(F, [sol(0.0)], tspan, maxorder = order, Fy = Fy, Fdy = Fdy)
        aerror = maximum(abs(map(first,yn)-sol(tn)))
        rerror = maximum(abs(map(first,yn)-sol(tn))/abs(sol(tn)))
        nsteps = length(tn)

        @fact aerror => less_than(2*nsteps*atol)
        @fact rerror => less_than(2*nsteps*rtol)
    end
end

facts("Testing minimal error tolerances") do
    eps=1e-6
    # van der Pol equation
    Fvdp(t,y,dy)=([dy[1]+y[2],
                   eps*dy[2]-y[1]+y[2]*(1.0-y[1]^2)])
    y0 = [2., 0]
    tn = (0.,[2.,0.],[0.,-2/eps])

    tol = 1e0
    while true
        tol/=10
        sol=dasslIterator(Fvdp, y0, 1.0; reltol=tol, abstol=tol)

        try
            (tn,yn,dyn)=consume(sol)
        catch
            break
        end

    end
    @fact tol => roughly(1e-19)
end
