abstract DASSLDAEAlgorithm <: AbstractODEAlgorithm
immutable dassl <: DASSLDAEAlgorithm
  maxorder
  factorize_jacobian
end

dassl(;maxorder = 6,factorize_jacobian = true) = dassl(maxorder,factorize_jacobian)

function solve{uType,duType,tType,isinplace,F}(
    prob::AbstractDAEProblem{uType,duType,tType,isinplace,F},
    alg::DASSLDAEAlgorithm,args...;timeseries_errors=true,
    abstol=1e-5,reltol=1e-3,dt = 1e-4, dtmin = 0.0, dtmax = Inf,kwargs...)

    tspan = [prob.tspan[1],prob.tspan[2]]

    #sizeu = size(prob.u0)
    #sizedu = size(prob.du0)

    ### Fix inplace functions to the non-inplace version
    if isinplace
      f = (t,u,du) -> (out = similar(u); prob.f(t,u,du,out); out)
    else
      f = prob.f
    end

    ### Finishing Routine

    ts,timeseries = dasslSolve(f,prob.u0,tspan,
                                abstol=abstol,
                                reltol=reltol,
                                maxstep=dtmax,
                                minstep=dtmin,
                                initstep=dt,
                                maxorder=alg.maxorder,
                                factorize_jacobian=alg.factorize_jacobian)
    #=
    timeseries = Vector{uType}(0)
    if typeof(prob.u0)<:Number
        for i=1:length(ures)
            push!(timeseries,ures[i][1])
        end
    else
        for i=1:length(ures)
            push!(timeseries,reshape(ures[i],sizeu))
        end
    end
    =#

    build_solution(prob,alg,ts,timeseries,
                      timeseries_errors = timeseries_errors)
end
