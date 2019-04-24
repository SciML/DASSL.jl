abstract type DASSLDAEAlgorithm <: DiffEqBase.AbstractDAEAlgorithm end
struct dassl <: DASSLDAEAlgorithm
  maxorder
  factorize_jacobian
end

dassl(;maxorder = 6,factorize_jacobian = true) = dassl(maxorder,factorize_jacobian)

function solve(
    prob::DiffEqBase.AbstractDAEProblem{uType,duType,tupType,isinplace},
    alg::DASSLDAEAlgorithm,args...;timeseries_errors=true,
    abstol=1e-5,reltol=1e-3,dt = 1e-4, dtmin = 0.0, dtmax = Inf,
    callback=nothing,kwargs...) where {uType,duType,tupType,isinplace}

    tType = eltype(tupType)

    if callback != nothing || prob.callback != nothing
        error("DASSL is not compatible with callbacks.")
    end

    tspan = [prob.tspan[1],prob.tspan[2]]

    #sizeu = size(prob.u0)
    #sizedu = size(prob.du0)
    p = prob.p

    ### Fix inplace functions to the non-inplace version
    if isinplace
      f = (t,u,du) -> (out = similar(u); prob.f(out,du,u,p,t); out)
    else
      f = (t,u) -> prob.f(u,p,t)
    end

    ### Finishing Routine

    ts,timeseries,dus = dasslSolve(f,prob.u0,tspan,
                                abstol=abstol,
                                reltol=reltol,
                                maxstep=dtmax,
                                minstep=dtmin,
                                initstep=dt,
                                dy0 = prob.du0,
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
    DiffEqBase.build_solution(prob,alg,ts,timeseries,du=dus,
                      timeseries_errors = timeseries_errors)
end
