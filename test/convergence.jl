using Winston
using dassl

# test the convergence of dasslSolve.  dasslTestConvergence returns a
# tuple of relative and absolute L^Inf norms of a difference of
# analytic and numerical solutions.  The third element in the returned
# touple is the time it took to obtain the numerical solution.
function dasslTestConvergence{T<:Number}(F          :: Function,  # equation to solve
                                         y0         :: Vector{T}, # initial data
                                         tspan      :: Vector{T}, # time span of a solution
                                         sol        :: Function,  # analytic solution, for comparison with numerical solution
                                         rtol_range :: Vector{T}, # vector of relative tolerances
                                         atol_range :: Vector{T}) # vector of absolute tolerances

    if length(rtol_range) != length(atol_range)
        error("The table of relative errors and absolute errors should be of the same size.")
    end

    n = length(rtol_range)

    norms_abs = zeros(T,n)
    norms_rel = zeros(T,n)
    times     = zeros(T,n)

    for i = 1:n
        times[i] = @elapsed (tn,yn)=dasslSolve(F,y0,tspan,
                                               rtol = rtol_range[i],
                                               atol = atol_range[i])
        k = length(tn)
        delta_rel = zeros(T,k)
        delta_abs = zeros(T,k)

        for j = 1:k
            t = tn[j]
            delta_abs[j] = norm(sol(t)-yn[:,j], Inf)
            delta_rel[j] = norm(sol(t)-yn[:,j], Inf)/norm(sol(t), Inf)
        end

        norms_abs[i]=norm(delta_abs,Inf)
        norms_rel[i]=norm(delta_rel,Inf)

    end

    return norms_rel, norms_abs, times
end


function dasslDrawConvergence{T<:Number}(F     :: Function,  # equation to solve
                                         y0    :: Vector{T}, # initial data
                                         tspan :: Vector{T}, # time span of a solution
                                         sol   :: Function,  # analytic solution, for comparison with numerical solution
                                         rtol  :: Vector{T}, # vector of relative tolerances
                                         atol  :: Vector{T}) # vector of absolute tolerances

    # run the dasslSolve to trigger the compilation
    dasslSolve(F,y0,tspan)

    # run the convergence tests
    (rerr,aerr,time) = dasslTestConvergence(F,y0,tspan,sol,rtol,atol)

    tbl = Table(1,3)

    tbl[1,1] = FramedPlot(title  = "Absolute error",
                          xlabel = "Absolute tolerance",
                          ylabel = "Absolute error",
                          logx   = true,
                          logy   = true)
    tbl[1,2] = FramedPlot(title  = "Relative error",
                          xlabel = "Relative tolerance",
                          ylabel = "Relative error",
                          logx   = true,
                          logy   = true)
    tbl[1,3] = FramedPlot(title  = "Execution time",
                          xlabel = "Relative error",
                          ylabel = "Elapsed time",
                          logx   = true,
                          logy   = true)

    add(tbl[1,1], Points(atol,aerr))
    add(tbl[1,2], Points(rtol,rerr))
    add(tbl[1,3], Points(rerr,time))

    return tbl

end
