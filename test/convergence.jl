using Winston
using DASSL

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
                          xlog   = true,
                          ylog   = true)
    tbl[1,2] = FramedPlot(title  = "Relative error",
                          xlabel = "Relative tolerance",
                          ylabel = "Relative error",
                          xlog   = true,
                          ylog   = true)
    tbl[1,3] = FramedPlot(title  = "Execution time",
                          xlabel = "Relative error",
                          ylabel = "Elapsed time",
                          xlog   = true,
                          ylog   = true)

    add(tbl[1,1], Points(atol,aerr))
    add(tbl[1,2], Points(rtol,rerr))
    add(tbl[1,3], Points(rerr,time))

    setattr(tbl, aspect_ratio=0.3)

    return tbl

end


#------------------------------------------------------------
# index-1 example
#------------------------------------------------------------

function F1{T<:Number}(t::T,y::Array{T,1},dy::Array{T,1})
    a=10.0
    [-dy[1]+(a-1/(2-t))*y[1]+(2-t)*a*y[3]+exp(t)*(3-t)/(2-t),
     -dy[2]+(1-a)/(t-2)*y[1]-y[2]+(a-1)*y[3]+2*exp(t),
     (t+2)*y[1]+(t^2-4)*y[2]-(t^2+t-2)*exp(t)]
end

function sol1{T<:Number}(t::T)
    a=10.0
    [exp(t),
     exp(t),
     -exp(t)/(2-t)]
end

const y1     = sol1(0.0)
const tspan1 = [0.0, 1.0]

const rtol=10.0.^-[1.0:0.1:5.0]
const atol=0.01*rtol

tbl=dasslDrawConvergence(F1,y1,tspan1,sol1,rtol,atol)
display(tbl)
