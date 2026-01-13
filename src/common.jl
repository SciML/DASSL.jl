using SciMLBase: AbstractDAEAlgorithm, AbstractDAEProblem, build_solution, isinplace

abstract type DASSLDAEAlgorithm <: AbstractDAEAlgorithm end

"""
    dassl(; maxorder = 6, factorize_jacobian = true)

Construct a DASSL algorithm for solving differential-algebraic equations (DAEs).

DASSL is an implementation of the DASSL (Differential Algebraic System Solver) algorithm
which uses backward differentiation formulas (BDF) to solve systems of the form
`F(t,y,y')=0`.

# Keyword Arguments
- `maxorder::Integer = 6`: Maximum order of the BDF method. Valid values are 1-6.
  BDF methods are stable up to 6th order.
- `factorize_jacobian::Bool = true`: Whether to store the factorized Jacobian matrix.
  This dramatically increases performance for large systems but may decrease performance
  for small systems.

# Example
```julia
using DASSL
prob = DAEProblem(f, du0, u0, tspan)
sol = solve(prob, dassl())
```

See also: [`dasslSolve`](@ref), [`dasslIterator`](@ref)
"""
struct dassl <: DASSLDAEAlgorithm
    maxorder::Any
    factorize_jacobian::Any
end

dassl(; maxorder = 6, factorize_jacobian = true) = dassl(maxorder, factorize_jacobian)

function solve(
        prob::AbstractDAEProblem{uType, duType, tupType, isinplace},
        alg::DASSLDAEAlgorithm, args...; timeseries_errors = true,
        abstol = 1.0e-5, reltol = 1.0e-3, dt = 1.0e-4, dtmin = 0.0, dtmax = Inf,
        callback = nothing, kwargs...
    ) where {uType, duType, tupType, isinplace}
    tType = eltype(tupType)

    if callback != nothing || :callback in keys(prob.kwargs)
        error("DASSL is not compatible with callbacks.")
    end

    tspan = [prob.tspan[1], prob.tspan[2]]
    p = prob.p

    if isinplace
        # In-place path: use pre-allocated cache for zero-allocation inner loop
        cache = alg_cache(alg, prob.u0, p, tspan[1], Val(true))

        # In-place function wrapper (no allocation per call!)
        F! = (out, t, u, du) -> prob.f(out, du, u, p, t)

        ts, timeseries, dus = dasslSolve!(
            cache, F!, prob.u0, tspan,
            abstol = abstol,
            reltol = reltol,
            maxstep = dtmax,
            minstep = dtmin,
            initstep = dt,
            dy0 = prob.du0,
            maxorder = alg.maxorder,
            factorize_jacobian = alg.factorize_jacobian
        )
    else
        # Out-of-place path (unchanged, backward compatible)
        f = (t, u) -> prob.f(u, p, t)

        ts, timeseries, dus = dasslSolve(
            f, prob.u0, tspan,
            abstol = abstol,
            reltol = reltol,
            maxstep = dtmax,
            minstep = dtmin,
            initstep = dt,
            dy0 = prob.du0,
            maxorder = alg.maxorder,
            factorize_jacobian = alg.factorize_jacobian
        )
    end

    return build_solution(
        prob, alg, ts, timeseries, du = dus,
        timeseries_errors = timeseries_errors
    )
end
