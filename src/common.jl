using SciMLBase: AbstractDAEAlgorithm, AbstractDAEProblem, build_solution, isinplace,
    ReturnCode, remake

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
        callback = nothing, initializealg = DefaultInit(), kwargs...
    ) where {uType, duType, tupType, isinplace}
    tType = eltype(tupType)

    if callback != nothing || :callback in keys(prob.kwargs)
        error("DASSL is not compatible with callbacks.")
    end

    tspan = [prob.tspan[1], prob.tspan[2]]
    t0 = prob.tspan[1]

    # Make copies of initial conditions for potential modification
    u0 = copy(prob.u0)
    du0 = copy(prob.du0)
    p = prob.p

    # Run DAE initialization
    u0, du0, p, init_success = initialize_dae!(u0, du0, p, t0, prob, initializealg, abstol, reltol)

    # Remake problem with updated initial conditions and parameters
    # so that build_solution stores the correct values (e.g., solved parameters)
    if p !== prob.p || u0 !== prob.u0
        prob = remake(prob; u0 = u0, du0 = du0, p = p)
    end

    if !init_success
        # Return a solution with InitialFailure retcode
        return build_solution(
            prob, alg, eltype(tspan)[], typeof(u0)[],
            du = typeof(du0)[],
            retcode = ReturnCode.InitialFailure,
            timeseries_errors = timeseries_errors
        )
    end

    if isinplace
        # In-place path: use pre-allocated cache for zero-allocation inner loop
        cache = alg_cache(alg, u0, p, t0, Val(true))

        # In-place function wrapper (no allocation per call!)
        F! = (out, t, u, du) -> prob.f(out, du, u, p, t)

        ts, timeseries, dus = dasslSolve!(
            cache, F!, u0, tspan,
            abstol = abstol,
            reltol = reltol,
            maxstep = dtmax,
            minstep = dtmin,
            initstep = dt,
            dy0 = du0,
            maxorder = alg.maxorder,
            factorize_jacobian = alg.factorize_jacobian
        )
    else
        # Out-of-place path (unchanged, backward compatible)
        f = (t, u) -> prob.f(u, p, t)

        ts, timeseries, dus = dasslSolve(
            f, u0, tspan,
            abstol = abstol,
            reltol = reltol,
            maxstep = dtmax,
            minstep = dtmin,
            initstep = dt,
            dy0 = du0,
            maxorder = alg.maxorder,
            factorize_jacobian = alg.factorize_jacobian
        )
    end

    return build_solution(
        prob, alg, ts, timeseries, du = dus,
        retcode = ReturnCode.Success,
        timeseries_errors = timeseries_errors
    )
end
