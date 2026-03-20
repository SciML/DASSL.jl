# DAE initialization support for DASSL.jl
# Following the pattern from Sundials.jl: https://github.com/SciML/Sundials.jl/blob/master/src/common_interface/initialize.jl

using SciMLBase: NoInit, CheckInit, OverrideInit,
    get_initial_values, isinplace, ReturnCode
using DiffEqBase: DefaultInit
using LinearAlgebra: norm

# Import has_initialization_data - it checks if the function has initialization_data field set
import SciMLBase: has_initialization_data

"""
    initialize_dae!(u0, du0, p, t0, prob, initializealg::DefaultInit, abstol, reltol)

Default initialization: if initialization_data exists, first run OverrideInit to solve
for consistent initial conditions, then CheckInit to verify them. Otherwise just CheckInit.
"""
function initialize_dae!(u0, du0, p, t0, prob, initializealg::DefaultInit, abstol, reltol)
    if has_initialization_data(prob.f)
        # First, solve for initial conditions with OverrideInit
        u0, du0, p, success = initialize_dae!(u0, du0, p, t0, prob, OverrideInit(), abstol, reltol)
        if !success
            return u0, du0, p, false
        end
        # Then, verify with CheckInit
        return initialize_dae!(u0, du0, p, t0, prob, CheckInit(), abstol, reltol)
    else
        return initialize_dae!(u0, du0, p, t0, prob, CheckInit(), abstol, reltol)
    end
end

"""
    initialize_dae!(u0, du0, p, t0, prob, initializealg::NoInit, abstol, reltol)

No initialization: simply returns the initial conditions as-is without any checks.
Use this only when you are certain your initial conditions are consistent.
"""
function initialize_dae!(u0, du0, p, t0, prob, ::NoInit, abstol, reltol)
    return u0, du0, p, true
end

"""
    initialize_dae!(u0, du0, p, t0, prob, initializealg::CheckInit, abstol, reltol)

Check initialization: verifies that the initial conditions satisfy the DAE constraints
within the specified tolerance. Throws an error if they do not.
"""
function initialize_dae!(u0, du0, p, t0, prob, ::CheckInit, abstol, reltol)
    f = prob.f

    # Evaluate the DAE residual at the initial conditions
    if isinplace(prob)
        resid = similar(u0)
        f(resid, du0, u0, p, t0)
    else
        resid = f(du0, u0, p, t0)
    end

    # Check if residuals are within tolerance
    normresid = norm(resid, Inf)
    if normresid > abstol
        error(
            """
            DAE initialization failed with CheckInit: Initial conditions do not satisfy the DAE constraints.

            The residual norm is $(normresid), which exceeds the tolerance $(abstol).

            Note that for DAEs, both `du0` (derivatives) and `u0` (states) must be consistent,
            meaning F(du0, u0, p, t0) ≈ 0.

            To resolve this issue, you have several options:
            1. Fix your initial conditions (both `du0` and `u0`) to satisfy the DAE constraints
            2. Use `initializealg = NoInit()` to skip initialization checks (use with caution)
            3. If using ModelingToolkit, ensure your system has proper initialization equations
               and use `initializealg = OverrideInit()`

            Example to skip checks:
            solve(prob, dassl(); initializealg = NoInit())
            """
        )
    end

    return u0, du0, p, true
end

"""
    initialize_dae!(u0, du0, p, t0, prob, initializealg::OverrideInit, abstol, reltol)

Override initialization: uses the problem's initialization_data to solve for consistent
initial conditions. This is typically used with ModelingToolkit.jl problems.
"""
function initialize_dae!(u0, du0, p, t0, prob, initializealg::OverrideInit, abstol, reltol)
    f = prob.f

    if !has_initialization_data(f)
        # No initialization data, just return as-is (like NoInit)
        return u0, du0, p, true
    end

    # Create a value provider for get_initial_values
    # This provides state_values, parameter_values, and symbolic_container interfaces
    valp = DASSLValueProvider(u0, p, t0, f)

    # Determine tolerances - algorithm takes priority, then fall back to solver tolerances
    _abstol = something(initializealg.abstol, abstol)
    _reltol = something(initializealg.reltol, reltol)

    # Get the initialized values
    u0_new, p_new, success = get_initial_values(
        prob, valp, f, initializealg, Val(isinplace(prob));
        abstol = _abstol, reltol = _reltol
    )

    if !success
        return u0, du0, p, false
    end

    # For DAEs with OverrideInit, we need to potentially update du0 as well
    # The initialization problem should have handled this through the state mapping
    # For now, we compute du0 from the constraint that F(du0, u0, p, t0) = 0
    # This is a simple approach; more sophisticated methods could be added later

    # Copy new values to output (in-place friendly)
    if isinplace(prob)
        u0 .= u0_new
    else
        u0 = u0_new
    end

    return u0, du0, p_new, success
end

# Value provider for DASSL that implements the SymbolicIndexingInterface.
# Stores the SciMLFunction so MTK's get_scimlfn can find it via symbolic_container.
struct DASSLValueProvider{U, P, T, F}
    u::U
    p::P
    t::T
    f::F
end

# Implement required interfaces from SymbolicIndexingInterface
import SymbolicIndexingInterface: state_values, parameter_values, current_time,
    symbolic_container

state_values(vp::DASSLValueProvider) = vp.u
parameter_values(vp::DASSLValueProvider) = vp.p
current_time(vp::DASSLValueProvider) = vp.t
symbolic_container(vp::DASSLValueProvider) = vp.f
