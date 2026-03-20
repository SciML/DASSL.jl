# In-place implementations of DASSL core functions
# These operate on pre-allocated cache buffers for zero-allocation integration

using LinearAlgebra: ldiv!

# ============================================================================
# In-place interpolation functions
# ============================================================================

"""
    interpolateAt!(p_out, x, y, x0)

Compute Lagrange interpolation polynomial value at x0, writing to p_out.
`y` is a vector of vectors (or references to vectors).
"""
function interpolateAt!(
        p_out::AbstractVector,
        x::AbstractVector{T},
        y::AbstractVector,
        x0::T
    ) where {T <: Real}
    n = length(x)
    fill!(p_out, zero(eltype(p_out)))

    return @inbounds for i in 1:n
        Li = one(T)
        for j in 1:n
            if j != i
                Li *= (x0 - x[j]) / (x[i] - x[j])
            end
        end
        # p_out .+= Li .* y[i]
        for k in eachindex(p_out)
            p_out[k] += Li * y[i][k]
        end
    end
end

"""
    interpolateDerivativeAt!(p_out, x, y, x0)

Compute derivative of Lagrange interpolation polynomial at x0, writing to p_out.
"""
function interpolateDerivativeAt!(
        p_out::AbstractVector,
        x::AbstractVector{T},
        y::AbstractVector,
        x0::T
    ) where {T <: Real}
    n = length(x)
    fill!(p_out, zero(eltype(p_out)))

    return @inbounds for i in 1:n
        dLi = zero(T)
        for k in 1:n
            if k != i
                dLi1 = one(T)
                for j in 1:n
                    if j != k && j != i
                        dLi1 *= (x0 - x[j]) / (x[i] - x[j])
                    end
                end
                dLi += dLi1 / (x[i] - x[k])
            end
        end
        for m in eachindex(p_out)
            p_out[m] += dLi * y[i][m]
        end
    end
end

"""
    interpolateHighestDerivative!(p_out, x, y)

Compute highest derivative of interpolation polynomial, writing to p_out.
"""
function interpolateHighestDerivative!(
        p_out::AbstractVector,
        x::AbstractVector,
        y::AbstractVector
    )
    n = length(x)
    fill!(p_out, zero(eltype(p_out)))

    return @inbounds for i in 1:n
        Li = one(eltype(x))
        for j in 1:n
            if j != i
                Li *= 1 / (x[i] - x[j])
            end
        end
        fact_val = factorial(n - 1) * Li
        for k in eachindex(p_out)
            p_out[k] += fact_val * y[i][k]
        end
    end
end

# ============================================================================
# In-place Newton iteration
# ============================================================================

"""
    newton_iteration!(cache, f!, y0, normy) -> status

In-place Newton iteration that writes result to cache.yn.
Uses cache.delta for working storage.

Returns status: 0 = converged, -1 = failed to converge.
"""
function newton_iteration!(
        cache::DASSLCache,
        f!,  # f!(out, y) computes the Newton step
        y0::AbstractVector{T},
        normy
    ) where {T <: Number}

    yn = cache.yn
    delta = cache.delta

    # First iteration: delta = f(y0)
    f!(delta, y0)
    norm1 = normy(delta)

    # yn = y0 + delta
    @. yn = y0 + delta

    # Check for immediate convergence
    ep = eps(eltype(abs.(y0)))
    if norm1 < 100 * ep * normy(y0)
        return 0  # Converged immediately
    end

    # Subsequent iterations (max MAXIT = 10)
    for i in 1:MAXIT
        f!(delta, yn)
        normn = normy(delta)
        rho = (normn / norm1)^(1 / i)

        # yn = yn + delta
        @. yn = yn + delta

        # Iteration failed to converge
        if rho > 9 / 10
            copyto!(yn, y0)
            return -1
        end

        err = rho / (1 - rho) * normn

        # Iteration converged successfully
        if err < 1 / 3
            return 0
        end
    end

    # Unable to converge after MAXIT iterations
    copyto!(yn, y0)
    return -1
end

# ============================================================================
# In-place numerical Jacobian computation
# ============================================================================

"""
    numerical_jacobian!(jac_out, cache, F!, t, y, dy, a, wt)

Compute numerical Jacobian in-place using finite differences.
Fills jac_out with d(F)/dy + a * d(F)/d(dy).

Uses cache.f_plus and cache.f_base for temporary storage.
"""
function numerical_jacobian!(
        jac_out::AbstractMatrix,
        cache::DASSLCache,
        F!,  # F!(out, t, y, dy)
        t, y, dy, a, wt
    )
    n = length(y)
    ep = eps(one(t))
    h = 1 / a

    # Compute base residual: F(t, y, dy)
    F!(cache.f_base, t, y, dy)

    # Compute each column via finite differences
    return @inbounds for i in 1:n
        # Compute perturbation delta for this column
        delta_i = max(abs(y[i]), abs(h * dy[i]), wt[i]) * sqrt(ep)

        # Perturb y[i] and corresponding dy[i]
        # For the Newton function f(y) = F(t, y, a*y + b) where b = dy - a*y
        # perturbing y by delta gives dy_perturbed = a*(y + delta) + b = dy + a*delta
        y_orig = y[i]
        dy_orig = dy[i]

        # Use cache.ytmp as perturbed y (copy y, modify element i)
        copyto!(cache.ytmp, y)
        cache.ytmp[i] = y_orig + delta_i

        # Use cache.ytmp2 as perturbed dy
        copyto!(cache.ytmp2, dy)
        cache.ytmp2[i] = dy_orig + a * delta_i

        F!(cache.f_plus, t, cache.ytmp, cache.ytmp2)

        # jac[:, i] = (f_plus - f_base) / delta_i
        for j in 1:n
            jac_out[j, i] = (cache.f_plus[j] - cache.f_base[j]) / delta_i
        end
    end
end

# ============================================================================
# In-place corrector
# ============================================================================

"""
    corrector!(cache, a_new, jac_new!, y0, f_newton!, normy) -> status

In-place corrector that updates cache.yn and cache.jac.
Returns status: 0 = converged, -1 = failed.

- `jac_new!`: function `jac_new!(jac_out)` that computes Jacobian in-place
- `f_newton!`: function `f_newton!(out, y)` that computes Newton residual
"""
function corrector!(
        cache::DASSLCache,
        a_new::T,
        jac_new!,
        y0::AbstractVector,
        f_newton!,
        normy
    ) where {T}

    a_old = cache.a

    # Determine if we need a new Jacobian
    need_new_jac = (a_old == 0) || abs((a_old - a_new) / (a_old + a_new)) > 1 / 4

    if need_new_jac
        # Compute fresh Jacobian
        jac_new!(cache.jac)
        cache.jac_factorized = lu(cache.jac; check = false)
        cache.a = a_new

        # Newton iteration with fresh Jacobian
        function f_newton_solve_fresh!(out, y)
            f_newton!(cache.residual, y)
            ldiv!(out, cache.jac_factorized, cache.residual)
            return @. out = -out
        end

        status = newton_iteration!(cache, f_newton_solve_fresh!, y0, normy)
    else
        # Reuse old Jacobian with scaling factor
        c = 2 * a_old / (a_new + a_old)

        function f_newton_solve_reuse!(out, y)
            f_newton!(cache.residual, y)
            ldiv!(out, cache.jac_factorized, cache.residual)
            return @. out = -c * out
        end

        status = newton_iteration!(cache, f_newton_solve_reuse!, y0, normy)

        if status < 0
            # Retry with fresh Jacobian
            jac_new!(cache.jac)
            cache.jac_factorized = lu(cache.jac; check = false)
            cache.a = a_new

            function f_newton_solve_new!(out, y)
                f_newton!(cache.residual, y)
                ldiv!(out, cache.jac_factorized, cache.residual)
                return @. out = -out
            end

            status = newton_iteration!(cache, f_newton_solve_new!, y0, normy)
        end
    end

    return status
end

# ============================================================================
# In-place stepper
# ============================================================================

"""
    stepper!(cache, ord, h_next, F!, normy, maxorder) -> (status, err)

In-place stepper that updates cache with new y, dy values.
Results are written to cache.yn and cache.dyn.

Returns (status, err) where status < 0 means Newton failed.
"""
function stepper!(
        cache::DASSLCache,
        ord::Integer,
        h_next::Real,
        F!,  # F!(out, t, y, dy) in-place residual
        normy,
        maxorder::Integer
    )
    # Get history for interpolation (zero-allocation via work buffers)
    tk = get_history_t!(cache, ord)
    yk = get_history_y!(cache, ord)

    t_next = tk[end] + h_next

    # Compute alpha sum
    alphas = zero(eltype(cache.t_hist))
    @inbounds for j in 1:ord
        alphas -= 1 / j
    end

    if cache.hist_len == 1
        # First step: use initial conditions
        copyto!(cache.dy0, get_latest_dy(cache))
        # y0 = y[1] + h_next * dy[1]
        y_last = get_latest_y(cache)
        @. cache.y0 = y_last + h_next * cache.dy0
        a = 2 / h_next
    else
        # Use predictor (in-place interpolation)
        interpolateDerivativeAt!(cache.dy0, tk, yk, t_next)
        interpolateAt!(cache.y0, tk, yk, t_next)
        a = -alphas / h_next
    end

    # Newton function closure (computes residual in-place)
    # f_newton(yc) = F(t_next, yc, a * yc + b) where b = dy0 - a * y0
    function f_newton!(out, yc)
        # Compute dy_c = a * yc + dy0 - a * y0 into cache.ytmp
        @. cache.ytmp = a * yc + cache.dy0 - a * cache.y0
        return F!(out, t_next, yc, cache.ytmp)
    end

    # Jacobian computation closure
    function jac_new!(jac_out)
        return numerical_jacobian!(jac_out, cache, F!, t_next, cache.y0, cache.dy0, a, cache.wt)
    end

    # Run corrector (writes to cache.yn)
    status = corrector!(cache, a, jac_new!, cache.y0, f_newton!, normy)

    # Compute dyn = a * yn + dy0 - a * y0
    @. cache.dyn = a * cache.yn + cache.dy0 - a * cache.y0

    # Compute error estimate
    alpha0 = zero(eltype(cache.t_hist))
    @inbounds for i in 1:ord
        t_i = get_t_at(cache, cache.hist_len - i + 1)
        alpha_i = h_next / (t_next - t_i)
        alpha0 -= alpha_i
    end

    # Get t0 for alpha_{ord+1}
    if cache.hist_len >= ord + 1
        t0 = get_t_at(cache, cache.hist_len - ord)
    elseif cache.hist_len >= 2
        h1 = get_t_at(cache, 2) - get_t_at(cache, 1)
        t0 = get_t_at(cache, 1) - h1
    else
        t0 = get_t_at(cache, 1) - h_next
    end

    alpha_ord_plus_1 = h_next / (t_next - t0)
    M = max(alpha_ord_plus_1, abs(alpha_ord_plus_1 + alphas - alpha0))

    # err = normy(yn - y0) * M (use delta as temp)
    @. cache.delta = cache.yn - cache.y0
    err = normy(cache.delta) * M

    return (status, err)
end

# ============================================================================
# In-place error estimation for order selection
# ============================================================================

"""
    errorEstimates!(errors, cache, normy, k) -> errors

Compute error estimates for orders k-2, k-1, k, and possibly k+1.
Writes to the provided errors vector.
"""
function errorEstimates!(
        errors::AbstractVector,
        cache::DASSLCache,
        normy,
        k::Integer
    )
    nsteps = cache.hist_len
    h = get_t_at(cache, nsteps) - get_t_at(cache, nsteps - 1)

    # Get views for interpolation (zero-allocation via work buffers)
    for order in max(k - 2, 1):k
        t_view = get_history_t!(cache, order + 2)
        y_view = get_history_y!(cache, order + 2)
        interpolateHighestDerivative!(cache.ytmp, t_view, y_view)
        errors[order] = h^(order + 1) * normy(cache.ytmp)
    end

    # Compute k+1 order estimate if steps are equal
    if nsteps >= k + 3
        t_view = get_history_t!(cache, k + 3)
        if _all_steps_equal(t_view)
            y_view = get_history_y!(cache, k + 3)
            interpolateHighestDerivative!(cache.ytmp, t_view, y_view)
            if length(errors) > k
                errors[k + 1] = h^(k + 2) * normy(cache.ytmp)
            end
        end
    end

    return errors
end

"""
    errorEstimates_cache!(cache, normy, k) -> ne

Compute error estimates for orders k-2 through k (and possibly k+1) using
pre-allocated cache.errors_work buffer. Returns the number of valid estimates.
"""
function errorEstimates_cache!(
        cache::DASSLCache,
        normy,
        k::Integer
    )
    nsteps = cache.hist_len
    h = get_t_at(cache, nsteps) - get_t_at(cache, nsteps - 1)

    errors = cache.errors_work
    @inbounds for i in 1:k
        errors[i] = zero(eltype(errors))
    end

    @inbounds for order in max(k - 2, 1):k
        t_view = get_history_t!(cache, order + 2)
        y_view = get_history_y!(cache, order + 2)
        interpolateHighestDerivative!(cache.ytmp, t_view, y_view)
        errors[order] = h^(order + 1) * normy(cache.ytmp)
    end

    # Estimate k+1 order if enough equal-sized steps
    ne = k
    if nsteps >= k + 3
        t_view = get_history_t!(cache, k + 3)
        if _all_steps_equal(t_view)
            y_view = get_history_y!(cache, k + 3)
            interpolateHighestDerivative!(cache.ytmp, t_view, y_view)
            errors[k + 1] = h^(k + 2) * normy(cache.ytmp)
            ne = k + 1
        end
    end

    return ne
end

"""
    newStepOrderContinuous_cache!(cache, normy, err, k, maxorder)

In-place version of newStepOrderContinuous using pre-allocated cache buffers.
Zero-allocation order/step-size selection.
"""
function newStepOrderContinuous_cache!(
        cache::DASSLCache,
        normy,
        err,
        k::Integer,
        maxorder::Integer
    )
    ne = errorEstimates_cache!(cache, normy, k)
    errors = cache.errors_work
    errors[k] = err

    lo = max(k - 2, 1)
    hi = min(ne, maxorder)
    errors_dec = true
    errors_inc = true
    @inbounds for i in lo:(hi - 1)
        if errors[i + 1] >= errors[i]
            errors_dec = false
        end
        if errors[i + 1] <= errors[i]
            errors_inc = false
        end
    end

    if ne == k + 1 && errors_dec
        order = min(k + 1, maxorder)
    elseif ne > 1 && errors_inc
        order = max(k - 1, 1)
    else
        order = k
    end

    est = errors[order]
    r = (2 * est + 1 / 10000)^(-1 / (order + 1))

    return r, order
end

# ============================================================================
# Main in-place solve function
# ============================================================================

"""
    dasslSolve!(cache, F!, y0, tspan; kwargs...)

In-place version of dasslSolve that uses pre-allocated cache.
The inner integration loop operates on cache buffers with minimal allocations.

# Arguments
- `cache`: Pre-allocated DASSLCache
- `F!`: In-place residual function `F!(out, t, y, dy)`
- `y0`: Initial value vector
- `tspan`: Time span [t0, tf]

# Returns
- `(tout, yout, dyout)`: Tuple of time points, solutions, and derivatives
"""
function dasslSolve!(
        cache::DASSLCache,
        F!,  # F!(out, t, y, dy) - in-place residual
        y0::AbstractVector{T},
        tspan;
        reltol = 1.0e-3,
        abstol = 1.0e-5,
        initstep = 1.0e-4,
        maxstep = Inf,
        minstep = 0,
        maxorder = MAXORDER,
        dy0 = zero(y0),
        norm = dassl_norm,
        factorize_jacobian = true,
        kwargs...
    ) where {T}

    # Clear history and initialize with initial conditions
    clear_history!(cache)
    push_history!(cache, tspan[1], y0, dy0)

    # Output storage (these are the only allocations in the main loop)
    tout = [tspan[1]]
    yout = [copy(y0)]
    dyout = [copy(dy0)]

    ord = 1
    h = initstep
    num_rejected = 0
    num_fail = 0

    # Initialize weights and Jacobian for first step improvement
    weights!(cache.wt, y0, reltol, abstol)
    numerical_jacobian!(cache.jac, cache, F!, tspan[1], y0, dy0, 1 / initstep, cache.wt)
    cache.jac_factorized = lu(cache.jac; check = false)
    cache.a = 1 / initstep

    # Improve initial dy0 estimate (one stepper iteration)
    # This uses the out-of-place stepper for the initial improvement
    wt_init = dassl_weights(y0, 1, 1)
    (
        _,
        _,
        _,
        dy_improved,
        _,
    ) = stepper(
        1, [tspan[1]], [y0], [dy0], 10 * eps(one(tspan[1])),
        (t, y, dy) -> begin
            F!(cache.residual, t, y, dy)
            return cache.residual
        end,
        JacData(cache.a, cache.jac),
        (t, y, dy, a) -> factorize(cache.jac),
        wt_init,
        v -> norm(v, wt_init), 1
    )

    # Update history with improved dy0
    copyto!(get_latest_dy(cache), dy_improved)
    dyout[1] = copy(dy_improved)

    # Main integration loop
    while tout[end] < tspan[2]
        hmin = max(4 * eps(one(tspan[1])), minstep)
        h = min(h, maxstep, tspan[2] - tout[end])

        if h < hmin
            throw(DomainError("Stepsize too small (h=$h at t=$(tout[end])."))
        elseif num_fail >= -2 / 3 * log(eps(one(tspan[1])))
            throw(ErrorException("Too many ($num_fail) failed steps"))
        end

        # Compute weights in-place
        weights!(cache.wt, get_latest_y(cache), reltol, abstol)
        normy(v) = norm(v, cache.wt)

        # Take step (in-place, writes to cache.yn, cache.dyn)
        (status, err) = stepper!(cache, ord, h, F!, normy, maxorder)

        if status < 0
            # Newton failed, reduce step
            num_fail += 1
            num_rejected += 1
            h *= 1 / 4
            continue

        elseif err > 1
            # Error too large, reduce step
            num_fail += 1
            num_rejected += 1

            # Temporarily add step for order selection
            t_new = tout[end] + h
            push_history!(cache, t_new, cache.yn, cache.dyn)

            # Determine new step size and order
            (r, ord) = newStepOrder_cache(cache, normy, err, ord, num_fail, maxorder)

            # Remove temporary step
            pop_oldest_history!(cache)
            # Actually we need to reset to previous state - the circular buffer
            # approach doesn't support easy "undo", so we use output arrays
            clear_history!(cache)
            for i in eachindex(tout)
                push_history!(cache, tout[i], yout[i], dyout[i])
            end

            h *= r
            continue

        else
            # Step accepted
            num_fail = 0
            t_new = tout[end] + h

            # Add to history
            push_history!(cache, t_new, cache.yn, cache.dyn)

            # Trim history if needed (keep at most ord + 3)
            while cache.hist_len > ord + 3
                pop_oldest_history!(cache)
            end

            # Save to output (only allocations in main loop)
            push!(tout, t_new)
            push!(yout, copy(cache.yn))
            push!(dyout, copy(cache.dyn))

            # Determine new step size and order
            (r, ord) = newStepOrder_cache(cache, normy, err, ord, num_fail, maxorder)

            h *= r
        end
    end

    return (tout, yout, dyout)
end

"""
    newStepOrder_cache(cache, normy, err, k, num_fail, maxorder)

Determine new step size and order based on cache history.
Similar to newStepOrder but operates on cache.
"""
function newStepOrder_cache(
        cache::DASSLCache,
        normy,
        err,
        k::Integer,
        num_fail::Integer,
        maxorder::Integer
    )
    available_steps = cache.hist_len

    if num_fail >= 3
        (r, order) = (1 / 4, 1)

    elseif num_fail == 1 && available_steps == 1 && err > 1
        (r, order) = (1 / 4, 1)

    elseif num_fail == 0 && available_steps == 2 && err < 1
        (r, order) = (1, 2)

    elseif available_steps < k + 2
        if num_fail == 0
            (r, order) = ((2 * err + 1 / 10000)^(-1 / (k + 1)), k)
        else
            (r, order) = (1 / 4, 1)
        end

    else
        # Use continuous order selection (zero-allocation via cache buffers)
        (r, order) = newStepOrderContinuous_cache!(cache, normy, err, k, maxorder)
        r = normalizeStepSize(r, num_fail)

        if num_fail > 0
            order = min(order, k)
        end
    end

    return r, order
end
