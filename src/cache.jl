# Cache structures for in-place DASSL operations
# Following OrdinaryDiffEq.jl pattern: pre-allocate all working arrays

"""
    DASSLCache{T, uType, jacType}

Pre-allocated cache for in-place DASSL operations.
All working arrays are allocated once and reused across integration steps.
"""
mutable struct DASSLCache{T, uType, jacType}
    # Current state vectors (reused each step)
    yn::uType           # Next y value (corrector output)
    dyn::uType          # Next dy value
    y0::uType           # Predictor y value
    dy0::uType          # Predictor dy value

    # Newton iteration working vectors
    delta::uType        # Newton step delta
    residual::uType     # Residual vector for in-place F evaluation
    ytmp::uType         # Temporary vector for computations
    ytmp2::uType        # Second temporary vector

    # Jacobian storage
    jac::jacType        # Jacobian matrix buffer
    jac_factorized::Any # Factorized Jacobian (stored as Any for flexibility)
    a::T                # Current Jacobian coefficient

    # Numerical Jacobian working arrays
    f_plus::uType       # F(y + delta) result
    f_base::uType       # F(y) result

    # History buffers (circular buffer, fixed size for BDF order 1-6)
    t_hist::Vector{T}           # Time history
    y_hist::Vector{uType}       # Solution history
    dy_hist::Vector{uType}      # Derivative history
    hist_start::Int             # Start index in circular buffer
    hist_len::Int               # Current number of valid entries

    # Error weights buffer
    wt::uType
end

"""
    alg_cache(alg, u, p, t, ::Val{true})

Create a mutable cache for in-place DASSL operations.
All working arrays are pre-allocated based on the size of `u`.
"""
function alg_cache(alg, u::uType, p, t::T, ::Val{true}) where {T, uType}
    n = length(u)

    # State vectors
    yn = similar(u)
    dyn = similar(u)
    y0 = similar(u)
    dy0 = similar(u)

    # Newton iteration vectors
    delta = similar(u)
    residual = similar(u)
    ytmp = similar(u)
    ytmp2 = similar(u)

    # Jacobian storage
    jac = zeros(eltype(u), n, n)
    jac_factorized = nothing
    a = zero(T)

    # Numerical Jacobian vectors
    f_plus = similar(u)
    f_base = similar(u)

    # History buffers (MAXORDER + 3 = 9 entries max needed)
    max_hist = MAXORDER + 3
    t_hist = zeros(T, max_hist)
    y_hist = [similar(u) for _ in 1:max_hist]
    dy_hist = [similar(u) for _ in 1:max_hist]

    # Error weights
    wt = similar(u)

    return DASSLCache(
        yn, dyn, y0, dy0,
        delta, residual, ytmp, ytmp2,
        jac, jac_factorized, a,
        f_plus, f_base,
        t_hist, y_hist, dy_hist, 1, 0,
        wt
    )
end

"""
    alg_cache(alg, u, p, t, ::Val{false})

For out-of-place problems, return nothing (no cache needed).
"""
function alg_cache(alg, u, p, t, ::Val{false})
    return nothing
end

# ============================================================================
# Circular buffer history management
# ============================================================================

"""
    push_history!(cache, t, y, dy)

Add a new entry to the history circular buffer.
If buffer is full, overwrites the oldest entry.
"""
function push_history!(cache::DASSLCache, t, y, dy)
    max_size = length(cache.t_hist)

    if cache.hist_len < max_size
        # Buffer not full, append
        cache.hist_len += 1
    else
        # Buffer full, advance start pointer (overwrite oldest)
        cache.hist_start = mod1(cache.hist_start + 1, max_size)
    end

    # Compute index for new entry
    idx = mod1(cache.hist_start + cache.hist_len - 1, max_size)

    cache.t_hist[idx] = t
    copyto!(cache.y_hist[idx], y)
    return copyto!(cache.dy_hist[idx], dy)
end

"""
    pop_oldest_history!(cache)

Remove the oldest entry from the history buffer.
"""
function pop_oldest_history!(cache::DASSLCache)
    return if cache.hist_len > 0
        cache.hist_start = mod1(cache.hist_start + 1, length(cache.t_hist))
        cache.hist_len -= 1
    end
end

"""
    get_t_at(cache, i)

Get time at history index i (1 = oldest in current window).
"""
function get_t_at(cache::DASSLCache, i::Integer)
    max_size = length(cache.t_hist)
    idx = mod1(cache.hist_start + i - 1, max_size)
    return cache.t_hist[idx]
end

"""
    get_y_at(cache, i)

Get y vector at history index i (1 = oldest in current window).
Returns a reference to the stored vector.
"""
function get_y_at(cache::DASSLCache, i::Integer)
    max_size = length(cache.y_hist)
    idx = mod1(cache.hist_start + i - 1, max_size)
    return cache.y_hist[idx]
end

"""
    get_dy_at(cache, i)

Get dy vector at history index i (1 = oldest in current window).
Returns a reference to the stored vector.
"""
function get_dy_at(cache::DASSLCache, i::Integer)
    max_size = length(cache.dy_hist)
    idx = mod1(cache.hist_start + i - 1, max_size)
    return cache.dy_hist[idx]
end

"""
    get_latest_t(cache)

Get the most recent time value.
"""
function get_latest_t(cache::DASSLCache)
    return get_t_at(cache, cache.hist_len)
end

"""
    get_latest_y(cache)

Get the most recent y vector.
"""
function get_latest_y(cache::DASSLCache)
    return get_y_at(cache, cache.hist_len)
end

"""
    get_latest_dy(cache)

Get the most recent dy vector.
"""
function get_latest_dy(cache::DASSLCache)
    return get_dy_at(cache, cache.hist_len)
end

"""
    get_history_t(cache, ord)

Get a vector of the last `ord` time values for interpolation.
Returns values from oldest to newest within the window.
Note: This allocates a small vector - could be optimized further.
"""
function get_history_t(cache::DASSLCache, ord::Integer)
    n = min(ord, cache.hist_len)
    t_vec = Vector{eltype(cache.t_hist)}(undef, n)
    start_idx = cache.hist_len - n + 1
    @inbounds for i in 1:n
        t_vec[i] = get_t_at(cache, start_idx + i - 1)
    end
    return t_vec
end

"""
    get_history_y(cache, ord)

Get a vector of the last `ord` y vectors for interpolation.
Returns references from oldest to newest within the window.
Note: This allocates a small vector of references.
"""
function get_history_y(cache::DASSLCache, ord::Integer)
    n = min(ord, cache.hist_len)
    y_vec = Vector{eltype(cache.y_hist)}(undef, n)
    start_idx = cache.hist_len - n + 1
    @inbounds for i in 1:n
        y_vec[i] = get_y_at(cache, start_idx + i - 1)
    end
    return y_vec
end

"""
    get_history_dy(cache, ord)

Get a vector of the last `ord` dy vectors for interpolation.
Returns references from oldest to newest within the window.
"""
function get_history_dy(cache::DASSLCache, ord::Integer)
    n = min(ord, cache.hist_len)
    dy_vec = Vector{eltype(cache.dy_hist)}(undef, n)
    start_idx = cache.hist_len - n + 1
    @inbounds for i in 1:n
        dy_vec[i] = get_dy_at(cache, start_idx + i - 1)
    end
    return dy_vec
end

"""
    clear_history!(cache)

Reset the history buffer to empty state.
"""
function clear_history!(cache::DASSLCache)
    cache.hist_start = 1
    return cache.hist_len = 0
end

"""
    weights!(wt, y, reltol, abstol)

Compute error weights in-place.
"""
function weights!(wt, y, reltol, abstol)
    return @. wt = reltol * abs(y) + abstol
end
