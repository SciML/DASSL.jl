__precompile__()

module DASSL

export dasslIterator, dasslSolve

using Reexport
@reexport using DiffEqBase
using LinearAlgebra

import DiffEqBase: solve

export dassl

include("common.jl")

const MAXORDER = 6
const MAXIT = 10

mutable struct JacData
    a::Real
    jac # Jacobian matrix for the newton solver
end

function dasslStep(channel,
                   F,
                   y0::AbstractVector{T},
                   tstart::Real;
                   reltol   = 1e-3,
                   abstol   = 1e-5,
                   initstep = 1e-4,
                   maxstep  = Inf,
                   minstep  = 0,
                   maxorder = MAXORDER,
                   dy0      = zero(y0),
                   tstop    = Inf,
                   norm     = dassl_norm,
                   weights  = dassl_weights,
                   factorize_jacobian = true, # whether to store factorized version of jacobian
                   jacobian = numerical_jacobian(F,reltol,abstol,weights), # computes the quantity F/dy+a*dF/dy'
                   args...) where T<:Number

    n  = length(y0)
    jd = JacData(zero(tstart),zeros(T,n,n)) # generate a dummy
                                   # jacobian, it will be replaced by
                                   # a real one after running stepper

    ord      = 1                # initial method order
    tout     = [tstart]         # initial time
    h        = initstep         # current step size
    yout     = Array{typeof(y0)}(undef,1)
    yout[1]  = y0
    dyout    = Array{typeof(y0)}(undef,1)
    dyout[1] = dy0              # initial guess for dy0
    num_rejected = 0            # number of rejected steps
    num_fail = 0                # number of consecutive failures

    # wrapper around the jacobian function
    if factorize_jacobian
        computejac = (t,y,dy,a) -> LinearAlgebra.factorize(jacobian(t,y,dy,a))
    else
        computejac = jacobian
    end

    # This is the trick to improve on the initial guess for
    # y0.  Basically we run a one iteration of stepper and use the
    # result as the initial derivative
    (_,_,_,dyout[1],_)=stepper(1,tout,yout,dyout,10*eps(one(tstart)),F,jd,computejac,weights(y0,1,1),v->norm(v,weights(y0,1,1)),1)

    while tout[end] < tstop

        hmin = max(4*eps(one(tstart)),minstep)
        h = min(h,maxstep,tstop-tout[end])

        if h < hmin
            throw(DomainError("Stepsize too small (h=$h at t=$(tout[end])."))
            break
        elseif num_fail >= -2/3*log(eps(one(tstart)))
            throw(ErrorException("Too many ($num_fail) failed steps in a row (h=$h at t=$(tout[end])."))
            break
        end

        # error weights
        wt = weights(yout[end],reltol,abstol)
        normy(v) = norm(v,wt)

        (status,err,yn,dyn,jd)=stepper(ord,tout,yout,dyout,h,F,jd,computejac,wt,normy,maxorder)

        if status < 0
            # Early failure: Newton iteration failed to converge, reduce
            # the step size and try again

            # increase the failure counter
            num_fail     += 1
            # keep track of the total number of rejected steps
            num_rejected += 1
            # reduce the step by 25%
            h *= 1/4
            continue

        elseif err > 1
            # local error is too large.  Step is rejected, and we try
            # again with new step size and order.

            # increase the failure counter
            num_fail     += 1
            # keep track of the total number of rejected steps
            num_rejected += 1
            # temporarily push the new step to the yout (needed by newStepOrder)
            push!(tout,tout[end]+h); push!(yout,yn)
            # determine the new step size and order, excluding the current step
            (r,ord) = newStepOrder(tout,yout,normy,err,ord,num_fail,maxorder)

            # pop the temporary steps
            pop!(tout); pop!(yout)
            h *= r
            continue

        else
            ####################
            # step is accepted #
            ####################

            # reset the failure counter
            num_fail      = 0

            # save the results
            push!(tout, tout[end]+h)
            push!(yout, yn)
            push!(dyout,dyn)

            # remove old results
            if length(tout) > ord+3
                popfirst!(tout)
                popfirst!(yout)
                popfirst!(dyout)
            end

            push!(channel,(tout[end],yout[end],dyout[end]))

            # determine the new step size and order, including the current step
            (r,ord) = newStepOrder(tout,yout,normy,err,ord,num_fail,maxorder)

            h *= r
        end

    end

end


# the iterator version of the dasslSolve
dasslIterator(F, y0, t0; args...) = Channel((channel)->dasslStep(channel,F, y0, t0; args...))

# solves the equation F with initial data y0 over for times t in tspan=[t0,t1]
function dasslSolve(F, y0::AbstractVector, tspan; dy0 = zero(y0), args...)
    tout  = Array{typeof(tspan[1])}(undef,1)
    yout  = Array{typeof(y0)}(undef,1)
    dyout = Array{typeof(y0)}(undef,1)
    tout[1]  = tspan[1]
    yout[1]  = y0
    dyout[1] = dy0
    for (t, y, dy) in dasslIterator(F, y0, tspan[1]; dy0=dy0, tstop=tspan[end], args...)
        push!( tout,  t)
        push!( yout,  y)
        push!(dyout, dy)
        if t >= tspan[end]
            break
        end
    end
    return (tout,yout,dyout)
end

# A scalar version of dasslSolve, implemented as a wrapper around dasslSolve
function dasslSolve(F, y0::Number, tspan; args...)
    (tout,yout,dyout) = dasslSolve(F,[y0],tspan; args...)
    return (tout,map(first,yout),map(first,dyout))
end


function newStepOrder(t::AbstractVector,
                      y::AbstractVector,
                      normy,
                      err,
                      k::Integer,
                      num_fail::Integer,
                      maxorder::Integer)

    if length(t) != length(y)
        error("incompatible size of y and t")
    end

    available_steps = length(t) # including the step t_{n+1}

    if num_fail >= 3
        # probably, the step size was drastically decreased for
        # several steps in a row, so we reduce the order to one and
        # further decrease the step size
        (r,order) = (1/4,1)

    elseif num_fail == 1 && available_steps == 1 && err > 1
        # the last step was accepted, increase order to two
        (r,order) = (1/4,1)

    elseif num_fail == 0 && available_steps == 2 && err < 1
        # The first successfull step, try increasing the order without changing the step size
        (r,order) = (1, 2)

    elseif available_steps < k+2
        # we are at the beginning of the integration, we don't have
        # enough steps to run newStepOrderContinuous, we have to rely
        # on a crude order/stepsize selection
        if num_fail == 0
            # previous step was accepted so we can increase the order
            # and the step size
            # (r,order) = (2.0,min(k+1,maxorder))
            (r,order) = ((2*err+1/10000)^(-1/(k+1)), k)
            # (r,order) = (2.0,1)
        else
            # @todo I am not sure about the choice of order
            #
            # previous step was rejected, we have to decrease the step
            # size and order
            (r,order) = (1/4,1)
        end

    else
        # we have at least k+3 previous steps available, so we can
        # safely estimate the order k-2, k-1, k and possibly k+1
        (r,order) = newStepOrderContinuous(t,y,normy,err,k,maxorder)
        # this function prevents from step size changing too rapidly
        r = normalizeStepSize(r,num_fail)

        # if the previous step failed don't increase the order
        if num_fail > 0
            order = min(order,k)
        end

    end

    return r, order

end


function newStepOrderContinuous(t::AbstractVector,
                                y::AbstractVector,
                                normy,
                                err,
                                k::Integer,
                                maxorder::Integer)

    # compute the error estimates of methods of order k-2, k-1, k and
    # (if possible) k+1
    errors  = errorEstimates(t,y,normy,k)
    errors[k] = err
    ne = length(errors)         # == k or k+1

    # we want to be conservative in our choice of order so we consider
    # not only a monotone sequences but geometrically monotone
    # sequences
    errors_dec  = all(diff([errors[i] for i=max(k-2,1):min(ne,maxorder)]).<0)
    errors_inc  = all(diff([errors[i] for i=max(k-2,1):min(ne,maxorder)]).>0)


    if ne == k+1 && errors_dec
        # If we can estimate the error for k+1 order and the Taylor
        # expansion errors form a decreasing sequence, we can safely
        # increase the order
        order = min(k+1,maxorder)
    elseif ne > 1 && errors_inc
        # Taylor expansion errors form an increasing sequence, we
        # should decrease the order
        order = max(k-1,1)
    else
        # otherwise, leave the current order
        order = k
    end

    # error estimate for the next step
    est = errors[order]

    # initial guess for the new step size multiplier
    r = (2*est+1/10000)^(-1/(order+1))

    return r, order

end


# Based on whether the previous steps were successful we determine
# the new step size
#
# num_fail is the number of steps that failed before this step, r is a
# suggested step size multiplier.
function normalizeStepSize(r, num_fail)

    if num_fail == 0
        # previous step was accepted
        if r >= 2
            r = 2.0
        elseif r < 1
            # choose r from between 0.5 and 0.9
            r = max(1/2,min(r,9/10))
        else
            r = 1.0
        end

    elseif num_fail == 1
        # previous step failed, we slightly decrease the step size,
        # the resulting r is between 0.25 and 0.9
        r = max(1/4,9/10*min(r,1))

    elseif num_fail == 2
        # previous step failed for a second time, error estimates are
        # probably not reliable so decrease the step size
        r = 1/4

    end

    return r

end



# this function estimates the errors of methods of order k-2,k-1,k,k+1
# and returns the estimates as an array.
#
# This method requires at least k previous steps (not including the
# predicted step), so the length of tables t and y should be at least
# k+2. In particular, these estimates won't work for the first step.
#
# The error estimates for order `kest` are computed as the maximal
# derivative `err=norm(h^{(kest+1)}*u^{(kest+1)})` of an interpolating
# polynomial constructed from kest+1 steps at
# t_{n+1},...,t_{n-kest}.  So to estimate the error of e.g. order
# `kest=1` we need steps `t_{n+1},t_{n},t_{n-1}`, in general this
# method needs k+1 previous steps and one future step.
#
# additionally, if the previous k+1 steps were made with the same time
# step, we can estimate the error for order k+1.
#

# here t is an array of times    t=[t_1, ..., t_n, t_{n+1}]
# and y is an array of solutions y=[y_1, ..., y_n, y_{n+1}]
function errorEstimates(t::AbstractVector,
                        y::AbstractVector,
                        normy,
                        k::Integer)

    nsteps = length(t)          # available steps (including counting
                                # the new n+1'st step)
    h  = t[end]-t[end-1]        # current step size

    if nsteps < k+2
        error("errorEstimates called with too few steps.")
    end

    # estimates the error for orders [k-2,k-1,k]
    errors = zeros(eltype(t),k)

    for i = max(k-2,1):k
        # @todo can this be optimized by inlining the body of
        # interpolateHighestDerivative?
        maxd=interpolateHighestDerivative(t[end-(i+1):end],y[end-(i+1):end])
        errors[i]=h^(i+1)*normy(maxd)
    end

    # compute the estimate the k+1 order only if all the steps were
    # made using the same stepsize
    if nsteps >= k+3
        hn = diff(t[end-(k+2):end])
        if all(hn.==hn[1])
            maxd=interpolateHighestDerivative(t[end-(k+2):end],y[end-(k+2):end])
            push!(errors,h^(k+2)*normy(maxd))
        end
    end

    # return error estimates (this is roughly [ERKM2,ERKM1,ERK,ERKP1]
    # from DASSL)
    return errors

end


# t is an array [t_1,...,t_n] of length n
# y is a matrix [y_1,...,y_n] of size k x l
# h_next is a size of next step
# F encodes the DAE: F(t,y,y')=0
# jd is a bunch of auxilary data saved between steps (jacobian and last coefficient 'a')
# wt is a vector of weights of the norm
function stepper(ord::Integer,
                 t::AbstractVector,
                 y::AbstractVector,
                 dy::AbstractVector,
                 h_next::Real,
                 F,
                 jd::JacData,
                 computejac,
                 wt,
                 normy,
                 maxorder::Integer)

    l        = length(y[1])        # the number of dependent variables

    # @todo this should be the view of the tail of the arrays t and y
    tk = t[end-ord+1:end]
    yk = y[end-ord+1:end]

    t_next   = tk[end]+h_next

    # I think there is an error in the book, the sum should be taken
    # from j=1 to k+1 instead of j=1 to k
    alphas = -sum([1/j for j=1:ord])

    if length(y) == 1
        # this is the first step, we initialize y0 and dy0 with
        # initial data provided by user
        dy0 = dy[1]
        y0  = y[1]+h_next*dy[1]
        # We assume that dy[1] is given, so we can use Hermite
        # interpolation to get a better approximation to the
        # derivative at point y: dy=a*y+b=2(y-y[1])/h-dy[1]
        a=2/h_next
        b=-dy[1]-2*y[1]/h_next
    else
        # we use predictor to obtain the starting point for the
        # modified newton method
        #
        # @todo I should optimize the following functions to return a
        # tuple (y0,dy0)
        dy0 = interpolateDerivativeAt(tk,yk,t_next)
        y0  = interpolateAt(tk,yk,t_next)
        a=-alphas/h_next
        b=dy0-a*y0
    end

    # f_newton is supplied to the modified Newton method.  Zeroes of
    # f_newton give the corrected value of the next step "yc"
    f_newton(yc)=F(t_next,yc,a*yc+b)

    # if called, this function computes the jacobian of f_newton at
    # the point y0 (via first order finite differences or
    # user-supplied functions)
    jac_new()=computejac(t_next,y0,dy0,a)

    # we compute the corrected value "yc", updating the gradient if necessary
    (status,yc,jd)=corrector(jd,       # old coefficient a and jacobian
                             a,        # current coefficient a
                             jac_new,  # this function is called when new jacobian is needed
                             y0,       # starting point for modified newton
                             f_newton, # we want to find zeroes of this function
                             normy)     # the norm used to estimate error needs weights

    alpha = Array{eltype(t)}(undef,ord+1)

    for i = 1:ord
        alpha[i] = h_next/(t_next-t[end-i+1])
    end

    if length(t) >= ord+1
        t0 = t[end-ord]
    elseif length(t) >= 2
        # @todo we choose some arbitrary value of t[0], here t[0]:=t[1]-(t[2]-h[1])
        h1 = t[2]-t[1]
        t0 = t[1]-h1
    else
        t0 = t[1]-h_next
    end

    alpha[ord+1] = h_next/(t_next-t0)

    alpha0 = -sum(alpha[1:ord])
    M      =  max(alpha[ord+1],abs.(alpha[ord+1]+alphas-alpha0))
    err::eltype(t) =  normy((yc-y0))*M


    # status<0 means the modified Newton method did not converge
    # err is the local error estimate from taking the step
    # yc is the estimated value at the next step
    return (status, err, yc, a*yc+b, jd)

end


# returns the corrected value yc and status.  If needed it updates
# the jacobian g_old and a_old.

function corrector(jd::JacData,
                   a_new::T,
                   jac_new,
                   y0::AbstractVector{Ty},
                   f_newton,
                   normy) where {T,Ty}

    # if jd.a == 0 the new jacobian is always computed, independently
    # of the value of a_new
    if abs((jd.a-a_new)/(jd.a+a_new)) > 1/4
        # old jacobian wouldn't give fast enough convergence, we have
        # to compute a current jacobian
        jd = JacData(a_new,jac_new())
        # run the corrector
        (status,yc)=newton_iteration( x->(-(jd.jac\f_newton(x))), y0, normy)
    else
        # old jacobian should give reasonable convergence
        c::T=2*jd.a/(a_new+jd.a) # factor "c" is used to speed up the
                                 # convergence when using an old
                                 # jacobian

        # reusing the old jacobian
        (status,yc)=newton_iteration( x->(-c*(jd.jac\f_newton(x))), y0, normy)

        if status < 0
            # the corrector did not converge, so we recompute jacobian and try again
            jd = JacData(a_new,jac_new())
            # run the corrector again
            (status,yc)=newton_iteration( x->(-(jd.jac\f_newton(x))), y0, normy)
        end
    end

    return (status,yc,jd)

end


# this function iterates f until it finds its fixed point, starting
# from f(y0).  The result either satisfies normy(yn-f(yn))=0+... or is
# set back to y0.  Status tells if the fixed point was obtained
# (status==0) or not (status==-1).
function newton_iteration(f,
                          y0::AbstractVector{T},
                          normy) where T<:Number

    # first guess comes from the predictor method, then we compute the
    # second guess to get the norm1

    delta::typeof(y0)=f(y0)
    norm1::T=normy(delta)
    yn=y0+delta

    # after the first iteration the normy turned out to be very small,
    # terminate and return the first correction step

    ep    = eps(eltype(abs.(y0))) # this is the epsilon for type y0

    if norm1 < 100*ep*normy(y0)
        status=0
        return(status,yn)
    end

    # maximal number of iterations is set by dassl algorithm to 4

    for i=1:MAXIT

        delta=f(yn)
        normn::T=normy(delta)
        rho=(normn/norm1)^(1/i)
        yn=yn+delta

        # iteration failed to converge

        if rho > 9/10
            status=-1
            return(status,y0)
        end

        err=rho/(1-rho)*normn

        # iteration converged successfully

        if err < 1/3
            status=0
            return(status,yn)
        end

    end

    # unable to converge after 4 iterations

    status=-1
    return(status,y0)
end


function dassl_norm(v, wt)
    norm(v./wt)/sqrt(length(v))
end

function dassl_weights(y,reltol,abstol)
    @. reltol*abs(y)+abstol
end

# returns the value of the interpolation polynomial at the point x0
function interpolateAt(x::AbstractVector{T},
                       y::AbstractVector,
                       x0::T) where T<:Real

    if length(x)!=length(y)
        error("x and y have to be of the same size.")
    end

    n = length(x)
    p = zero(y[1])

    for i=1:n
        Li =one(T)
        for j=1:n
            if j==i
                continue
            else
                Li*=(x0-x[j])/(x[i]-x[j])
            end
        end
        p+=Li*y[i]
    end
    return p
end


# returns the value of the derivative of the interpolation polynomial
# at the point x0
function interpolateDerivativeAt(x::AbstractVector{T},
                                 y::AbstractVector,
                                 x0::T) where T<:Real

    if length(x)!=length(y)
        error("x and y have to be of the same size.")
    end

    n = length(x)
    p = zero(y[1])

    for i=1:n
        dLi=zero(T)
        for k=1:n
            if k==i
                continue
            else
                dLi1=one(T)
                for j=1:n
                    if j==k || j==i
                        continue
                    else
                        dLi1*=(x0-x[j])/(x[i]-x[j])
                    end
                end
                dLi+=dLi1/(x[i]-x[k])
            end
        end
        p+=dLi*y[i]
    end
    return p
end


# if the interpolating polynomial is given as
# p(x)=a_{k-1}*x^{k-1}+...a_1*x+a_0 then this function returns the
# k-th derivative of p, i.e. (k-1)!*a_{k-1}
function interpolateHighestDerivative(x::AbstractVector,
                                      y::AbstractVector)

    if length(x)!=length(y)
        error("x and y have to be of the same size.")
    end

    n = length(x)
    p = zero(y[1])

    for i=1:n
        Li =one(eltype(x))
        for j=1:n
            if j==i
                continue
            else
                Li*=1/(x[i]-x[j])
            end
        end
        p+=Li*y[i]
    end
    return factorial(n-1)*p
end

# generate a function that computes approximate jacobian using forward
# finite differences
function numerical_jacobian(F,reltol,abstol,weights)
    function numjac(t, y, dy, a)
        ep      = eps(one(t))   # this is the machine epsilon
        h       = 1/a           # h ~ 1/a
        wt      = weights(y,reltol,abstol)
        # delta for approximation of jacobian.  I removed the
        # sign(h_next*dy0) from the definition of delta because it was
        # causing trouble when dy0==0 (which happens for ord==1)
        edelta  = diagm(0=>max.(abs.(y),abs.(h*dy),wt)*sqrt(ep))

        b=dy-a*y
        f(y1) = F(t,y1,a*y1+b)

        n   = length(y)
        jac = Array{eltype(y)}(undef,n,n)
        for i=1:n
            jac[:,i]=(f(y+edelta[:,i])-f(y))/edelta[i,i]
        end

        jac
    end
end

end
