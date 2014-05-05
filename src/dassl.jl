# @todo better estimate on error of order k (error[k])
# @todo changing MAXORDER to 3 results in BoundsError()

module dassl

using InterPol

export driver

const MAXORDER = 6

type StepperStuff{T<:Number}
    a     :: T
    g     :: Matrix{T}
    evals :: Integer
end

function driver{T<:Number}(F             :: Function,
                           y0            :: Vector{T},
                           tspan         :: Vector{T};
                           rtol = 1.0e-3 :: T,
                           atol = 1.0e-3 :: T,
                           h0   = 1.0e-4 :: T)

    t_start = tspan[1]
    t_stop  = tspan[end]

    # we allocate the space for Jacobian of a function F(t,y,a*y+b)
    # with a and b defined in the stepper!
    g = zeros(T,length(y0),length(y0))
    # The parameter a has to be kept between consecutive calls of
    # stepper!
    a = zero(T)
    # zip the stepper temporary variables in a type
    stuff = StepperStuff(a,g,0)

    wt = zeros(T,length(y0))

    ord      = 1                    # initial method order
    t        = [t_start]            # initial time
    h        = h0                   # current step size
    y        = hcat(y0)             # initial data
    num_fail = 0                    # number of consecutive failures
    num_accepted = 0
    num_rejected = 0
    nfixed = 0

    ordn = [ord]
    evalsn = [0]
    r = 1
    ord = 1

    while t[end] < t_stop

        epsilon = eps(one(T))
        hmin = 4*epsilon*max(abs(t[end]),abs(t_stop))

        if h < hmin
            warn("Stepsize too small (h=$h at t=$(t[end]), terminating")
            break
        end

        # weights for the norm
        wt = rtol*abs(y[:,end]).+atol
        # norm used to determine the local error of the numerical
        # solution
        dassl_norm(v)=norm(v./wt)/sqrt(length(v))

        (status,err,yn)=stepper!(ord,t,y,h,F,stuff,wt)

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

        elseif err > 1.0
            # local error is too large.  Step is rejected, and we try
            # again with new step size and order.

            # increase the failure counter
            num_fail     += 1
            # keep track of the total number of rejected steps
            num_rejected += 1
            # determine the new step size and order, excluding the current step
            (r,ord) = newStepOrder([t, t[end]+h],y,dassl_norm,ord,num_fail,nfixed,err)
            h *= r
            # error("ord -> ord+1 ale nie zaakceptowaliśmy wcześniejszego kroku")
            continue

        else
            ####################
            # step is accepted #
            ####################

            # reset the failure counter
            num_fail      = 0
            num_accepted += 1

            # save the results
            push!(t,t[end]+h)
            y    = [y   yn]
            push!(ordn, ord)
            # determine the new step size and order, including the current step
            (r_new,ord_new) = newStepOrder([t, t[end]+h],y,dassl_norm,ord,num_fail,nfixed,err)

            if ord_new == ord && r_new == r
                nfixed += 1
            else
                # @todo am I counting the number of fixed
                # order/stepsize steps correctly?
                nfixed = 1
            end

            (r,ord) = (r_new,ord_new)

            h *= r
        end

    end

    return(t,y,ordn,evalsn,num_accepted,num_rejected)

end


function newStepOrder{T<:Number}(t         :: Vector{T},
                                 y         :: Matrix{T},
                                 norm      :: Function,
                                 k         :: Integer,
                                 num_fail  :: Integer,
                                 nfixed    :: Integer,
                                 erk       :: T)

    if length(t) != size(y,2)+1
        error("incompatible size of y and t")
    end

    available_steps = length(t)

    if num_fail >= 3
        # probably, the step size was drastically decreased for
        # several steps in a row, so we reduce the order to one and
        # further decrease the step size
        (r,order) = (1/4,1)

    elseif available_steps < k+3
        # we are at the beginning of the integration, we don't have
        # enough steps to run newStepOrderContinuous, we have to rely
        # on a crude order/stepsize selection
        if num_fail == 0
            # previous step was accepted so we can increase the order
            # and the step size
            (r,order) = (2,min(k+1,MAXORDER))
        else
            # @todo fix this step, I am not sure about the choice of order
            #
            # previous step was rejected, we have to decrease the step
            # size and order
            (r,order) = (1/4,max(k-1,1))
        end

    else
        # we have at least k+3 previous steps available, so we can
        # safely estimate the order k-2, k-1, k and possibly k+1
        (r,order) = newStepOrderContinuous(t,y,norm,k,nfixed,erk)
        # this function prevents from step size changing too rapidly
        r = normalizeStepSize(r,num_fail)
        # if the previous step failed don't increase the order
        if num_fail > 0
            order = min(order,k)
        end

    end

    return r, order

end


function newStepOrderContinuous{T<:Number}(t      :: Vector{T},
                                           y      :: Matrix{T},
                                           norm   :: Function,
                                           k      :: Integer,
                                           nfixed :: Integer,
                                           erk    :: T)

    # compute the error estimates of methods of order k-2, k-1, k and
    # (if possible) k+1
    errors  = errorEstimates(t,y,norm,k,nfixed)
    errors[k] = erk
    # normalized errors, this is TERK from DASSL
    nerrors = errors .* [2:MAXORDER+1]

    order = k

    if k == 1
        if nerrors[k]/2 > nerrors[k+1]
            order = k+1
        end

    elseif k >= 2
        if k == 2 && nerrors[k-1] < nerrors[k]/2
            order = k-1
        elseif k >= 3 && max(nerrors[k-1],nerrors[k-2]) <= nerrors[k]
            order = k-1
        elseif k == MAXORDER
            order = k
        elseif false
            # @todo don't increase order two times in a row
            order = k
        elseif nfixed >= k+1
            # if the estimate for order k+1 is available
            if nerrors[k-1] <= min(nerrors[k],nerrors[k+1])
                order = k-1
            elseif nerrors[k] <= nerrors[k+1]
                order = k
            else
                order = k+1
            end
        end
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
function normalizeStepSize{T<:Number}(r        :: T,
                                      num_fail :: Integer)

    if num_fail == 0
        # previous step was accepted
        if r >= 2
            r = 2
        elseif r < 1
            # choose r from between 0.5 and 0.9
            r = max(1/2,min(r,9/10))
        else
            r = 1
        end

    elseif num_fail == 1
        # previous step failed, we slightly decrease the step size,
        # the resulting r is between 0.25 and 0.9
        r = max(1/4,9/10*min(r,1))

    elseif num_fail == 2
        # previous step failed for a second time, error estimates are
        # probably not reliable so decrease the step size
        r = 1/4

    elseif num_fail >= 3
        # @todo remove this case
        #
        # this should never be the case, it should be covered by the
        # newStepOrder function
        error("num_fail >= 3 inside normalizeStepSize")

    end

    return r

end



# this function estimates the errors of methods of order k-2,k-1,k,k+1
# and returns the estimates as an array seq the estimates require

# here t is an array of times    [t_1, ..., t_n, t_{n+1}]
# and y is an array of solutions [y_1, ..., y_n]
function errorEstimates{T<:Number}(t      :: Vector{T},
                                   y      :: Matrix{T},
                                   norm   :: Function,
                                   k      :: Integer,
                                   nfixed :: Integer)

    h = diff(t)

    l = size(y,1)

    psi    = cumsum(reverse(h[end-k-1:end]))

    # @todo there is no need to allocate array of size 1:k+3, we only
    # need a four element array k:k+3
    phi    = zeros(T,l,k+3)
    # fill in all but a last (k+3)-rd row of phi
    for i = 1:k+2
        phi[:,i] = prod(psi[1:i-1])*interpolateHighestDerivative(t[end-i+1:end],y[:,end-i+1:end])
    end

    sigma  = zeros(T,k+2)
    sigma[1] = 1
    for i = 2:k+2
        sigma[i] = (i-1)*sigma[i-1]*h[end]/psi[i]
    end

    errors    = zeros(T,MAXORDER)
    errors[k] = sigma[k+1]*norm(phi[:,k+2])

    if k >= 2
        # error estimate for order k-1
        errors[k-1] = sigma[k]*norm(phi[:,k+1])
    end

    if k >= 3
        # error estimate for order k-2
        errors[k-2] = sigma[k-1]*norm(phi[:,k])
    end

    if k <= 5 && nfixed >= k+1
        # error estimate for order k+1
        # fill in the rest of the phi array (the (k+3)-rd row)
        for i = k+3:k+3
            phi[:,i] = prod(psi[1:i-1])*interpolateHighestDerivative(t[end-i+1:end],y[:,end-i+1:end])
        end

        # estimate for the order k+1
        errors[k+1] = norm(phi[:,k+3])
    end

    # return error estimates (this is ERK{M2,M1,,P1} from DASSL)
    return errors

end


# t is an array [t_1,...,t_n] of length n
# y is a matrix [y_1,...,y_n] of size k x l
# h_next is a size of next step
# F encodes the DAE: F(y,y',t)=0
# stuff is a bunch of auxilary data saved between steps
# wt is a vector of weights of the norm
function stepper!{T<:Number}(ord    :: Integer,
                             t      :: Vector{T},
                             y      :: Matrix{T},
                             h_next :: T,
                             F      :: Function,
                             stuff  :: StepperStuff{T},
                             wt     :: Vector{T})

    l        = size(y,1)        # the number of dependent variables

    # sanity check
    # @todo remove it in final version
    if length(t) < ord || size(y,2) < ord
        error("Not enough points in a grid to use method of order $ord")
    end

    # @todo this should be the view of the tail of the arrays t and y
    tk = t[end-ord+1:end]
    yk = y[:,end-ord+1:end]

    # check whether order is between 1 and 6, for orders higher than 6
    # BDF does not converge
    if ord < 1 || ord > MAXORDER
        error("Order ord=$(ord) should be [1,...,$MAXORDER]")
        return(-1)
    end

    t_next   = tk[end]+h_next

    # we use predictor to obtain the starting point for the modified
    # newton method
    y0  = interpolateAt(tk,yk,t_next)
    dy0 = interpolateDerivativeAt(tk,yk,t_next)

    # I think there is an error in the book, the sum should be taken
    # from j=1 to k+1 instead of j=1 to k
    alphas = -sum([1/j for j=1:ord])

    a=-alphas/h_next
    b=dy0-a*y0

    # delta for approximation of jacobian.  I removed the
    # sign(h_next*dy0) from the definition of delta because it was
    # causing trouble when dy0==0 (which happens for ord==1)
    delta = jac_delta(y0,dy0,h_next,wt)

    # f_newton is supplied to the modified Newton method.  Zeroes of
    # f_newton give the corrected value of the next step "yc"
    f_newton(yc)=F(t_next,yc,a*yc+b)

    # if called, this function computes the jacobian of f_newton at
    # the point y0 via first order finite differences
    g_new()=G(f_newton,y0,delta)

    # this is the updated value of coefficient a, if jacobian is
    # udpated, corrector will replace stuff.a with a_new
    a_new=a

    # we compute the corrected value "yc", updating the gradient if necessary
    (status,yc)=corrector(stuff,    # old coefficient a and jacobian
                          a_new,    # current coefficient a
                          g_new,    # this function is called when new jacobian is needed
                          y0,       # starting point for modified newton
                          f_newton, # we want to find zeroes of this function
                          wt)       # the norm used to estimate error needs weights

    alpha = Array(T,ord+1)

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
    M      =  max(alpha[ord+1],abs(alpha[ord+1]+alphas-alpha0))
    err    =  norm(yc-y0)*M

    # status<0 means the modified Newton method did not converge
    # err is the local error estimate from taking the step
    # yc is the estimated value at the next step
    return (status, err, yc)

end


# returns the corrected value yc and status.  If needed it updates
# the jacobian g_old and a_old.

function corrector{T<:Number}(stuff    :: StepperStuff{T},
                              a_new    :: T,
                              g_new    :: Function,
                              y0       :: Vector{T},
                              f_newton :: Function,
                              wt       :: Vector{T})

    # if a_old == 0 the new jacobian is always computed, independently
    # of the value of a_new
    if abs((stuff.a-a_new)/(stuff.a+a_new)) > 1/4
        # old jacobian wouldn't give fast enough convergence, we have
        # to compute a current jacobian
        stuff.g=g_new()
        stuff.evals += 1
        stuff.a=a_new
        # run the corrector
        (status,yc)=newton_iteration( x->(-stuff.g\f_newton(x)), y0, wt)
    else
        # old jacobian should give reasonable convergence
        c=2*stuff.a/(a_new+stuff.a)     # factor "c" is used to speed up
                                    # the convergence when using an
                                    # old jacobian
        # reusing the old jacobian
        (status,yc)=newton_iteration( x->(-c*(stuff.g\f_newton(x))), y0, wt)

        if status < 0
            # the corrector did not converge, so we recompute jacobian and try again
            stuff.g=g_new()
            stuff.evals += 1
            stuff.a=a_new
            # run the corrector again
            (status,yc)=newton_iteration( x->(-stuff.g\f_newton(x)), y0, wt)
        end
    end

    return (status,yc)

end


# this function iterates f until it finds its fixed point, starting
# from f(y0).  The result either satisfies norm(yn-f(yn))=0+... or is
# set back to y0.  Status tells if the fixed point was obtained
# (status==0) or not (status==-1).
function newton_iteration{T<:Number}(f  :: Function,
                                     y0 :: Vector{T},
                                     wt :: Vector{T})

    # first guess comes from the predictor method, then we compute the
    # second guess to get the norm1

    delta=f(y0)
    norm1=dassl_norm(delta,wt)
    yn=y0+delta

    # after the first iteration the norm turned out to be very small,
    # terminate and return the first correction step

    if norm1 < 10*eps(T)
        status=0
        return(status,yn)
    end

    # maximal number of iterations is set by dassl algorithm to 4

    for i=1:3

        delta=f(yn)
        normn=dassl_norm(delta,wt)
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


# h=[h_{n-k}, ... , h_{n-1}]
# y=[y_{n-k}, ... , y_{n-1}, y_{n}]
function div_diff{T<:Number}(h :: Vector{T},
                             y :: Vector{T})
    if length(y) < 1
        error("length(y) is $(length(y)) in div_diff, should be >=1")
        return 0::T
    elseif length(y) == 1
        return y[1]::T
    elseif length(y) == 2
        return ((y[2]-y[1])/h[1])::T
    else
        return ((div_diff(h[2:end],y[2:end])-div_diff(h[1:end-1],y[1:end-1]))/sum(h))::T
    end
end


function dassl_norm{T<:Number}(v  :: Vector{T},
                               wt :: Vector{T})
    norm(v./wt)/sqrt(length(v))
end

# compute the G matrix from dassl (jacobian of F(t,x,a*x+b))
# @todo replace with symmetric finite difference?
function G{T<:Number}(f     :: Function,
                      y0    :: Vector{T},
                      delta :: Vector{T})
    n=length(y0)
    edelta=diagm(delta)
    s=Array(eltype(delta),n,n)
    for i=1:n
        s[:,i]=(f(y0+edelta[:,i])-f(y0))/delta[i]
    end
    return(s)
end

function jac_delta{T<:Number}(y0     :: Vector{T}
                              ,dy0   :: Vector{T},
                              h_next :: T,
                              wt     :: Vector{T})
    d = [ max(abs(y0[j]),abs(h_next*dy0[j]),wt[j]) for j=1:length(y0)]
    d*= sqrt(eps(T))
    return d
end


# returns the value of the interpolation polynomial at the point x0
function interpolateAt{T<:Number}(x::Vector{T}, y::Matrix{T}, x0::T)

    if length(x)!=size(y,ndims(y))
        error("x and y have to be of the same size.")
    end

    n = length(x)
    p = zeros(T,size(y,1))

    for i=1:n
        Li =one(T)
        for j=1:n
            if j==i
                continue
            else
                Li*=(x0-x[j])/(x[i]-x[j])
            end
        end
        p+=Li*y[:,i]
    end
    return p
end

# returns the value of the derivative of the interpolation polynomial
# at the point x0
function interpolateDerivativeAt{T<:Number}(x::Vector{T}, y::Matrix{T}, x0::T)

    if length(x)!=size(y,ndims(y))
        error("x and y have to be of the same size.")
    end

    n = length(x)
    p = zeros(T,size(y,1))

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
        p+=dLi*y[:,i]
    end
    return p
end

# returns the value of the interpolation polynomial at the point x0
function interpolateHighestDerivative{T<:Number}(x::Vector{T}, y::Matrix{T})

    if length(x)!=size(y,ndims(y))
        error("x and y have to be of the same size.")
    end

    n = length(x)
    p = zeros(T,size(y,1))
    Li =one(T)

    for i=1:n
        Li =one(T)
        for j=1:n
            if j==i
                continue
            else
                Li*=1/(x[i]-x[j])
            end
        end
        p+=Li*y[:,i]
    end
    return p
end

end
