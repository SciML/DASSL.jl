module dassl

export driver

const MAXORDER = 6

type StepperStuff{T<:Number}
    a :: T
    g :: AbstractArray{T,2}
    evals :: Integer
end

function driver{T<:Number}(F             :: Function,
                           y0            :: AbstractArray{T,1},
                           tspan         :: AbstractArray{T,1};
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

    ord      = 1                    # initial method order
    t        = [t_start]            # initial time
    h        = h0                   # current step size
    y        = hcat(y0)             # initial data
    num_fail = 0                    # number of consecutive failures
    num_accepted = 0
    num_rejected = 0

    ordn = [ord]

    while t[end] < t_stop


        if( num_accepted + num_rejected >  1000 )
            break
        end

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

        (status,err,yn)=stepper!(t[end-ord+1:end],y[:,end-ord+1:end],h,F,stuff,wt)

        if status < 0
            # Early failure: Newton iteration failed to converge, reduce
            # the step size and try again

            # increase the failure counter
            num_fail = num_fail+1
            num_rejected += 1
            # reduce the step by 25%
            h = 3/4*h
            continue

        elseif err > 1.0
            # local error is too large.  Step is rejected, and we try
            # again with new step size and order.

            # increase the failure counter
            num_fail = num_fail+1
            num_rejected += 1
            # determine the new step size and order
            # (h,ord)=newStepOrder([t, t[end]+h],[y yn],dassl_norm,ord,num_fail)
            ord=min(length(t),MAXORDER)
            continue

        else
            # step is accepted

            # reset the failure counter
            num_fail=0
            num_accepted += 1
            # determine the new step size and order
            # (h,ord)=newStepOrder([t, t[end]+h],[y yn],dassl_norm,ord,num_fail)
            ord=min(length(t),MAXORDER)
            # save the results
            t   = [t,  t[end]+h]
            y   = [y   yn]
            ordn = [ordn, ord]
        end

    end

    return(t,y,ordn,stuff.evals,num_accepted,num_rejected)

end

function newStepOrder{T<:Number}(t        :: AbstractArray{T,1},
                                 y        :: AbstractArray{T,2},
                                 norm     :: Function,
                                 ord_old  :: Integer,
                                 num_fail :: Integer)

    h = diff(t)
    k = ord_old
    hn = h[end]

    # we cannot increse order if there is not enough previous steps
    if length(h) <= k
        return(hn,k)
    end

    # if the order is lower than three, we cannot estimate the  increase it
    if ord_old < 3
        return(hn,ord_old+1)
    end

    l = size(y,1)

    # return if we didn't make enough steps to determine the new
    # order/step size
    if length(h) <= k+2
        return(hn,ord_old)
    end

    psi    = cumsum( h[end:-1:end-k-1] )

    phi = zeros(T,l,k+3)
    for j=1:l, i = 1:k+3
        phi[j,i] = prod(psi[1:i-1])*div_diff(h[end-i+1:end-1],y[j,end-i+1:end][:])
    end

    sigma  = [ hn^i*factorial(i-1)/prod(psi[1:i]) for i=1:k+2 ]

    terkm2 = norm((k-1)*sigma[k-1]*phi[:,k])
    terkm1 = norm(k*sigma[k]*phi[:,k+1])
    terk   = norm((k+1)*sigma[k+1]*phi[:,k+2])

    # determine the new method order
    seq = [terkm2,terkm1,terk]

    # if k+1 previous steps were of equal size we can estimate the
    # error of the higher order method
    if all( h[end-k-1:end] .== h[end-1] )
        terkp1=norm(phi[:,k+3])
        seq = [seq, terkp1]
    end

    difs = diff(seq)

    # sequence is decreasing => increase order
    if all(difs .< 0)
        ordn = min(ord_old+1,MAXORDER)
        # increasing => decrease order
    elseif all(difs .> 0)
        ordn = ord_old-1
    else
        # otherwise => don't change order
        ordn = ord_old
    end

    # Estimate of the error for the new order.  Order increment can be
    # positive or negative, we choose one of the error estimates
    # [terkm2,terkm1,terk,terkp1 (if available)] and base the new step
    # size on that estimate
    ord_incr=(ordn-ord_old)
    est=seq[min(3+ord_incr,end)]/(ordn+1)

    # determine the new step size based on the estimate est for the
    # new order
    r=(2.0*est)^(-1/(ordn+1))

    # based on whether the previous steps were successful we determine
    # the new step size
    #
    # num_fail is the number of steps that failed before this step
    if num_fail == 0
        if r >= 2.0
            hn = 2*hn
        elseif r < 1.0
            r = max(0.5,min(r,0.9))
            hn = r*hn
        end
    elseif num_fail == 1
        r=r*0.9
        hn = max(0.25,min(r,0.9))
    elseif num_fail >= 2
        hn = 3/4*hn
    end

    return(hn,ordn)
end


# h=[         h_{n-k+1},    ...   ,                h_{n},       h_{n+1}] --- length = k+1
# y=[y_{n-k},            y_{n-k+1}, ... , y_{n-1},        y_{n}        ] --- length = k+1
# k is the order of BDF
# for reference see p. 119
function stepper!{T<:Number}(t      :: AbstractArray{T,1},
                             y      :: AbstractArray{T,2},
                             h_next :: T,
                             F      :: Function,
                             stuff  :: StepperStuff{T},
                             wt     :: AbstractArray{T,1})

    ord      = length(t)        # this is the true order of BDF method
    l        = size(y,1)        # the number of dependent variables
    h        = [diff(t), h_next]

    # sanity check
    # @todo remove it in final version
    if size(y,2) != length(t)
        error("Incompatible size of h and y")
        return(-1)
    elseif ndims(y) != 2
        error("ndims(y) != 2")
        return(-1)
    end

    # check whether order is between 1 and 6, for orders higher than 6
    # BDF does not converge
    if ord < 1 || ord > 6

        error("Order ord=$(ord) should be [1,2,...,6]")
        return(-1)

    end

    t_next   = t[end]+h_next

    # we use predictor to obtain the starting point for the modified
    # newton method
    (y0,dy0,alpha)=predictor(y,h)

    # I think there is an error in the book, the sum should be taken
    # from j=1 to k+1 instead of j=1 to k
    alphas = -sum([1/j for j=1:ord])

    a=-alphas/h_next
    b=dy0-a*y0

    # delta for approximation of jacobian.  I removed the
    # sign(h_next*dy0) from the definition of delta because it was
    # causing trouble when dy0==0 (which happens for ord==1)
    delta = sqrt(eps(T))*float([ max(abs(y0[j]),
                                     abs(h_next*dy0[j]),
                                     wt[j])
                                for j=1:l])

    # This is a sanity check, if delta is zero we can't continue, this
    # shouldn't happen though, so this test should be removed in the
    # final version
    if any(delta.==0)
        error("delta==0")
    end

    # f_newton is supplied to the modified Newton method.  Zeroes of
    # f_newton give the corrected value of the next step "yc"
    f_newton(yc)=F(t_next,yc,a*yc+b)

    # if called, this function computes the jacobian of f_newton at
    # the point y0 via first order finite differences
    g_new()=G(f_newton,y0,delta)

    # this is the updated value of coefficient a, if jacobian is
    # udpated, corrector will replace stuff.a with a_new
    a_new=a

    # the norm used to test convergence of the newton method.  The
    # norm depends on the solution y0 through the weights wt.
    norm(v)=dassl_norm(v,wt)

    # we compute the corrected value "yc", updating the gradient if necessary
    (status,yc)=corrector(stuff,    # old coefficient a and jacobian
                          a_new,    # current coefficient a
                          g_new,    # this function is called when new jacobian is needed
                          y0,       # starting point for modified newton
                          f_newton, # we want to find zeroes of this function
                          norm)     # the norm used to estimate error

    # alpha0 is needed to estimate error
    alpha0   =-sum(alpha[1:ord-1])

    # @todo I don't know if this error estimate still holds for
    # backwards Euler (when ord==1)
    M   = max(alpha[ord],abs(alpha[ord]+alphas-alpha0))
    err = norm(yc-y0)*M

    # status<0 means the modified Newton method did not converge
    # err is the local error estimate from taking the step
    # yc is the estimated value at the next step
    return (status, err, yc)

end


# returns the corrected value yc and status.  If needed it updates
# the jacobian g_old and a_old.

function corrector{T<:Number}(stuff::StepperStuff{T},
                              a_new::T,
                              g_new::Function,
                              y0::AbstractArray{T,1},
                              f_newton::Function,
                              norm::Function)

    # if a_old == 0 the new jacobian is always computed, independently
    # of the value of a_new
    if abs((stuff.a-a_new)/(stuff.a+a_new)) > 1/4
        # old jacobian wouldn't give fast enough convergence, we have
        # to compute a current jacobian
        stuff.g=g_new()
        stuff.evals += 1
        stuff.a=a_new
        # run the corrector
        (status,yc)=newton_iteration( x->(-stuff.g\f_newton(x)), y0, norm)
    else
        # old jacobian should give reasonable convergence
        c=2*stuff.a/(a_new+stuff.a)     # factor "c" is used to speed up
                                    # the convergence when using an
                                    # old jacobian
        # reusing the old jacobian
        (status,yc)=newton_iteration( x->(-c*(stuff.g\f_newton(x))), y0, norm )

        if status < 0
            # the corrector did not converge, so we recompute jacobian and try again
            stuff.g=g_new()
            stuff.evals += 1
            stuff.a=a_new
            # run the corrector again
            (status,yc)=newton_iteration( x->(-stuff.g\f_newton(x)), y0, norm )
        end
    end

    return (status,yc)

end

# this function iterates f until it finds its fixed point, starting
# from f(y0).  The result either satisfies norm(yn-f(yn))=0+... or is
# set back to y0.  Status tells if the fixed point was obtained
# (status==0) or not (status==-1).
function newton_iteration{T<:Number}(f::Function,y0::AbstractArray{T,1},norm::Function)

    # first guess comes from the predictor method, then we compute the
    # second guess to get the norm1

    delta=f(y0)
    norm1=norm(delta)
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
        normn=norm(delta)
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


function predictor{T<:Number}(y::AbstractArray{T,2},h::AbstractArray{T,1})
    k        = length(h)-1      # k from the book is _not_ the order
                                # of the BDF method
    ord      = k+1              # this is the true order of BDF method
    l        = size(y,1)        # the number of dependent variables

    # these parameters are used by the predictor method
    psi      = cumsum(h[end:-1:1])
    alpha    = h[end]./psi

    phi_star = zeros(T,l,ord)
    for j=1:l, i = 1:ord
        phi_star[j,i] = prod(psi[1:i-1])*div_diff(h[end-i+1:end-1],y[j,end-i+1:end][:])
    end

    gamma    = cumsum( [i==1 ? zero(T) : alpha[i-1]/h[end] for i=1:k+1] )

    y0       = sum(phi_star,2)[:,1]
    dy0      = sum([gamma[i]*phi_star[j,i] for j=1:l, i=1:k+1],2)[:,1]

    # alpha is neede by the stepper method to estimate error
    return(y0,dy0,alpha)
end

# h=[h_{n-k}, ... , h_{n-1}]
# y=[y_{n-k}, ... , y_{n-1}, y_{n}]
function div_diff{T<:Number}(h::AbstractArray{T,1},y::AbstractArray{T,1})
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

function dassl_norm{T<:Number}(v::AbstractArray{T,1},wt::AbstractArray{T,1})
    norm(v./wt)/sqrt(length(v))
end

# compute the G matrix from dassl (jacobian of F(t,x,a*x+b))
# @todo replace with symmetric finite difference?
function G{T<:Number}(f::Function,y0::AbstractArray{T,1},delta::AbstractArray{T,1})
    n=length(y0)
    edelta=diagm(delta)
    s=Array(eltype(delta),n,n)
    for i=1:n
        s[:,i]=(f(y0+edelta[:,i])-f(y0))/delta[i]
    end
    return(s)
end

end
