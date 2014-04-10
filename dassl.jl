module dassl

export driver

const MAXORDER = 6

type AG
    a
    g
end

function driver(F,y0,tspan; rtol = 1.0e-3, atol = 1.0e-3, h0 = 0.01)

    t_start = tspan[1]
    t_stop  = tspan[end]

    T       = eltype(y0)
    epsilon = eps(one(T))

    g = zeros(T,length(y0),length(y0))
    a = zero(T)
    ag=AG(a,g)

    # make the next step with DASSL method

    ord = [1]
    t   = [t_start]
    h   = [h0]
    y   = hcat(y0)
    num_fail=0

    while t[end] < t_stop

        hmin = 4*epsilon*max(abs(t[end]),abs(t_stop))
        hn   = h[end]

        if hn < 2*hmin
            error("Stepsize too small, aborting")
            break
        end

        ordn=ord[end]

        (status,err,yn)=stepper!(t[end],y[:,end-ordn+1:end],h[end-ordn+1:end],F,ag,rtol,atol)

        # Newton iteration failed, reduce the step size and try again
        if status < 0
            warn("Newton iteration was unable to converge, reducing step size")
            h[end]=3/4*hn
            continue
        end

        # keep track of the number of failed steps
        if err > 1
            num_fail = num_fail+1
        end

        # determine the new step size and order
        (hn,ordn)=newStepOrder(h,[y yn],rtol,atol,ordn,num_fail)

        # reject the step and continue with new step size
        if err > 1
            warn("Step rejected: estimated error=$(err) is too large, h=$(h[end])")
            num_fail = num_fail+1
            h[end]   = hn
            ord[end] = ordn
            continue
        end

        # if length(h) > 100
        #     break
        # end

        num_fail=0              # reset the failure counter
        ord = [ord, ordn]
        h   = [h,   hn]
        t   = [t,   t[end]+h[end]]
        y   = [y   yn]

    end

    return(ord,h,t,y)

end

function newStepOrder(h,y,rtol,atol,ord_old,num_fail)

    k = ord_old
    hn = h[end]

    if num_fail >= 0            # don't decrease the step size
        hn = 3/4*hn
    end

    # we cannot increse order if there is not enough previous steps
    if length(h) <= k
        return(hn,ord_old)
    end

    # if the order is lower than three increase it
    if ord_old < 3
        return(hn,ord_old+1)
    end

    l = size(y,1)

    norm(v) = dassl_norm(v,y[:,end],rtol,atol)

    # @todo this assumption comes from the BoundsError coming from the
    # code below
    if length(h) <= k+1
        return(hn,ord_old)
    end

    # psi_i(n+1)
    psi    = cumsum( h[end:-1:end-k-1] )
    phi    = float([ reduce(*,psi[1:i-1])*div_diff(h[end-i+2:end],y[j,end-i+1:end]) for j=1:l, i=1:k+3 ])
    sigma  = [ hn^i*factorial(i-1)/prod(psi[1:i]) for i=1:k+2 ]

    terkm2 = norm((k-1)*sigma[k-1]*phi[:,k])
    terkm1 = norm(k*sigma[k]*phi[:,k+1])
    terk   = norm((k+1)*sigma[k+1]*phi[:,k+2])

    # determine the new method order
    seq = [terkm2,terkm1,terk]

    # if k+1 previous steps were of equal size we can estimate the
    # error of the higher order method
    if all( h[end-k-1:end] .== h[end-1] )
        terkp1=norm(phi[k+3])
        seq = [seq, terkp1]
    end

    difs = diff(seq)

    # if sequence is decreasing
    if all(difs .< 0)
        ordn = min(ord_old+1,MAXORDER)
    elseif all(difs .> 0)
        ordn = ord_old-1
    else
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
function stepper!(t,y,h,F,ag,rtol,atol)

    k        = length(h)-1      # k from the book is _not_ the order
                                # of the BDF method
    ord      = k+1              # this is the true order of BDF method

    if size(y,2) != length(h)
        error("Incompatible size of h and y")
        return(-1)
    end

    # check whether order is between 1 and 6, for orders higher than 6
    # BDF does not converge
    if ord < 1 || ord > 6

        error("Order ord=$(ord) should be [1,2,...,6]")
        return(-1)

    end

    l        = size(y,1)
    T        = eltype(y)
    h_next   = h[end]
    t_next   = t+h_next

    psi      = cumsum(h[end:-1:1])
    alpha    = h[end]/psi
    alpha0   =-sum(alpha[1:k])
    alphas   =-sum([1/j for j=1:k])

    if ord > 1                  # use fixed leading coefficient predictor/corrector BDF
        phi_star   = float([ reduce(*,psi[1:i-1])*div_diff(h[end-i+1:end-1],y[j,end-i+1:end]) for j=1:l, i=1:k+1 ])
        gamma    = cumsum( [i==1 ? zero(T) : alpha[i-1]/h[end] for i=1:k+1] )
        (y0,dy0)=predictor(phi_star,gamma)
        a=-alphas/h_next
        b=dy0-a*y0

        # delta for approximation of jacobian
        delta = sqrt(eps(T))*float([ sign(h_next*dy0[j])*max(abs(y0[j]),
                                                             abs(h_next*dy0[j]),
                                                             (rtol*y0+atol)[j])
                                    for j=1:l])

    elseif ord == 1             # use Backwards Newton
        a=1/h[end]
        b=-y[:,end]/h[end]
        y0=y[:,end]

        # delta for approximation of jacobian
        # @todo I don't know if it works
        delta = sqrt(eps(T))*float([ max(abs(y0[j]),(rtol*y0+atol)[j])
                                    for j=1:l])
    end

    # this function is supplied to the modified Newton method
    f_newton(x)=F(t_next,x,a*x+b)

    # if called, this function computes the current jacobian (G-function)
    g_new()=G(f_newton,y0,delta)
    a_new=a

    # the norm used to test convergence of the newton method
    norm(v)=dassl_norm(v,y0,rtol,atol)

    # we compute the corrected value "yc", recomputing the gradient if necessary
    (status,yc)=corrector_wrapper!(ag, # old coefficient a
                                   a_new, # current coefficient a
                                   g_new, # this function is called when new jacobian is needed
                                   y0, # starting point for modified newton
                                   f_newton, # we want to find zeroes of this function
                                   norm) # the norm used to estimate error

    # @todo I don't know if this error estimate still holds for
    # backwards Euler (when ord==1)
    M   = max(alpha[k+1],abs(alpha[k+1]+alphas-alpha0))
    err = dassl_norm(yc-y0,y0,rtol,atol)*M

    # status<0 means the modified Newton method did not converge
    # err is the local error estimate from taking the step
    # yc is the estimated value at the next step
    return (status, err, yc)

end


# returns the corrected value yc and status.  If needed it updates
# the jacobian g_old and a_old.

function corrector_wrapper!(ag,a_new,g_new::Function,y0,f_newton,norm)

    # if a_old == 0 the new jacobian is always computed, independently
    # of the value of a_new
    if abs((ag.a-a_new)/(ag.a+a_new)) > 1/4
        info("estimated convergence too slow")
        # old jacobian wouldn't give fast enough convergence, we have
        # to compute a current jacobian
        ag.g=g_new()
        ag.a=a_new
        # run the corrector
        (status,yc)=corrector( x->(-ag.g\f_newton(x)), y0, norm)
    else
        # old jacobian should give reasonable convergence
        c=2*ag.a/(a_new+ag.a)     # factor "c" is used to speed up
                                    # the convergence when using an
                                    # old jacobian
        # reusing the old jacobian
        (status,yc)=corrector( x->(-c*(ag.g\f_newton(x))), y0, norm )

        if status < 0
            info("Unable to converge using old jacobian")
            # the corrector did not converge, so we recompute jacobian and try again
            ag.g=g_new()
            ag.a=a_new
            # run the corrector again
            (status,yc)=corrector( x->(-ag.g\f_newton(x)), y0, norm )
        end
    end

    return (status,yc)

end

function corrector(f,y0,norm)

    # first guess comes from the predictor method, then we compute the
    # second guess to get the norm1

    delta=f(y0)
    norm1=norm(delta)
    yn=y0+delta

    # after the first iteration the norm turned out to be very small,
    # terminate and return the first correction step

    if norm1 < 10*eps(norm1)
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


function predictor(phi_star,gamma)
    k=length(gamma)
    l=size(phi_star,1)
    y0       = sum(phi_star,2)[:,1]
    dy0      = sum([gamma[i]*phi_star[j,i] for j=1:l, i=1:k],2)[:,1]
    return(y0,dy0)
end

# h=[h_{n-k}, ... , h_{n-1}]
# y=[y_{n-k}, ... , y_{n-1}, y_{n}]
function div_diff(h,y)
    if length(y) < 1
        error("length(y) is $(length(y)) in div_diff, should be >=1")
        return(zero(eltype(y)))
    elseif length(y) == 1
        return(y[1])
    elseif length(y) == 2
        return((y[2]-y[1])/h[1])
    else
        return((div_diff(h[2:end],y[2:end])-div_diff(h[1:end-1],y[1:end-1]))/sum(h))
    end
end

function dassl_norm(v,y,rtol,atol)
    norm(v/(rtol*abs(y)+atol))/sqrt(length(v))
end

# compute the G matrix from dassl
function G(f,y0,delta)
    info("Recalculating the jacobian")
    n=length(y0)
    edelta=diagm(delta)
    s=Array(eltype(delta),n,n)
    for i=1:n
        s[:,i]=(f(y0+edelta[:,i])-f(y0))/delta[i]
    end
    return(s)
end

end
