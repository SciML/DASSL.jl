# TODO:
# - don't compute jacobian in every step
# - step size and order selection
#

module dassl

export stepper

# h=[         h_{n-k+1},    ...   ,                h_{n},       h_{n+1}] --- length = k+1
# y=[y_{n-k},            y_{n-k+1}, ... , y_{n-1},        y_{n}        ] --- length = k+1
# k is the order of BDF
# for reference see p. 119
function stepper(t,y,h,F,g_old,a_old,rtol,atol)
    if size(y,2) != length(h)
        error("Incompatible size of h and y")
        return(-1)
    end

    k        = length(h)-1      # k from the book is _not_ the order
                                # of the BDF method
    ord      = k+1              # this is the true order of BDF method
    l        = size(y,1)
    T        = eltype(y)
    h_next=h[end]
    t_next=t+h_next

    # this stepper does not work for order 1 bdf method, when
    # predictor is undefined
    if ord < 1 | ord > 6

        error("Order ord=$(ord) should be [1,2,...,6]")
        return(-1)

    end

    psi      = cumsum(h[end:-1:1])
    alpha    = h[end]/psi
    alpha0   =-sum(alpha[1:k])
    alphas   =-sum([1/j for j=1:k])

    if ord > 1                  # use fixed leading coefficient BDF
        phi_star   = float([ reduce(*,psi[1:i-1])*div_diff(h[end-i+1:end-1],y[j,end-i+1:end]) for j=1:l, i=1:k+1 ])
        gamma    = cumsum( [i==1 ? zero(T) : alpha[i-1]/h[end] for i=1:k+1] )
        (y0,dy0)=predictor(phi_star,gamma)
        a=-alphas/h_next
        b=dy0-a*y0

        # delta for approximation of jacobian
        delta = sqrt(eps(t))*float([ sign(h_next*dy0[j])*max(abs(y0[j]),
                                                             abs(h_next*dy0[j]),
                                                             (rtol*y0+atol)[j])
                                    for j=1:l])

    elseif ord == 1             # use Backwards Newton
        a=1/h[end]
        b=-y[:,end]/h[end]
        y0=y

        # delta for approximation of jacobian
        # @todo I don't know if it works
        delta = sqrt(eps(t))*float([ max(abs(y0[j]),(rtol*y0+atol)[j])
                                    for j=1:l])
    end

    # this function is supplied to the modified Newton method
    f_newton(x)=F(t_next,x,a*x+b)

    # if called, this function computes the current jacobian (G-function)
    g_new()=G(ed->f_newton(y0+ed),delta)
    a_new=a

    # the norm used to test convergence of the newton method
    norm(v)=dassl_norm(v,y0,rtol,atol)

    # we compute the corrected value "yc", recomputing the gradient if necessary
    (status,yc)=corrector_wrapper!(a_old, # old coefficient a
                                   a_new, # current coefficient a
                                   g_old, # old jacobian
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

function corrector_wrapper!(a_new,a_old,g_old,g_new::Function,y0,f_newton,norm)

    if abs((a_old-a_new)/(a_old+a_new)) > 1/4
        # old jacobian wouldn't give fast enough convergence, we have
        # to compute a current jacobian
        g_old=g_new()
        a_old=a_new
        # run the corrector
        (status,yc)=corrector( x->(-g_old\f_newton(x)), y0, norm)
    else
        # old jacobian should give reasonable convergence
        c=2*a_old/(a_new+a_old)     # factor "c" is used to speed up
                                    # the convergence when using an
                                    # old jacobian
        # reusing the old jacobian
        (status,yc)=corrector( x->(-c*(g_old\f_newton(x))), y0, norm )

        if status < 0
            # the corrector did not converge, so we recompute jacobian and try again
            g_old=g_new()
            a_old=a_new
            # run the corrector again
            (status,yc)=corrector( x->(-g_old\f_newton(x)), y0, norm )
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

            info("Unable to converge after $(i) iterations, rho=$(rho)")
            return(status,y0)
        end

        err=rho/(1-rho)*normn

        # iteration converged successfully

        if err < 1/3
            info("Converged after i=" * string(i) * " iterations with estimated error=" * string(err))
            status=0
            return(status,yn)
        end

    end

    # unable to converge after 4 iterations

    info("Unable to converge after 4 iterations")

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
function G(f,delta)
    info("Recalculating the jacobian")
    n=length(delta)
    edelta=diagm(delta)
    s=Array(eltype(delta),n,n)
    for i=1:n
        s[:,i]=(f(edelta[:,i])-f(0))/delta[i]
    end
    return(s)
end

end
