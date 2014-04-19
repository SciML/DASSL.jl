module dassl

export driver, stepper!, AG

const MAXORDER = 6

type AG
    a
    g
end

function driver(F,y0,tspan; rtol = 1.0e-3, atol = 1.0e-3, h0 = 1.0e-4)

    t_start = tspan[1]
    t_stop  = tspan[end]

    T       = eltype(y0)
    epsilon = eps(one(T))

    g = zeros(T,length(y0),length(y0))
    a = zero(T)
    ag=AG(a,g)

    ord = [1]
    t   = [t_start]
    h   = [h0]
    y   = hcat(y0)
    er  = [0.0]
    nf  = [0]
    num_fail=0

    while t[end] < t_stop

        hmin = 4*epsilon*max(abs(t[end]),abs(t_stop))
        hn   = h[end]
        ordn = ord[end]

        if hn < 2*hmin
            error("Stepsize too small, aborting")
            break
        end

        wt = rtol*abs(y[:,end]).+atol
        (status,err,yn)=stepper!(t[end],y[:,end-ordn+1:end],h[end-ordn+1:end],F,ag,wt)

        # Early failure: Newton iteration failed to converge, reduce
        # the step size and try again
        if status < 0
            warn("Newton iteration was unable to converge, reducing step size and trying again")
            h[end]=3/4*hn
            continue
        end

        if err > 1.0
            # step is rejected, change the current step size and order and try again
            warn("Step rejected: estimated error=$(err) is too large, h=$(h[end])")

            num_fail = num_fail+1
            # determine the new step size and order
            (hn,ordn)=newStepOrder(h,[y yn],wt,ordn,num_fail)
            # ordn=min(MAXORDER,length(h))
            h[end]   = hn
            ord[end] = ordn
            continue

        else
            # step is accepted

            num_fail=0              # reset the failure counter
            # determine the new step size and order
            (hn,ordn)=newStepOrder(h,[y yn],wt,ordn,num_fail)
            # ordn=min(MAXORDER,length(h))
            # save the results
            ord = [ord, ordn]
            h   = [h,   hn]
            t   = [t,   t[end]+hn]
            y   = [y   yn]
            er  = [er, err]
            nf  = [nf, num_fail]
        end

    end

    s=open("yn.dat", "w+")
    for i = 1 : size(y,2)
        @printf(s,"%.15f, %.15f, %.15f, %.15f, %i, %i\n",t[i], y[1,i], er[i], h[i], ord[i], nf[i])
    end
    close(s)

    return(ord,h,t,y,er)

end

function newStepOrder(h,y,wt,ord_old,num_fail)

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

    norm(v) = dassl_norm(v,wt)

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
            info("step increased, r=$r")
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
function stepper!{T<:Number}(t  ::T,
                             y  ::AbstractArray{T,2},
                             h  ::AbstractArray{T,1},
                             F  ::Function,
                             ag ::AG,
                             wt ::AbstractArray{T,1})

    # sanity check
    # @todo remove it in final version
    if size(y,2) != length(h)
        error("Incompatible size of h and y")
        return(-1)
    end

    k        = length(h)-1      # k from the book is _not_ the order
                                # of the BDF method
    ord      = k+1              # this is the true order of BDF method
    l        = size(y,1)        # the number of dependent variables

    # check whether order is between 1 and 6, for orders higher than 6
    # BDF does not converge
    if ord < 1 || ord > 6

        error("Order ord=$(ord) should be [1,2,...,6]")
        return(-1)

    end

    h_next   = h[end]
    t_next   = t+h_next

    #### relocate to the predictor?

    # these parameters are used by the predictor method
    psi      = cumsum(h[end:-1:1]) # ok
    alpha    = h[end]./psi         # ok

    phi_star = zeros(T,l,ord)   # ok
    for j=1:l, i = 1:ord
        phi_star[j,i] = prod(psi[1:i-1])*div_diff(h[end-i+1:end-1],y[j,end-i+1:end][:])
    end

    gamma    = cumsum( [i==1 ? zero(T) : alpha[i-1]/h[end] for i=1:k+1] ) # ok

    #### END relocate to the predictor

    # we use predictor to obtain the starting point for the modified
    # newton method
    (y0,dy0)=predictor(phi_star,gamma) # ok

    # I think there is an error in the book, the sum should be taken
    # from j=1 to k+1 instead of j=1 to k
    alphas = -sum([1/j for j=1:ord]) # ok

    a=-alphas/h_next            # ok
    b=dy0-a*y0                  # ok

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
    # the point y0
    g_new()=G(f_newton,y0,delta)

    # this is the updated value of coefficient a, if jacobian is
    # udpated, corrector will replace ag.a with a_new
    a_new=a

    # the norm used to test convergence of the newton method.  The
    # norm depends on the solution y0 through the weights wt.
    norm(v)=dassl_norm(v,wt)

    # we compute the corrected value "yc", updating the gradient if necessary
    (status,yc)=corrector(ag,       # old coefficient a and jacobian
                          a_new,    # current coefficient a
                          g_new,    # this function is called when new jacobian is needed
                          y0,       # starting point for modified newton
                          f_newton, # we want to find zeroes of this function
                          norm)     # the norm used to estimate error

    # alpha0 is needed to estimate error
    alpha0   =-sum(alpha[1:k])

    # @todo I don't know if this error estimate still holds for
    # backwards Euler (when ord==1)
    M   = max(alpha[k+1],abs(alpha[k+1]+alphas-alpha0))
    err = norm(yc-y0)*M

    # status<0 means the modified Newton method did not converge
    # err is the local error estimate from taking the step
    # yc is the estimated value at the next step
    return (status, err, yc)

end


# returns the corrected value yc and status.  If needed it updates
# the jacobian g_old and a_old.

function corrector{T<:Number}(ag::AG,a_new::T,g_new::Function,y0::AbstractArray{T,1},f_newton::Function,norm::Function)

    # if a_old == 0 the new jacobian is always computed, independently
    # of the value of a_new
    if abs((ag.a-a_new)/(ag.a+a_new)) > 1/4
        info("Estimated convergence too slow: a_old=$(ag.a), a_new=$a_new")
        # old jacobian wouldn't give fast enough convergence, we have
        # to compute a current jacobian
        ag.g=g_new()
        ag.a=a_new
        # run the corrector
        (status,yc)=modified_newton( x->(-ag.g\f_newton(x)), y0, norm)
    else
        # old jacobian should give reasonable convergence
        c=2*ag.a/(a_new+ag.a)     # factor "c" is used to speed up
                                    # the convergence when using an
                                    # old jacobian
        # reusing the old jacobian
        (status,yc)=modified_newton( x->(-c*(ag.g\f_newton(x))), y0, norm )

        if status < 0
            info("Unable to converge with old jacobian")
            # the corrector did not converge, so we recompute jacobian and try again
            ag.g=g_new()
            ag.a=a_new
            # run the corrector again
            (status,yc)=modified_newton( x->(-ag.g\f_newton(x)), y0, norm )
        end
    end

    return (status,yc)

end

function modified_newton{T<:Number}(f::Function,y0::AbstractArray{T,1},norm::Function)

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


function predictor{T<:Number}(phi_star::AbstractArray{T,2},gamma::AbstractArray{T,1})
    k=length(gamma)
    l=size(phi_star,1)
    y0       = sum(phi_star,2)[:,1]
    dy0      = sum([gamma[i]*phi_star[j,i] for j=1:l, i=1:k],2)[:,1]
    return(y0,dy0)
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
