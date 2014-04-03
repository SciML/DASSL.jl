module dassl

export modified_newton, predictor, dummy

function dummy(t,y,h,F,rtol,atol)
    k=length(h)-1

    alphas=-sum([1/j for j=1:k])
    t_next=t+h[end]
    (y0,dy0)=predictor(h,y)

    F_y =jacobian(x->F(t_next, x,dy0), y0)
    F_dy=jacobian(x->F(t_next,y0,  x),dy0)

    a=-alphas/h[end]
    b=dy0-a*y0
    jac=(a*F_dy+F_y)

    (status,yn)=modified_newton( x->(-jac\F(t_next,x,a*x+b)), y0, v->dassl_norm(v,y0,rtol,atol) )
end


function modified_newton(f,y0,norm)

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
            info("Converged after i=" * string(i) * " iterations with estimated error=" * string(err))
            status=0
            return(status,yn)
        end

    end

    # unable to converge after 4 iterations

    status=-1
    return(status,y0)
end


# h=[         h_{n-k+1},    ...   ,                h_{n},       h_{n+1}] --- length = k+1
# y=[y_{n-k},            y_{n-k+1}, ... , y_{n-1},        y_{n}        ] --- length = k+1
# k is the order of BDF
# size(h)=k+2, size(y)=k+1
# for reference see p. 119
function predictor(h,y)
    if size(y,2) != length(h)
        error("Incompatible size of h and y")
        return(0)
    end

    k        = length(h)-1

    psi      = cumsum(h[end:-1:2])
    psi_old  = cumsum(h[end-1:-1:1])
    alpha    = h[end]/psi
    beta     = vcat(1,cumprod(psi[1:end-1]./psi_old[1:end-1]))
    phi_star = [ beta[1]*y[:,end] hcat([beta[i]*prod(psi_old[1:i-1])*divided_differences(h[end-i+1:end-1],y[:,end-i+1:end]) for i=2:k]...)]
    g        = cumsum( [i==1 ? zero(eltype(y)) : alpha[i-1]/h[end] for i=1:k] )
    alpha0   =-sum(alpha[1:k])
    y0       = sum(phi_star,2)[:,1]
    dy0      = sum(hcat([g[i]*phi_star[:,i] for i=1:k]...),2)[:,1]
    return(y0,dy0)
end


# h=[h_{n-k}, ... , h_{n-1}]
# y=[y_{n-k}, ... , y_{n-1}, y_{n}]
function divided_differences(h,y)
    if size(y,2) == 2
        return((y[:,2]-y[:,1])/h[1])
    else
        return((divided_differences(h[2:end],y[:,2:end])-divided_differences(h[1:end-1],y[:,1:end-1]))/sum(h))
    end
end


function dassl_norm(v,y,rtol,atol)
    norm(v/(rtol*abs(y)+atol))/sqrt(length(v))
end


# needs some improvement
function jacobian{T<:Number}(f::Function,x::Array{T,1})
    epsilon=sqrt(eps(T))
    n=length(x)
    eyeps=epsilon*eye(n)
    s=Array(T,n,n)
    for i=1:n
        s[:,i]=((f(x+eyeps[:,i])-f(x-eyeps[:,i]))/2epsilon)
    end
    return s
end

end
