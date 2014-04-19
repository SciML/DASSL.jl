using dassl

# number of previously known steps
const ord=2
const dh=1.e-1
const rtol=0.1
const atol=0.1

function F(t,y,dy)
    return(dy-y)
end

function sol(t)
    exp(t)
    # 1/(1-t)
end

# t=float([j*(j+1)/2*dh for j=0:ord])
# t=float([j*dh for j=0:ord])
const t=float([0,1,3])*dh
# y=hcat(sol(t[1:end-1])...)
const y=[sol(t[j]) for i=1:1, j=1:ord]
const h=diff(t)
const y0=y[:,end]
const wt=rtol*abs(y0).+atol
const ag=AG(1.0/dh,eye(1))      # some arbitrary jacobian for testing

(status,err,yn)=stepper!(t[end],y,h,F,ag,wt)

# for i = 1 : (1.0/dh)-ord+1
#     (status,err,yn)=stepper!(t[end],y[:,end-ord+1:end],h[end-ord+1:end],F,ag,wt)
#     y=[y yn]
#     t=[t, t[end]+h[end]]
# end
