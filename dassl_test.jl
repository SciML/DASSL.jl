using dassl

# number of previously known steps
ord=1
dh=0.00001
rtol=0.001
atol=0.001

function F(t,y,dy)
    return(dy-y)
end

function sol(t)
    exp(t)
end

t=float([j*dh for j=1:ord])
y=hcat(map(sol,t)...)
h=[diff(t), dh]
y0=y[:,end]
g_old=eye(1)                   # some arbitrary jacobian
a_old=float(1)
