using dassl

# number of previously known steps
k=5
dh=0.1
rtol=0.001
atol=0.001

function F(t,y,dy)
    return(dy-y)
end

function sol(t)
    exp(t)
end

t=float([j*dh for j=0:k])
y=hcat(map(sol,t)...)
h=[diff(t), dh]
y0=y[:,end]
g=eye(1)
a_old=float(1)
