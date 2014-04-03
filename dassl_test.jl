using dassl

# number of previously known steps
k=5
dh=0.1

t=float([j*dh for j=0:k])
y=hcat(map(x->[exp(x)],t)...)
h=[diff(t), dh]

function F(t,y,dy)
    return(dy-y)
end
