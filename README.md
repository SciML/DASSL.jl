dassl.jl
========

An implementation of DASSL algorithm for solving algebraic differential equations.

Usage example:

```
using dassl
F(t,y,dy)=dy+y     # the equation solved is F(t,y,dy)=0
y0=[1.0]           # the initial value
tspan=[0.0,10.0]   # time span over which we integrate
dasslSolve(F,y0,tspan) # returns (tn,yn)
```

You can also change the relative error tolerance `rtol`, absolute
error tolerance `atol` as well as initial step size `h0` as follows

```
dasslSolve(F,y0,tspan,rtol=10.0^-3,atol=10.0^-5,h0=10.0^-4) # returns (tn,yn)
```
