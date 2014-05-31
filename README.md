DASSL.jl
========

An implementation of DASSL algorithm for solving algebraic differential equations.

To inastall run

```
Pkg.add("DASSL")
```

Usage example:

```
using DASSL
F(t,y,dy) = dy+y                   # the equation solved is F(t,y,dy)=0
y0        = [1.0]                  # the initial value
tspan     = [0.0,10.0]             # time span over which we integrate
(tn,yn)   = dasslSolve(F,y0,tspan) # returns (tn,yn)
```

You can also change the relative error tolerance `rtol`, absolute
error tolerance `atol` as well as initial step size `h0` as follows

```
(tn,yn)   = dasslSolve(F,y0,tspan,rtol=10.0^-3,atol=10.0^-5,h0=10.0^-4)
```

You can test the convergence and execution time for index-1 problem by
running `convergence.jl` from the `test` directory.

DASSL.jl also supports multiple equations.  For example the
mathematical pendulum equation

```
u'-v=0
v'+sin(u)=0
```

can be solved by defining the following residual function

```
function F(t,y,dy)
       [
       dy[1]-y[2],
       dy[2]+sin(y[1])
       ]
end
```

The solution can be obtained by calling

```
tspan   = [0.0,10.0]
y0      = [0.0,1.0]           # u(0)=y0[1], v(0)=y0[2]
(tn,yn) = dasslSolve(F,y0,tspan)
```



[![Build Status](https://travis-ci.org/pwl/DASSL.jl.png)](https://travis-ci.org/pwl/DASSL.jl)
[![Coverage Status](https://img.shields.io/coveralls/pwl/DASSL.jl.svg)](https://coveralls.io/r/pwl/DASSL.jl)
[![Package Evaluator](http://iainnz.github.io/packages.julialang.org/badges/DASSL_0.2.svg)](http://iainnz.github.io/packages.julialang.org/?pkg=DASSL&ver=0.2)
