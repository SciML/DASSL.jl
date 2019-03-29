DASSL.jl
========

[![Build Status](https://travis-ci.org/JuliaDiffEq/DASSL.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/DASSL.jl)
[![Coverage Status](https://img.shields.io/coveralls/pwl/DASSL.jl.svg)](https://coveralls.io/r/pwl/DASSL.jl)

This is an implementation of DASSL algorithm for solving algebraic
differential equations.  To install a stable version run

```
Pkg.add("DASSL")
```

Common Interface Example
------------------------

This package is compatible with the JuliaDiffEq common solver interface which is documented in the [DifferentialEquations.jl documentation](http://docs.juliadiffeq.org/latest/). Following the [DAE Tutorial](http://docs.juliadiffeq.org/latest/tutorials/dae_example.html), one can use `dassl()` as follows:

```julia
using DASSL
u0 = [1.0, 0, 0]
du0 = [-0.04, 0.04, 0.0]
tspan = (0.0,100000.0)

function resrob(r,yp,y,p,t)
    r[1]  = -0.04*y[1] + 1.0e4*y[2]*y[3]
    r[2]  = -r[1] - 3.0e7*y[2]*y[2] - yp[2]
    r[1] -=  yp[1]
    r[3]  =  y[1] + y[2] + y[3] - 1.0
end

prob = DAEProblem(resrob,du0,u0,tspan)  
sol = solve(prob, dassl())
```

For more details on using this interface, [see the ODE tutorial](http://docs.juliadiffeq.org/latest/tutorials/ode_example.html).

Examples
--------

To solve a scalar equation `y'(t)+y(t)=0` with initial data `y(0)=0.0` up to time `t=10.0` run the following code

```
using DASSL
F(t,y,dy) = dy+y                   # the equation solved is F(t,y,dy)=0
y0        = 1.0                    # the initial value
tspan     = [0.0,10.0]             # time span over which we integrate
(tn,yn)   = dasslSolve(F,y0,tspan) # returns (tn,yn)
```

You can also change the relative error tolerance `rtol`, absolute
error tolerance `atol` as well as initial step size `h0` as follows

```
(tn,yn)   = dasslSolve(F,y0,tspan)
```

To test the convergence and execution time for index-1 problem run
`convergence.jl` from the `test` directory.

Naturally, DASSL.jl also supports multiple equations.  For example the
pendulum equation

```
u'-v=0
v'+sin(u)=0
```

with initial data `u(0)=0.0` and `v(0)=1.0` can be solved by defining
the following residual function

```
function F(t,y,dy)
       [
       dy[1]-y[2],           #  y[1]=u,   y[2]=v
       dy[2]+sin(y[1])       # dy[1]=u', dy[2]=v'
       ]
end
```

The initial data shoud now be set as a vector

```
y0      = [0.0,1.0]           # y0=[u(0),v(0)]
```

The solution can be computed by calling

```
tspan   = [0.0,10.0]
(tn,yn) = dasslSolve(F,y0,tspan)
```

Output
------

Apart from producing the times `tn` and values `yn`, dasslSolve also
produces the derivatives `dyn` (as the byproduct of BDF
algorithm), e.g.

```
(tn,yn,dyn) = dasslSolve(F,y0,tspan)
```

The decision to produce these values is that it is not entirely
trivial to compute `y'` from `F(t,y,y')=0` when `t` and `y` are given.

Keyword arguments
-----------------

DASSL supports a number of keyword arguments, the names of most of
them are compatible with the namse used in ODE package.

- `reltol=1e-3`/`abstol=1e-5` set the relative/absolute local error tolerances

- `initstep=1e-4`/`minstep=0`/`maxstep=Inf` set the
  initial/minimal/maximal step sizes (when step size drops below
  minimum the integration stops)

- `jacobian` The most expensive step during the integration is solving
  the nonlinear equation `F(t,y,a*y+b)=0` via Newton's method, which
  requires a jacobian of the form `dF/dy+a*dF/dy'`.  By default, the
  solver approximates this Jacobian by a method of finite differences
  but you can provide your own method as a function
  `(t,y,dy,a)->dF/dy+a*dF/dy'`.  For the pendulum equation we would
  define jacobian as

  ```
  jacobian=(t,y,dy,a)->[[a,cos(y[1])] [-1,a]]
  ```

- `maxorder=6` Apart from selecting the current step size DASSL method
  can also dynamically change the order of BDF method used.  BDF is
  stable up to 6-th order, which is the defaul upper limit but for
  some systems of equations it may make more sense to use lower
  orders.

- `dy0=zero(y)` When solving differential algebraic equations it is
  important to start with consistent initial conditions, i.e. to
  choose `y` and `y'` such that `F(t,y,y')=0` initially.  DASSL tries
  to guess the initial value of `y'`, but if it fails you can set your
  own initial condtions for the derivative.

- `norm=dassl_norm`/`weights=dassl_weights` DASSL computes the error
  roughly as `err=norm(yc-y0)`, and accepting the step when
  `err<1`.  The local error tolerances `reltol` and `abstol` are
  hidden in the definition of `dassl_norm(v,
  wt)=norm(v./wt)/sqrt(length(v))`, where weights `wt` are defined by
  `dassl_weights(y,reltol,abstol)=reltol*abs(y).+abstol`.  You can
  supply your own weights and norms when they are more appropriate for
  the problem at hand.

- `factorize_jacobian=true` is a Boolean option which forces the
  factorization of Jacobian before storing it.  It dramatically
  increases performance for large systems, but may decrease the
  computation speed for small systems.


Iterator version
----------------

DASSL.jl supports an iterative version of solver (implemented via
coroutines, so debugging might be a little tricky) via
`dasslIterator`.  In the following example the `dasslIterator` is used
to stop the integration when the solution `y` drops below `0.1`


```
F(t,y,dy)=dy+y

# iterator version of dassl solver
for (t,y,dy) in dasslIterator(F,1.0,0.0)
    if y < 0.1
        @show (t,y,dy)
        break
    end
end
```
