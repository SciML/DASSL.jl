dassl.jl
========

A work in progress on implementing DASSL solver in Julia.

Usage example:

```
using dassl
F(t,y,dy)=dy+y     # the equation solved is F(t,y,dy)=0
y0=[1.0]           # the initial value
tspan=[0.0,10.0]   # time span over which we integrate
driver(F,y0,tspan) # returns (tn,yn)
```

TODO
----

- Step size/order adaptation is reluctant to switch to higher order
  methods.

- API doesn not confirm to the ODE.jl package

- Tests

- Still there are relatively large errors

- Performance
