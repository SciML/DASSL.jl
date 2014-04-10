dassl.jl
========
A work in progress on implementing DASSL solver in Julia.

To run run the test case load the module and run

```
F(t,y,dy)=dy+y # the equation solved is F(t,y,dy)=0
y0=[1.0] # the initial value
tspan=[0.0,10.0] # time span over which we integrate
driver(F,y0,tspan) # returns plenty of intermediate info, at this point the structure of output is tailored mostly for debugging
```

So far it can adapt the order and step size but integration gives relatively large errors of unknown (at least to me) origin.
At this point I am working on verifying the algorithm without focusing on performance.

I would appreciate any advice/help.
