using DiffEqProblemLibrary, DiffEqBase, DASSL

prob = prob_dae_resrob

sol = solve(prob,dassl())

sol = solve(prob,dassl(),abstol = 1e-1, reltol = 1e-2)
