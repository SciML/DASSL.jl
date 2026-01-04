using DiffEqBase, DASSL
using DAEProblemLibrary: prob_dae_resrob

prob = prob_dae_resrob

sol = solve(prob, dassl())

sol = solve(prob, dassl(), abstol = 1.0e-1, reltol = 1.0e-2)
