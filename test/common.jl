using DiffEqProblemLibrary, DiffEqBase, DASSL
using DiffEqProblemLibrary.DAEProblemLibrary: importdaeproblems; importdaeproblems()
using DiffEqProblemLibrary.DAEProblemLibrary: prob_dae_resrob

prob = prob_dae_resrob

sol = solve(prob,dassl())

sol = solve(prob,dassl(),abstol = 1e-1, reltol = 1e-2)
