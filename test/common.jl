using DiffEqBase, DASSL
using DAEProblemLibrary: prob_dae_resrob

prob = prob_dae_resrob

sol = solve(prob, dassl())

sol = solve(prob, dassl(), abstol = 1.0e-1, reltol = 1.0e-2)

# Out-of-place DAEProblem path: prob.f is the 4-arg residual (du, u, p, t). Regression
# test for the OOP wrapper, which previously called it as a 2-arg ODE RHS (t, u) and
# errored for every out-of-place DAE. Index-1 DAE: u1' = -u1, 0 = u1 - u2, so the exact
# solution is u1(t) = u2(t) = exp(-t).
@testset "Out-of-place DAEProblem" begin
    dae_oop(du, u, p, t) = [du[1] + u[1], u[1] - u[2]]
    u0 = [1.0, 1.0]
    du0 = [-1.0, -1.0]
    oop_prob = DAEProblem(dae_oop, du0, u0, (0.0, 1.0))
    @test !DiffEqBase.isinplace(oop_prob)   # confirm the OOP branch is exercised
    oop_sol = solve(oop_prob, dassl(), abstol = 1.0e-8, reltol = 1.0e-8)
    @test oop_sol.retcode == ReturnCode.Success
    @test isapprox(oop_sol.u[end][1], exp(-1); atol = 1.0e-5)
    @test isapprox(oop_sol.u[end][2], exp(-1); atol = 1.0e-5)   # algebraic constraint held
end
