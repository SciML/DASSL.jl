using ExplicitImports
using DASSL
using Test

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(DASSL) === nothing
    @test check_no_stale_explicit_imports(DASSL) === nothing
end
