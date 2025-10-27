println("
################
#  Three-Body  #
################
")

tol = 1e-2


############
# One-band #
############

E_norm = -0.32384

@testset "Three-body interaction terms" begin
    symm = SymmetryConfig(U1Irrep, SU2Irrep, 2, (1,1))
    t = [0.0, 1.0]
    U = [6.0]
    V = [2.0]
    model = ModelParams(t, U, V)
    calc = CalcConfig(symm, model)
    gs = compute_groundstate(calc; tol=tol/10)
    ψ₀ = gs["groundstate"]
    H = gs["ham"]
    E0 = expectation_value(ψ₀, H)/length(ψ₀)
    @test real(E0) ≈ E_norm atol=tol
end