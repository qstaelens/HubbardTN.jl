println("""
##############
#  One-Band  #
##############
""")

tol = 1e-2

E_ref = -1.2713317702997016

@testset "Hubbard–Holstein: g = 0, ω₀ > 0" begin

    symm = SymmetryConfig(Trivial, U1Irrep, 2)
    w = 1.0
    g = 0.0
    max_b = 4

    model = HolsteinParams([2.0, 1.0], [4.0], w, g, max_b)
    calc  = CalcConfig(symm, model)

    gs = compute_groundstate(calc)
    ψ  = gs["groundstate"]
    H  = gs["ham"]

    # energy check
    E0 = sum(real(expectation_value(ψ, H))) / length(ψ)
    @test E0 ≈ E_ref atol=tol

    # phonon occupation check
    Nb = density_b(ψ, symm, max_b)

    @test Nb[1] ≈ 0.0 atol=tol
    @test Nb[2] ≈ 0.0 atol=tol
end

E_ref = -3.2705801927593416

@testset "Hubbard–Holstein: g = 0, ω₀ < 0" begin

    symm = SymmetryConfig(Trivial, U1Irrep, 2)
    
    w = -1.0
    g = 0.0
    max_b = 4

    model = HolsteinParams([2.0, 1.0], [4.0], w, g, max_b)
    calc  = CalcConfig(symm, model)

    gs = compute_groundstate(calc)
    ψ  = gs["groundstate"]
    H  = gs["ham"]

    # energy check
    E0 = sum(real(expectation_value(ψ, H))) / length(ψ)
    @test E0 ≈ E_ref atol=tol

    # phonon occupation check
    Nb = density_b(ψ, symm, max_b)

    @test Nb[1] ≈ max_b atol=tol
    @test Nb[2] ≈ max_b atol=tol
end

E_ref = -2.038990604938512

@testset "Hubbard–Holstein: g > 0, ω₀ = 1.0" begin

    symm = SymmetryConfig(Trivial, U1Irrep, 2)
    
    w = 1.0
    g = 2.0
    max_b = 10

    model = HolsteinParams([2.0, 1.0], [4.0], w, g, max_b)
    calc  = CalcConfig(symm, model)

    gs = compute_groundstate(calc; svalue = 3.0)
    ψ  = gs["groundstate"]
    H  = gs["ham"]

    # energy check
    E0 = sum(real(expectation_value(ψ, H))) / length(ψ)
    @test E0 ≈ E_ref atol=tol

    Ne = density_e_HH(ψ, symm)
    Nb = density_b(ψ, symm, max_b)

    # phonons are activated
    @test Nb[1] > 0.0
    @test Nb[2] > 0.0

    # electronic density is non-uniform
    @test Ne[1] != Ne[2]
end