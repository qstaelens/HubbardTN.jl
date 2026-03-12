println("""
##############
#  Holstein  #
##############
""")

tol = 1e-2

E_ref = -1.2713317702997016

@testset "Hubbard–Holstein: g = 0, ω₀ > 0" begin
    symm = SymmetryConfig(Trivial, U1Irrep, 2)
    w = [1.0]
    g = [0.0;;]
    max_b = 4

    model = HubbardParams([2.0, 1.0], [4.0])
    calc  = CalcConfig(symm, model, HolsteinTerm(w, g, max_b, 1.0))

    gs = compute_groundstate(calc)
    ψ  = gs["groundstate"]
    H  = gs["ham"]

    # energy check
    E0 = sum(real(expectation_value(ψ, H))) / length(ψ)
    @test E0 ≈ E_ref atol=tol

    Ne = density_e(ψ, calc)
    Nb = density_b(ψ, calc)

    @test Nb[1] ≈ 0.0 atol=tol
    @test Nb[2] ≈ 0.0 atol=tol

    @test Ne[1] ≈ Ne[2] atol=tol
end

E_ref = -3.2705801927593416

@testset "Hubbard–Holstein: g = 0, ω₀ < 0" begin
    symm = SymmetryConfig(Trivial, U1Irrep, 2)
    w = [-1.0]
    g = [0.0;;]
    max_b = 4

    model = HubbardParams([2.0, 1.0], [4.0])
    calc  = CalcConfig(symm, model, HolsteinTerm(w, g, max_b, 1.0))

    gs = compute_groundstate(calc)
    ψ  = gs["groundstate"]
    H  = gs["ham"]

    # energy check
    E0 = sum(real(expectation_value(ψ, H))) / length(ψ)
    @test E0 ≈ E_ref atol=tol

    Ne = density_e(ψ, calc)
    Nb = density_b(ψ, calc)

    @test Nb[1] ≈ max_b atol=tol
    @test Nb[2] ≈ max_b atol=tol

    @test Ne[1] ≈ Ne[2] atol=tol
end

E_ref = -2.038990604938512

@testset "Hubbard–Holstein: g > 0, ω₀ = 1.0" begin
    symm = SymmetryConfig(Trivial, U1Irrep, 2)
    
    w = [1.0]
    g = [2.0;;]
    max_b = 10

    model = HubbardParams([2.0, 1.0], [4.0])
    calc  = CalcConfig(symm, model, HolsteinTerm(w, g, max_b, 1.0))

    gs = compute_groundstate(calc; svalue=3.0)
    ψ  = gs["groundstate"]
    H  = gs["ham"]

    # energy check
    E0 = sum(real(expectation_value(ψ, H))) / length(ψ)
    @test E0 ≈ E_ref atol=tol

    Ne = density_e(ψ, calc)
    Nb = density_b(ψ, calc)

    # phonons are present
    @test Nb[1] > 0.0
    @test Nb[2] > 0.0

    # electronic density is non-uniform
    @test Ne[1] != Ne[2]
end

E_ref = -0.895169445564952

@testset "Hubbard–Holstein: 2 phonons for 1 band" begin
    symm = SymmetryConfig(Trivial, U1Irrep, 2)
    
    w = [1.0, 1.0]
    g = [1.0 1.0;]
    max_b = 5

    model = HubbardParams([2.0, 1.0], [4.0])
    calc  = CalcConfig(symm, model, HolsteinTerm(w, g, max_b, 1.0))

    gs = compute_groundstate(calc; svalue=2.5)
    ψ  = gs["groundstate"]
    H  = gs["ham"]

    # energy check
    E0 = sum(real(expectation_value(ψ, H))) / length(ψ)
    @test E0 ≈ E_ref atol=tol
    
    Ne = density_e(ψ, calc)
    Nb = density_b(ψ, calc)

    # phonons are present
    @test Nb[1] > 0.0
    @test Nb[2] > 0.0

    # electronic density is non-uniform
    @test Ne[1] != Ne[2]
end

E_ref = -1.717276517157937

@testset "Hubbard–Holstein: 1 phonon for 2 bands" begin
    symm = SymmetryConfig(Trivial, U1Irrep, 2)
    
    w = [1.0]
    g = [0.5; 0.5;;]
    max_b = 5

    t = Dict(
    (1,1) => 2.0, (2,2) => 2.0,
    (1,2) => 1.0, (2,1) => 1.0,
    (2,3) => 1.0, (3,2) => 1.0,
    )

    U = Dict(
    (1,1,1,1) => 4.0,
    (2,2,2,2) => 4.0,
    )

    model = HubbardParams(2, t, U)
    calc  = CalcConfig(symm, model, HolsteinTerm(w, g, max_b, 1.0))

    gs = compute_groundstate(calc; svalue=2.5)
    ψ  = gs["groundstate"]
    H  = gs["ham"]

    # energy check
    E0 = sum(real(expectation_value(ψ, H))) / length(ψ)
    @test E0 ≈ E_ref atol=tol

    Ne = density_e(ψ, calc)
    Nb = density_b(ψ, calc)

    @test Nb[1] ≈ 0.0 atol=tol
    @test Nb[2] ≈ 0.0 atol=tol

    @test Ne[1] ≈ Ne[2] atol=tol
end