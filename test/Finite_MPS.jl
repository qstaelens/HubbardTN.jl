using HubbardTN
using TensorKit, MPSKit

tol = 1e-2

s = 2.5
particle_symmetry = U1Irrep
spin_symmetry = U1Irrep

t = Dict((1,2)=>1.0, (2,1)=>1.0, (1,1)=>2.0)
U = Dict((1,1,1,1) => 4.0)
cell_width = 2
L = 12

@testset "Compare infinite with finite MPS" begin
    symm = SymmetryConfig(particle_symmetry, spin_symmetry, cell_width)
    model = HubbardParams(1, t, U)
    calc = CalcConfig(symm, model)
    gs = compute_groundstate(calc; svalue=s)
    ψ = gs["groundstate"]
    H = gs["ham"]

    E0 = expectation_value(ψ, H)
    Einf = sum(real(E0)) / length(H)

    symm = SymmetryConfig(particle_symmetry, spin_symmetry; length = L)
    calc = CalcConfig(symm, model)

    gs = compute_groundstate(calc; svalue=s)
    ψ = gs["groundstate"]
    H = gs["ham"]

    E0 = expectation_value(ψ, H)
    E = sum(real(E0)) / length(H)
    @test Einf ≈ E atol=tol

    dim = dim_state(ψ)

    Ne = density_e(ψ, calc)
    println("Number of electrons per site: ", Ne)
    @test Ne ≈ 1.0 atol=tol

    ent = entanglement_spectrum(ψ, Int(L/2))
    println("Entanglement spectrum: \n")
    display(ent)
end