using HubbardTN
using TensorKit, MPSKit

L = 8
s = 2.0
particle_symmetry = U1Irrep
spin_symmetry = U1Irrep

t = Dict((1,2)=>1.0, (2,1)=>1.0, (1,1)=>2.0)

U = Dict(
    (1,1,1,1) => 4.0)


model = HubbardParams(1, t, U)
symm = SymmetryConfig(particle_symmetry, spin_symmetry; length = L)
calc = CalcConfig(symm, model)

gs = compute_groundstate(calc; svalue=s)
ψ = gs["groundstate"]
H = gs["ham"]

E0 = expectation_value(ψ, H)
E = sum(real(E0)) / length(H)
println("Groundstate energy: ", E)
dim = dim_state(ψ)
println("Max bond dimension: ", maximum(dim))

Ne = density_e(ψ, calc)
println("Number of electrons per site: ", Ne)

ent = entanglement_spectrum(ψ, Int(L/2))
println("Entanglement spectrum: \n")
display(ent)