using HubbardTN
using TensorKit, MPSKit
using Revise


# Step 1: Define the symmetries
particle_symmetry = Trivial
spin_symmetry = U1Irrep
cell_width = 2

symm = SymmetryConfig(particle_symmetry, spin_symmetry, cell_width)

# Step 2: Set up model parameters
t = [2.0, 1.0]   # [chemical_potential, nn_hopping, nnn_hopping, ...]
U = [4.0]        # [on-site interaction, nn_interaction, ...]

model = HubbardParams(t, U)
calc = CalcConfig(symm, model)

# Step 3: Compute the ground state
gs = compute_groundstate(calc)
ψ = gs["groundstate"]
H = gs["ham"]

E0 = expectation_value(ψ, H)
E = sum(real(E0)) / length(H)
println("Groundstate energy: ", E)

dim = dim_state(ψ)
println("Max bond dimension: ", maximum(dim))

ent = entanglement_spectrum(ψ)
println("Entanglement spectrum: \n")
display(ent)

Ne = density_e(ψ, calc)
println("Number of electrons per site: ", Ne)

u, d = density_spin(ψ, calc)
println("Spin up per site: ", u)
println("Spin down per site: ", d)

Ms = calc_ms(ψ, calc)
println("Staggered magnetization: ", Ms)