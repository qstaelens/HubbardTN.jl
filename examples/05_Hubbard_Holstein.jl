using HubbardTN
using TensorKit, MPSKit

# Step 1: Define the symmetries
particle_symmetry = Trivial
spin_symmetry = U1Irrep
cell_width = 2

symm = SymmetryConfig(particle_symmetry, spin_symmetry, cell_width)

# Step 2: Set up model parameters
t = [2.0, 1.0]   # [chemical_potential, nn_hopping, nnn_hopping, ...]
U = [4.0]        # [on-site interaction, nn_interaction, ...]
w = 1.0
g = [0.5]
max_b = 4

model = HubbardParams(t, U)
calc  = CalcConfig(symm, model, HolsteinTerm(w, g, max_b, 1.0))

# Step 3: Compute the ground state
gs = compute_groundstate(calc; svalue = 3.0)
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
println("Mean number of electrons = ", sum(Ne)/length(Ne))

Nb = density_b(ψ, calc)
println("Number of phonons per site: ", Nb)
println("Mean number of phonons = ", sum(Nb)/length(Nb))

u, d = density_spin(ψ, calc)
println("Spin up: $u")
println("Spin down: $d")