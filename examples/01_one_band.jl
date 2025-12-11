using HubbardTN
using TensorKit, MPSKit

# Step 1: Define the symmetries
particle_symmetry = U1Irrep
spin_symmetry = U1Irrep
cell_width = 2
filling = (1, 1)

symm = SymmetryConfig(particle_symmetry, spin_symmetry, cell_width, filling)

# Step 2: Set up model parameters
t = [0.0, 1.0]   # [chemical_potential, nn_hopping, nnn_hopping, ...]
U = [4.0]        # [on-site interaction, nn_interaction, ...]

model = ModelParams(t, U)
calc = CalcConfig(symm, model)

# Step 3: Compute the ground state
gs = compute_groundstate(calc)
ψ = gs["groundstate"]
H = gs["ham"]

E0 = expectation_value(ψ, H)
E = sum(real(E0)) / length(H)
println("Groundstate energy: $E")

dim = dim_state(ψ)
b = max(dim)
println("Max bond dimension: $b")

e = entanglement_spectrum(ψ)
println("Entanglement spectrum: $e")

Ne = density_e(ψ,symm)
println("Number of electrons per site: ", Ne)

u, d = density_spin(ψ,symm)
println("Spin up: $u")
println("Spin down: $d")

# Step 4: Compute first excitation in fZ2(0) × U1Irrep(0) × U1Irrep(0) sector
momenta = collect(range(0, 2π, length = 10))
charges = [0.0, 0.0, 0.0]
ex = compute_excitations(gs, momenta, charges)