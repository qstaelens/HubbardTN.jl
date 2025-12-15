using HubbardTN
using TensorKit, MPSKit, Statistics

# Step 1: Define the symmetries
particle_symmetry = Trivial
spin_symmetry = U1Irrep
cell_width = 2

symm = SymmetryConfig(particle_symmetry, spin_symmetry, cell_width)

# Step 2: Set up model parameters
t = [2.0, 1.0]   # [chemical_potential, nn_hopping, nnn_hopping, ...]
U = [4.0]        # [on-site interaction, nn_interaction, ...]
W_G_cutoff = (1.0,2.0,10.0)

model = HolsteinParams(t, U; W_G_cutoff)
calc = CalcConfig(symm, model)

# Step 3: Compute the ground state
gs = compute_groundstate(calc; svalue = 3.0)
ψ = gs["groundstate"]
H = gs["ham"]

E0 = expectation_value(ψ, H)
E = sum(real(E0)) / length(H)
println("Groundstate energy: $E")

dim = dim_state(ψ)
b = mean(dim)
println("Mean bond dimension: $b")

e = entanglement_spectrum(ψ)
println("Entanglement spectrum: $e")

Ne = density_e_HH(ψ,symm)
println("Number of electrons per site: ", Ne)
println("Mean number of electrons = ", mean(Ne))

Nb = density_b(ψ,symm; W_G_cutoff)
println("Number of phonons per site: ", Nb)
println("Mean number of phonons = ", mean(Nb))

u, d = density_spin_HH(ψ,symm)
println("Spin up: $u")
println("Spin down: $d")