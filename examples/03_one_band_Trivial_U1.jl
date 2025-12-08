using HubbardTN
using TensorKit, MPSKit, Statistics

# Step 1: Define the symmetries
particle_symmetry = Trivial
spin_symmetry = U1Irrep
cell_width = 2

symm = SymmetryConfig(particle_symmetry, spin_symmetry, cell_width)

# Step 2: Set up model parameters
t = [4.0, 1.0]   # [chemical_potential, nn_hopping, nnn_hopping, ...]
U = [4.0, 1.0]        # [on-site interaction, nn_interaction, ...]

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
b = mean(dim)
println("Mean bond dimension: $b")

e = entanglement_spectrum(ψ)
println("Entanglement spectrum: $e")

Ne = density_e(ψ,symm)
println("Ne: $Ne")
println("Mean Ne = ", mean(Ne))

u, d = density_spin(ψ,symm)
println("Spin up: $u")
println("Spin down: $d")

Ms = calc_ms(ψ,symm)
println("Ms: $Ms")
