using HubbardTN
using TensorKit, MPSKit
using Revise

# This example simulates the mean-field coupling of two magnetic chains
# by self-consistently calculating the staggered magnetization Ms.

# Step 1: Define the symmetries and initial parameters
particle_symmetry = U1Irrep
spin_symmetry = U1Irrep
cell_width = 2
filling = (1, 1)
symm = SymmetryConfig(particle_symmetry, spin_symmetry, cell_width, filling)

# Model parameters (used inside the loop)
t = [0.0, 0.49, 0.077]     # [chemical_potential, nn_hopping, nnn_hopping, ...]
U = [3.41, 1.04]           # [on-site interaction, nn_interaction, ...]
J_inter = 0.00162          # Inter-chain coupling / coupling constant

s = 3.5

bands = 3

# Hopping amplitudes:
# (1,2) and (2,1): inter-band hopping
# (2,3) and (3,2): next-nearest-neighbor hopping across unit cells
t = Dict(
    # band 1
    (1,4) => 0.49,
    (4,1) => 0.49,

    # band 2
    (2,5) => 0.49,
    (5,2) => 0.49,

    # band 3
    (3,6) => 0.49,
    (6,3) => 0.49,

    # explicitly zero interband hoppings (same site)
    (1,2) => 0.0, (2,1) => 0.0,
    (1,3) => 0.0, (3,1) => 0.0,
    (2,3) => 0.0, (3,2) => 0.0
)

# Interaction terms:
# (i,j,k,l) correspond to U_ijkl c⁺_i c⁺_j c_k c_l
U = Dict(
    (1,1,1,1) => 3.41,
    (2,2,2,2) => 3.41,
    (3,3,3,3) => 3.41,
    (1,1,4,4) => 1.04,
    (4,4,1,1) => 1.04,
    (2,2,5,5) => 1.04,
    (5,5,2,2) => 1.04,
    (3,3,6,6) => 1.04,
    (6,6,3,3) => 1.04,
    (1,2,1,2) => 0.00162,
    (2,1,2,1) => 0.00162,
    (2,3,2,3) => 0.00162,
    (3,2,3,2) => 0.00162,
)

t_tag = dict_tag(t)
U_tag = dict_tag(U)
filename = "06__$(t_tag)__$(U_tag)__s=$(s)" #_$(J_inter)_$(svalue)_$i

model = HubbardParams(bands, t, U)
calc = CalcConfig(symm, model)

# Step 3: Compute the ground state
path = "data/"
file_path = joinpath(path, filename * ".jld2")

gs = if isfile(file_path)
    println("Loading existing computation:")
    println(file_path)
    load_computation(file_path)
else
    println("Computing and saving:")
    println(file_path)
    gs = compute_groundstate(calc; svalue=s)
    save_computation(gs, path, filename)
    gs
end

ψ = gs["groundstate"]
H = gs["ham"]

println("Ground-state energy density: ", expectation_value(ψ, H) / length(H))

dim = dim_state(ψ)
println("Max bond dimension: ", maximum(dim))

ent = entanglement_spectrum(ψ)
println("Entanglement spectrum: \n")
display(ent)

Ne = density_e(ψ, calc)
println("Number of electrons per site: ", Ne)

u, d = density_spin(ψ, calc)
println("Spin up: $u")
println("Spin down: $d")