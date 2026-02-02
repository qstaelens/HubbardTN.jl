using HubbardTN
using TensorKit, MPSKit
using Revise

# Step 1: Define the symmetries
particle_symmetry = U1Irrep
spin_symmetry = U1Irrep
cell_width = 2
filling = (1, 1)

symm = SymmetryConfig(particle_symmetry, spin_symmetry, cell_width, filling)

# Step 2: Set up model parameters
t = [0.0, 1.0]   # [chemical_potential, nn_hopping, nnn_hopping, ...]
U = [-10.0]        # [on-site interaction, nn_interaction, ...]
svalue = 3.0

model = HubbardParams(t, U)
calc = CalcConfig(symm, model)
gs = compute_groundstate(calc; svalue=svalue)
ψ = gs["groundstate"]
H = gs["ham"]

E0 = expectation_value(ψ, H)
E = sum(real(E0)) / length(H)
println("Groundstate energy: ", E)

dim = dim_state(ψ)
println("Max bond dimension: ", maximum(dim))

Ne = density_e(ψ, calc)
println("Number of electrons per site: ", Ne)

momenta = collect(range(0, π, length = 5))

spin_gap, spin_k = compute_spingap(gs, momenta)
pairing_gap, pairing_k = compute_pairing_energy(gs, momenta)

println("Spin gap   at k = $(spin_k): Δs = $(spin_gap)")
println("Pairing gap   at k = $(pairing_k): Δp = $(pairing_gap)")

# Step 1: Define the symmetries
particle_symmetry = Trivial
symm = SymmetryConfig(particle_symmetry, spin_symmetry, cell_width)

# Step 2: Set up model parameters
t = [-5.0, 1.0]   # [chemical_potential, nn_hopping, nnn_hopping, ...]
U = [-10.0]        # [on-site interaction, nn_interaction, ...]
alpha = [0.0,0.0]
beta = [0.0,0.0]
model = HubbardParams(t, U)

alpha_list = Vector{Vector{Float64}}()
beta_list = Vector{Vector{Float64}}()
E_list     = Float64[]

for i in 1:5
    global alpha, beta
    println("Step $i: alpha = $alpha, beta = $beta")
    calc = CalcConfig(symm, model, Bollmark(alpha, beta))

    # Step 3: Compute the ground state
    gs = compute_groundstate(calc; svalue=svalue)
    ψ = gs["groundstate"]
    H = gs["ham"]

    E0 = expectation_value(ψ, H)
    E = sum(real(E0)) / length(H)

    dim = dim_state(ψ)
    println("Max bond dimension: ", maximum(dim))

    Ne = density_e(ψ, calc)
    println("Number of electrons per site: ", Ne)

    push!(alpha_list, copy(alpha))
    push!(beta_list, copy(beta))
    push!(E_list, E)

    ty = 0.5
    tz = 0.5
    alpha = get_alpha(ψ, symm, ty, tz, pairing_gap)
    beta = get_beta(ψ, symm, ty, tz, pairing_gap)
end