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
w = [1.0,1.0]
g = [0.5 0.5;]    #g = [1.0 1.0;]  #g = [1.0;;]
max_b = 4
svalue = 3.0

model = HubbardParams(t, U)
calc  = CalcConfig(symm, model, HolsteinTerm(w, g, max_b, 1.0))

filename = "05__$(t)_$(U)_$(w)_$(g)_$(max_b)_s=$(svalue)"

path = "data/"
file_path = joinpath(path, filename * ".jld2")

gs = if isfile(file_path)
    println("Loading existing computation:")
    println(file_path)
    load_computation(file_path)
else
    # Step 3: Compute the ground state
    println("Computing and saving:")
    println(file_path)
    gs = compute_groundstate(calc; svalue=svalue)
    save_computation(gs, path, filename)
    gs
end

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