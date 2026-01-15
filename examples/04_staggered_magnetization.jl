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

t = Dict(
    (1,2) => 0.486, (2,1) => 0.486,
    (1,3) => 0.077, (3,1) => 0.077,
    (1,4) => 0.018, (4,1) => 0.018
)

# Interaction terms:
# (i,j,k,l) correspond to U_ijkl c⁺_i c⁺_j c_k c_l
U = Dict(
    (1,1,1,1) => 3.411,
    (1,2,2,1) => 1.042,
    (2,1,1,2) => 1.042,
    (1,2,1,2) => 0.033,
    (2,1,2,1) => 0.033,
    (1,1,2,2) => 0.033,
    (2,2,1,1) => 0.033
)

t_tag = dict_tag(t)
U_tag = dict_tag(U)

J_inter = 0.00167          # Inter-chain coupling / coupling constant

# Main function
function run_self_consistent_ms(symm::SymmetryConfig, t::AbstractDict{Tuple{Int,Int},Float64}, U::AbstractDict{Tuple{Int,Int,Int,Int},Float64}, J_inter::Float64; 
                                max_iter::Int=5, tol::Float64=1e-4, svalue::Float64=4.5)
    Ms = 0.0

    path = "data/"
    filename = "04__$(t_tag)_$(U_tag)_$(J_inter)_1_s=4.4"
    file_path = joinpath(path, filename * ".jld2")
    gs = load_computation(file_path)
    ψ_init = gs["groundstate"]
    Ms_list = Float64[Ms]
    E_list = Float64[]

    println("Starting self-consistent calculation of staggered magnetization")
    println("(Max iterations: $max_iter, Tolerance: $tol)")
    for i in 1:max_iter
        println("\n--- Iteration $i: Ms = $Ms ---")
        
        # Step 2: Set up model parameters with current Ms
        model = HubbardParams(1, t, U)
        calc = CalcConfig(symm, model, StaggeredField(J_inter, Ms))

        filename = "04__$(t_tag)_$(U_tag)_$(J_inter)_$(i)_s=$(svalue)" 

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
            gs = compute_groundstate(calc; svalue=svalue, init_state=ψ_init)
            save_computation(gs, path, filename)
            gs
        end
        
        ψ = gs["groundstate"]
        ψ_init = ψ

        H = gs["ham"]
        E0 = expectation_value(ψ, H)
        E = sum(real(E0)) / length(H)
        push!(E_list, E)
        
        dim = dim_state(ψ)
        println("Max bond dimension: ", maximum(dim))
        Ne = density_e(ψ, calc)
        println("Number of electrons per site: ", Ne)
        
        # Step 4: Calculate new Ms
        Ms_new = calc_ms(ψ, calc)
        Ms_change = abs(Ms_new - Ms)
        push!(Ms_list, Ms_new)
        
        # Check for convergence
        if Ms_change < tol
            println("\n***** CONVERGED *****")
            println("Ms converged to $Ms_new after $i iterations.")
            break
        end
        Ms = Ms_new
        
        if i == max_iter
            println("\n***** MAX ITERATIONS REACHED *****")
        end
    end

    return E_list, Ms_list
end

E_list, Ms_list = run_self_consistent_ms(symm, t, U, J_inter)
println("\n===== ITERATION RESULTS =====")
println("Energies = ", E_list)
println("Ms values = ", Ms_list)