using HubbardTN
using TensorKit, MPSKit

# This example simulates the mean-field coupling of two magnetic chains
# by self-consistently calculating the staggered magnetization Ms.

# Step 1: Define the symmetries and initial parameters
particle_symmetry = U1Irrep
spin_symmetry = U1Irrep
cell_width = 2
filling = (1, 1)
symm = SymmetryConfig(particle_symmetry, spin_symmetry, cell_width, filling)

# Model parameters (used inside the loop)
t = [0.0, 0.49, 0.078]     # [chemical_potential, nn_hopping, nnn_hopping, ...]
U = [3.56, 1.09]           # [on-site interaction, nn_interaction, ...]
J_inter = 0.00115          # Inter-chain coupling / coupling constant

# Main function
function run_self_consistent_ms(symm::SymmetryConfig, t::Vector{Float64}, 
                                U::Vector{Float64}, J_inter::Float64; 
                                max_iter::Int=5, tol::Float64=1e-4, svalue::Float64=2.5)
    Ms = 0.0
    ψ_init = nothing
    Ms_list = Float64[Ms]
    E_list = Float64[]

    println("Starting self-consistent calculation of staggered magnetization")
    println("(Max iterations: $max_iter, Tolerance: $tol)")
    for i in 1:max_iter
        println("\n--- Iteration $i: Ms = $Ms ---")
        
        # Step 2: Set up model parameters with current Ms
        model = ModelParams(t, U; J_M0=(J_inter, Ms))
        calc = CalcConfig(symm, model)

        # Step 3: Compute the ground state
        gs = compute_groundstate(calc; svalue = svalue, init_state=ψ_init)
        
        ψ = gs["groundstate"]
        ψ_init = ψ

        H = gs["ham"]
        E0 = expectation_value(ψ, H)
        E = sum(real(E0)) / length(H)
        push!(E_list, E)
        
        # Step 4: Calculate new Ms
        Ms_new = calc_ms(ψ, symm)
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