println("
#####################
#  Staggered Field  #
#####################
")

tol = 2e-1

# Symmetries
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
                                max_iter::Int=5, tol::Float64=1e-4, svalue::Float64=3.0)
    Ms = 0.0
    ψ_init = nothing
    Ms_list = Float64[Ms]
    E_list = Float64[]

    for i in 1:max_iter
        # Step 2: Set up model parameters with current Ms
        model = HubbardParams(t, U)
        calc = CalcConfig(symm, model, StaggeredField(J_inter, Ms))

        # Step 3: Compute the ground state
        gs = compute_groundstate(calc; svalue = svalue, init_state=ψ_init)
        
        ψ = gs["groundstate"]
        ψ_init = ψ

        H = gs["ham"]
        E0 = expectation_value(ψ, H)
        E = sum(real(E0)) / length(H)
        push!(E_list, E)
        
        # Step 4: Calculate new Ms
        Ms_new = calc_ms(ψ, calc)
        Ms_change = abs(Ms_new - Ms)
        push!(Ms_list, Ms_new)
        
        # Check for convergence
        if Ms_change < tol
            break
        end
        Ms = Ms_new
    end

    return E_list, Ms_list
end

<<<<<<< HEAD
E_norm = [0.8573199392123321, 0.8572936394684572, 0.8572939920475093, 0.8572928105424324, 0.8572927048812922]
Ms_norm = [0.0, 0.14651530948645353, 0.164353722686657, 0.17033500288817305, 0.17091587359555932, 0.17092203478471346]
=======
E_norm = [0.857459787766349, 0.8574129912354749, 0.8574099883087606, 0.8574097771101556]
Ms_norm = [0.0, 0.19747222195453984, 0.20971989470700436, 0.21058138922470027, 0.21064333590417433]
>>>>>>> af2b6a11c9c55a71fe1769b664c3566fbf2aa99e

@testset "find " begin
    E_list, Ms_list = run_self_consistent_ms(symm, t, U, J_inter)
    @test E_list[2] ≈ E_norm[2] atol=tol
    @test Ms_list[2] ≈ Ms_norm[2] atol=tol
end