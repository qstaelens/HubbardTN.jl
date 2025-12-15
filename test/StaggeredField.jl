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
                                max_iter::Int=5, tol::Float64=1e-4, svalue::Float64=2.5)
    Ms = 0.0
    ψ_init = nothing
    Ms_list = Float64[Ms]
    E_list = Float64[]

    for i in 1:max_iter
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
            break
        end
        Ms = Ms_new
    end

    return E_list, Ms_list
end

E_norm = [0.8574597873171289, 0.8575037848413146, 0.8575008308545669, 0.8575010718651674, 0.8575010426788501]
Ms_norm = [0.0, 0.19747463281272887, 0.18381480739281603, 0.18488147851279135, 0.18474268922613346, 0.1847491211944738]

@testset "find " begin
    E_list, Ms_list = run_self_consistent_ms(symm, t, U, J_inter)
    @test E_list[2] ≈ E_norm[2] atol=tol
    @test Ms_list[2] ≈ Ms_norm[2] atol=tol
end