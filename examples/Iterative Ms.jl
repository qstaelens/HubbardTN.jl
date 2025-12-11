using HubbardTN
using TensorKit, MPSKit, Statistics

function run_iterative_ms()
    # Step 1: Define the symmetries
    particle_symmetry = U1Irrep
    spin_symmetry = U1Irrep
    cell_width = 2
    filling = (1, 1)

    symm = SymmetryConfig(particle_symmetry, spin_symmetry, cell_width, filling)

    J_inter = 0.00115
    Ms = 0.0

    E_list  = Float64[]
    Ms_list = Float64[Ms]
    ψ_init = nothing

    for i in 1:2
        println("\nStep $i: Ms = $Ms")

        # Step 2: Set up model parameters
        t = [0.0, 0.49,0.078]   # [chemical_potential, nn_hopping, nnn_hopping, ...]
        U = [3.56,1.09]        # [on-site interaction, nn_interaction, ...]

        model = ModelParams2(t, U; Ms = Ms, J_inter = J_inter)
        calc  = CalcConfig(symm, model)

        # Step 3: Compute the ground state
        if ψ_init === nothing
            gs = compute_groundstate(calc; svalue = 2.5)
        else
            println("Using previous gs as initial state")
            gs = compute_groundstate(calc; svalue = 2.5,init_state = ψ_init)
        end
        ψ = gs["groundstate"]
        ψ_init = ψ
        H = gs["ham"]

        E0 = expectation_value(ψ, H)
        E = sum(real(E0)) / length(H)
        push!(E_list, E)
        println("Groundstate energy: $E")

        dim = dim_state(ψ)
        b = mean(dim)
        println("Mean bond dimension: $b")

        u, d = density_spin(ψ, symm)
        Ne = density_e(ψ, symm)

        println("Mean number op electrons = ", mean(Ne))
        println("Spin up = ", u)
        println("Spin down = ", d)

        # update Ms
        Ms_new = calc_ms(ψ, symm)
        push!(Ms_list, Ms_new)
        println("New Ms: $Ms_new")

        Ms = Ms_new
    end

    return E_list, Ms_list
end

E_list, Ms_list = run_iterative_ms()

println("\n===== ITERATION DONE =====")
println("Energies = ", E_list)
println("Ms values = ", Ms_list)
