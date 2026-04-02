###############
# Excitations #
###############

"""
    compute_excitations(groundstate_dict, calc, momenta, charges; nums=1, svalue=0.0, solver=Arnoldi(...))

Compute the low-lying quasiparticle excitations above a given ground state.

# Arguments
- `groundstate_dict::Dict{String,Any}`: A dictionary produced by `compute_groundstate`, containing the keys `"groundstate"`, `"ham"`, and `"environments"`.
- `calc::CalcConfig`: Calculation configuration used to construct filenames and identify the model parameters.
- `momenta::Union{Float64,Vector{Float64}}`: Momentum value or collection of momentum values at which excitations are evaluated.
- `charges::Union{Vector{Int64},Vector{Float64}}`: Target quantum numbers defining the excitation sector, with one value per symmetry sector.
- `nums::Int64=1`: Number of excitations to compute per momentum.
- `svalue::Float64=0.0`: Value included in the excitation filename to distinguish cached data from different runs.
- `solver`: The eigensolver used for diagonalization (default is `Arnoldi(; krylovdim=30, tol=1e-6, eager=true)`).

# Returns
A dictionary with the following keys:
- `"Es"`: Eigenenergies of the excitations.
- `"qps"`: Quasiparticle ansatz states, or `nothing` if the data were loaded from disk.
- `"momenta"`: The input momentum values.
"""
function compute_excitations(groundstate_dict::Dict{String,Any}, calc::CalcConfig, momenta::Union{Float64,Vector{Float64}}, charges::Union{Vector{Int64},Vector{Float64}}; nums::Int64=1, svalue::Float64=0.0, solver=Arnoldi(;krylovdim=30,tol=1e-6,eager=true))
    ψ = groundstate_dict["groundstate"]
    H = groundstate_dict["ham"]
    envs = groundstate_dict["environments"]
    trivial_sector = first(sectors(oneunit(physicalspace(H, 1))))
    @assert length(charges) == length(trivial_sector) "Number of charges must match number of symmetries ($(length(trivial_sector)))."
    sector = foldl(⊠, [typeof(f)(charges[i]) for (i, f) in enumerate(trivial_sector)])
    println(sector)

    folder = "data/excitations"
    isdir(folder) || mkpath(folder)
    t_tag = dict_tag(calc.hubbard.t)
    U_tag = dict_tag(calc.hubbard.U)
    filename = "exc__$(t_tag)_$(U_tag)_s=$(svalue)_ch=$(join(string.(charges), "_"))_k=$(join(string.(round.(collect(momenta), digits=4)), "_"))_num=$(nums).jld2"
    filepath = joinpath(folder, filename)

    if isfile(filepath)
        data = load(filepath)
        return Dict("Es" => data["Es"], "qps" => nothing, "momenta" => data["momenta"])
    end

    Es, qps = excitations(H, QuasiparticleAnsatz(solver, MPSKit.Defaults.alg_environments(;dynamic_tols=false)), momenta./length(H), ψ, envs; num=nums, sector=sector)
    @save filepath Es momenta charges svalue
    return Dict("Es" => Es, "qps" => qps, "momenta" => momenta)
end

"""
    compute_domainwall(groundstate_dict, momenta, charges; nums=1, shift=1, solver=Arnoldi(...))

Compute domain-wall excitations between a ground state and a spatially shifted version of itself.

# Arguments
- `groundstate_dict::Dict{String,Any}`: A dictionary produced by `compute_groundstate`, containing `"groundstate"`, `"ham"`, and `"environments"`.
- `momenta::Union{Float64,Vector{Float64}}`: Momentum value or collection of momentum values at which domain-wall excitations are evaluated.
- `charges::Union{Vector{Int64},Vector{Float64}}`: Target quantum numbers defining the excitation sector, with one value per symmetry sector.
- `nums::Int64=1`: Number of excitations to compute per momentum.
- `shift::Int64=1`: Number of lattice sites by which to shift the reference ground state to form the domain wall.
- `solver`: The eigensolver used for diagonalization (default is `Arnoldi(; krylovdim=30, tol=1e-6, eager=true)`).

# Returns
A dictionary with the following keys:
- `"Es"`: Eigenenergies of the domain-wall excitations.
- `"qps"`: Domain-wall quasiparticle ansatz states.
- `"momenta"`: The input momentum values.

# Notes
This function constructs the shifted ground state and its environments automatically using `circshift` and `environments`.
It then computes excitations between the two MPS states within the specified symmetry sector.
"""
function compute_domainwall(
                    groundstate_dict::Dict{String,Any},
                    momenta::Union{Float64,Vector{Float64}},
                    charges::Union{Vector{Int64},Vector{Float64}};
                    nums::Int64=1,
                    shift::Int64=1,
                    solver=Arnoldi(;krylovdim=30,tol=1e-6,eager=true)
                )
    ψ = groundstate_dict["groundstate"]
    H = groundstate_dict["ham"]
    envs = groundstate_dict["environments"]

    trivial_sector = first(sectors(oneunit(physicalspace(H, 1))))
    @assert length(charges) == length(trivial_sector) "Number of charges must match number of symmetries ($(length(trivial_sector)))."
    sector = foldl(⊠, [typeof(f)(charges[i]) for (i, f) in enumerate((trivial_sector))])

    ψ_s = circshift(ψ, shift)
    envs_s = environments(ψ_s, H);
    Es, qps = excitations(H, QuasiparticleAnsatz(solver, MPSKit.Defaults.alg_environments(;dynamic_tols=false)),
                            momenta./length(H), ψ, envs, ψ_s, envs_s; num=nums, sector=sector)

    return Dict("Es" => Es, "qps" => qps, "momenta" => momenta)
end

"""
    compute_bandgap(gs, calc; resolution=5, svalue=0.0)

Compute the single-particle band gap from particle-addition and particle-removal excitations over a uniform momentum grid.

# Arguments
- `gs::Dict{String,Any}`: Ground-state data as produced by `compute_groundstate`, containing the state, Hamiltonian, and environments.
- `calc::CalcConfig`: Calculation configuration. The particle symmetry `calc.symmetries.particle_symmetry` must be `U1Irrep`.

# Keyword Arguments
- `resolution::Int=5`: Number of momentum points in the uniform grid between `0` and `π` (inclusive).
- `svalue::Float64=0.0`: Value passed to `compute_excitations` for cache naming.

# Returns
- `(gap, kmin)`: The minimum value of `E_add(k) + E_remove(k)` over the sampled momenta and the corresponding momentum `k`.
"""
function compute_bandgap(gs::Dict{String,Any}, calc::CalcConfig; resolution::Int64=5, svalue::Float64=0.0)
    @assert calc.symmetries.particle_symmetry==U1Irrep "Particle symmetry must be of type U1Irrep."
    d = denominator(calc.symmetries.filling)
    if calc.symmetries.spin_symmetry==Trivial
        charges_particle = [1, d]
    else
        charges_particle = [1, d, 1/2]
    end
    charges_hole = copy(charges_particle)
    charges_hole[2] *= -1

    momenta = collect(range(0, π, resolution))

    ex_add = compute_excitations(gs, calc, momenta, charges_particle; nums=1, svalue=svalue)
    ex_rem = compute_excitations(gs, calc, momenta, charges_hole; nums=1, svalue=svalue)
    println(ex_add["Es"])
    println(ex_rem["Es"])
    Eb = ex_add["Es"] .+ ex_rem["Es"]
    gap, k = findmin(real.(Eb[:,1]))
    return gap, momenta[k]
end

"""
    compute_spingap(gs, calc; resolution=5, svalue=0.0)

Compute the spin gap from spin-flip excitations above the ground state over a uniform momentum grid.

# Arguments
- `gs::Dict{String,Any}`: Ground-state data as produced by `compute_groundstate`, containing the state, Hamiltonian, and environments.
- `calc::CalcConfig`: Calculation configuration. The spin symmetry `calc.symmetries.spin_symmetry` must not be `Trivial`.

# Keyword Arguments
- `resolution::Int=5`: Number of momentum points in the uniform grid between `0` and `π` (inclusive).
- `svalue::Float64=0.0`: Value passed to `compute_excitations` for cache naming.

# Returns
- `(gap, kmin)`: The minimum spin excitation energy over the sampled momenta and the corresponding momentum `k`.
"""
function compute_spingap(gs::Dict{String,Any}, calc::CalcConfig; resolution::Int64=5, svalue::Float64=0.0)
    @assert calc.symmetries.spin_symmetry!=Trivial "Spin symmetry must not be Trivial."
    if calc.symmetries.particle_symmetry==Trivial
        charges = [0, 1]
    else
        charges = [0, 0, 1]
    end

    momenta = collect(range(0, π, resolution))
    ex = compute_excitations(gs, calc, momenta, charges; nums=1, svalue=svalue)
    Es = ex["Es"]
    println("Es, $Es")
    gap, k = findmin(real.(Es[:,1]))
    return gap, momenta[k]
end

"""
    compute_chargegap(gs, calc; resolution=5, svalue=0.0)

Compute the charge gap from two-particle addition and removal excitations on a uniform momentum grid.

# Arguments
- `gs::Dict{String,Any}`: Ground-state data as produced by `compute_groundstate`, containing the state, Hamiltonian, and environments.
- `calc::CalcConfig`: Calculation configuration. The particle symmetry `calc.symmetries.particle_symmetry` must be `U1Irrep`.

# Keyword Arguments
- `resolution::Int=5`: Number of momentum points in the uniform grid between `0` and `π` (inclusive).
- `svalue::Float64=0.0`: Value passed to `compute_excitations` for cache naming.

# Returns
- `(gap, kmin)`: The minimum charge excitation energy over the sampled momenta and the corresponding momentum `k`.
"""
function compute_chargegap(gs::Dict{String,Any}, calc::CalcConfig; resolution::Int64=5, svalue::Float64=0.0)
    @assert calc.symmetries.particle_symmetry==U1Irrep "Particle symmetry must be of type U1Irrep."
    d = denominator(calc.symmetries.filling)
    if calc.symmetries.spin_symmetry==Trivial
        charges_particle = [0, 2*d]
    else
        charges_particle = [0, 2*d, 0]
    end
    charges_hole = copy(charges_particle)
    charges_hole[2] *= -1

    momenta = collect(range(0, π, resolution))

    ex_add = compute_excitations(gs, calc, momenta, charges_particle; nums=1, svalue=svalue)
    ex_rem = compute_excitations(gs, calc, momenta, charges_hole; nums=1, svalue=svalue)

    Ec = 0.5 .* (ex_add["Es"] .+ ex_rem["Es"])
    println("Ec, $Ec")
    gap, k = findmin(real.(Ec[:,1]))
    return gap, momenta[k]
end

"""
    compute_pairing_energy(gs, calc; resolution=5, svalue=0.0)

Compute the pairing energy from single-particle and two-particle addition excitations on a uniform momentum grid.

# Arguments
- `gs::Dict{String,Any}`: Ground-state data as produced by `compute_groundstate`, containing the state, Hamiltonian, and environments.
- `calc::CalcConfig`: Calculation configuration. The particle symmetry `calc.symmetries.particle_symmetry` must be `U1Irrep`.

# Keyword Arguments
- `resolution::Int=5`: Number of momentum points in the uniform grid between `0` and `π` (inclusive).
- `svalue::Float64=0.0`: Value passed to `compute_excitations` for cache naming.

# Returns
- `(gap, kmin)`: The minimum value of `2E_add(k) - E_double(k)` over the sampled momenta and the corresponding momentum `k`.
"""
function compute_pairing_energy(gs::Dict{String,Any}, calc::CalcConfig; resolution::Int64=5, svalue::Float64=0.0, tol::Float64=1e-6)
    @assert calc.symmetries.particle_symmetry == U1Irrep "Particle symmetry must be of type U1Irrep."

    d = denominator(calc.symmetries.filling)

    if calc.symmetries.spin_symmetry == Trivial
        charges_particle = [1, d]
        charges_double = [0, 2*d]
    else
        charges_particle = [1, d, 1/2]
        charges_double = [0, 2*d, 0]
    end

    momenta = collect(range(0, 2π, resolution))

    ex_add = compute_excitations(gs, calc, momenta, charges_particle; nums=1, svalue=svalue)
    ex_double = compute_excitations(gs, calc, momenta, charges_double; nums=3, svalue=svalue)

    E1 = real.(vec(ex_add["Es"]))
    E2 = real.(vec(ex_double["Es"]))

    println("E1 = $E1")
    println("E2 = $E2")

    E1min = minimum(E1)
    idx_E1mins = findall(x -> isapprox(x, E1min; atol=tol), E1)

    valid_k = nothing
    for i in idx_E1mins
        k = momenta[i]
        k_partner = mod(2π - k, 2π)

        j = findfirst(x -> isapprox(x, k_partner; atol=tol), momenta)
        if j !== nothing && isapprox(E1[j], E1min; atol=tol)
            valid_k = (i, j)
            break
        end
    end

    valid_k === nothing && error("No valid E1 minimum found such that 2π-k is also a minimum.")

    E2min, idx_E2min = findmin(E2)
    k_E2min = momenta[idx_E2min]

    if !isapprox(k_E2min, 0.0; atol=tol) && !isapprox(k_E2min, 2π; atol=tol)
        error("E2 minimum is not at k = 0.")
    end
    Ep = E1min  + E1min - E2min
    return Ep, k_E2min
end