###############
# Excitations #
###############

"""
    compute_excitations(groundstate_dict, momenta, charges; nums=1, solver=Arnoldi(...))

Compute the low-lying quasiparticle excitations above a given ground state.

# Arguments
- `groundstate_dict::Dict{String,Any}`: A dictionary produced by `compute_groundstate`, containing the keys `"groundstate"`, `"ham"`, and `"environments"`.
- `momenta::Union{Float64,Vector{Float64}}`: A collection of momentum values (in units of lattice sites) at which excitations are evaluated.
- `charges::Vector{Float64}`: Target quantum numbers defining the excitation sector (one value per symmetry).
- `nums::Int64=1`: Number of excitations to compute per momentum.
- `solver`: The eigensolver used for diagonalization (default is `Arnoldi(; krylovdim=30, tol=1e-6, eager=true)`).

# Returns
A dictionary with the following keys:
- `"Es"`: Eigenenergies of the excitations.
- `"qps"`: Quasiparticle ansatz states.
- `"momenta"`: The input momentum values.
"""
function compute_excitations(
                    groundstate_dict::Dict{String,Any},
                    momenta::Union{Float64,Vector{Float64}},
                    charges::Union{Vector{Int64},Vector{Float64}};
                    nums::Int64=1,
                    solver=Arnoldi(;krylovdim=30,tol=1e-6,eager=true)
                )
    ψ = groundstate_dict["groundstate"]
    H = groundstate_dict["ham"]
    envs = groundstate_dict["environments"]

    trivial_sector = first(sectors(oneunit(physicalspace(H, 1))))
    @assert length(charges) == length(trivial_sector) "Number of charges must match number of symmetries ($(length(trivial_sector)))."
    sector = foldl(⊠, [typeof(f)(charges[i]) for (i, f) in enumerate((trivial_sector))])

    Es, qps = excitations(H, QuasiparticleAnsatz(solver, MPSKit.Defaults.alg_environments(;dynamic_tols=false)), 
                            momenta./length(H), ψ, envs; num=nums, sector=sector)

    return Dict("Es" => Es, "qps" => qps, "momenta" => momenta)
end

"""
    compute_domainwall(groundstate_dict, momenta, charges; nums=1, shift=1, solver=Arnoldi(...))

Compute domain-wall excitations between a ground state and a spatially shifted version of itself.

# Arguments
- `groundstate_dict::Dict{String,Any}`: A dictionary produced by `compute_groundstate`, containing `"groundstate"`, `"ham"`, and `"environments"`.
- `momenta::Union{Float64,Vector{Float64}}`: A collection of momentum values (in units of lattice sites) at which domain-wall excitations are evaluated.
- `charges::Vector{Float64}`: Target quantum numbers defining the excitation sector (one value per symmetry).
- `nums::Int64=1`: Number of excitations to compute per momentum.
- `shift::Int64=1`: The number of lattice sites by which to shift the reference ground state to form the domain wall.
- `solver`: The eigensolver used for diagonalization (default is `Arnoldi(; krylovdim=30, tol=1e-6, eager=true)`).

# Returns
A dictionary with the following keys:
- `"Es"`: Eigenenergies of the domain-wall excitations.
- `"qps"`: Domain-wall quasiparticle ansatz states.
- `"momenta"`: The input momentum values.

# Notes
This function constructs the second “shifted” ground state and its environments automatically using `circshift` and `environments`.  
It then computes excitations between the two MPS states within the specified symmetry sector.
"""
function compute_domainwall(
                    groundstate_dict::Dict{String,Any},
                    momenta::Union{Float64,Vector{Float64}},
                    charges::Vector{Float64};
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
    compute_bandgap(groundstate_dict, momenta; nums=1)

Compute the single-particle (charge) band gap from particle–addition and particle–removal excitations.

# Arguments
- `groundstate_dict::Dict{String,Any}`: A dictionary produced by `compute_groundstate`, containing the ground state, Hamiltonian, and environments.
- `momenta::Union{Float64,Vector{Float64}}`: Momentum values at which the excitations are evaluated.
- `nums::Int64=1`: Number of excitations computed per momentum in each sector.

# Returns
- `(gap, kmin)`: The minimum value of `E_add(k) + E_remove(k)` and the corresponding momentum.
"""
function compute_bandgap(gs, momenta; nums::Int64=1)
    ex_add = compute_excitations(gs, momenta, [1,  1.0, 1//2]; nums=nums)
    ex_rem = compute_excitations(gs, momenta, [1, -1.0, 1//2]; nums=nums)

    Es = ex_add["Es"] .+ ex_rem["Es"]
    gap, k = findmin(real.(Es[:,1]))
    return gap, momenta[k]
end

"""
    compute_spingap(groundstate_dict, momenta; nums=1)

Compute the spin gap from spin-flip excitations above the ground state.

# Arguments
- `groundstate_dict::Dict{String,Any}`: A dictionary produced by `compute_groundstate`, containing the ground state, Hamiltonian, and environments.
- `momenta::Union{Float64,Vector{Float64}}`: Momentum values at which the excitations are evaluated.
- `nums::Int64=1`: Number of excitations computed per momentum.

# Returns
- `(gap, kmin)`: The minimum excitation energy `E(k)` in the spin sector and the corresponding momentum.
"""
function compute_spingap(gs, momenta; nums::Int64=1)
    ex = compute_excitations(gs, momenta, [0, 0.0, 1]; nums=nums)
    Es = ex["Es"]
    gap, k = findmin(real.(Es[:,1]))
    return gap, momenta[k]
end