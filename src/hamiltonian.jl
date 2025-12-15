####################
# Helper functions #
####################

# Maps (i,j,k,l) → Symbol(:aabb), :abab, :abcd, etc.
function pattern_key(indices::Vararg{Int})
    seen = Dict{Int,Char}()
    nextchar = 'a'
    keychars = Char[]
    for idx in indices
        if !haskey(seen, idx)
            seen[idx] = nextchar
            nextchar = Char(nextchar + 1)
        end
        push!(keychars, seen[idx])
    end

    return Symbol(String(keychars))
end
# Maps indices to actual lattice sites based on pattern key
function compute_sites(indices::NTuple{N,Int}, lattice, key::Symbol) where {N}
    letters = collect(string(key))
    unique_letters = unique(letters)

    selected_indices = [indices[findfirst(==(letter), letters)] for letter in unique_letters]

    return tuple((lattice[idx] for idx in selected_indices)...)
end


#############################
# Two-body interaction term #
#############################

# Direct on-site: (i,i,i,i)
function two_body_int(ops, ::Val{:aaaa})
    return @tensor operator[-1; -2] := 2 * ops.n_pair[-1; -2] # factor 2 to counter double counting
end
# Direct inter-site: (i,j,j,i)
function two_body_int(ops, ::Val{:abba})
    return @tensor operator[-1 -2; -3 -4] := ops.n[-1; -3] * ops.n[-2; -4]
end
# Exchange: (i,j,i,j)
function two_body_int(ops, ::Val{:abab})
    return @tensor operator[-1 -2; -3 -4] := ops.c⁺c[-1 a; b -4] * ops.c⁺c[-2 b; a -3]
end
# Pair-exchange: (i,i,k,k)
function two_body_int(ops, ::Val{:aabb})
    return @tensor operator[-1 -2; -3 -4] := ops.c⁺c[-1 a; b -4] * ops.c⁺c[b -2; -3 a]
end
# Bond-charge: (i,i,i,l)
function two_body_int(ops, ::Val{:aaab})
    return @tensor operator[-1 -2; -3 -4] := ops.n[a; -3] * ops.c⁺c[-1 -2; a -4]
end
# Conjugated variant: (i,j,j,j)
function two_body_int(ops, ::Val{:abbb})
    return @tensor operator[-1 -2; -3 -4] := ops.c⁺c[-1 a; -3 -4] * ops.n[-2; a]
end
# Bond-charge: (i,i,k,i)
function two_body_int(ops, ::Val{:aaba})
    return @tensor operator[-1 -2; -3 -4] := ops.c⁺c[-1 a; b -3] * ops.c⁺c[b -2; a -4]
end
# Conjugated variant: (i,j,i,i)
function two_body_int(ops, ::Val{:abaa})
    return @tensor operator[-1 -2; -3 -4] := ops.c⁺c[-1 a; b -3] * ops.c⁺c[-2 b; -4 a]
end
# Three distinct sites: (i,i,k,l)
function two_body_int(ops, ::Val{:aabc})
    return @tensor operator[-1 -2 -3; -4 -5 -6] := ops.c⁺c[-1 -3; a -6] * ops.c⁺c[a -2; -4 -5]
end
# (i,j,i,l)
function two_body_int(ops, ::Val{:abac})
    return @tensor operator[-1 -2 -3; -4 -5 -6] := ops.c⁺c[-1 -3; a -6] * ops.c⁺c[-2 a; -5 -4]
end
# (i,j,k,i)
function two_body_int(ops, ::Val{:abca})
    return @tensor operator[-1 -2 -3; -4 -5 -6] := ops.n[-1; -4] * ops.c⁺c[-2 -3; -5 -6]
end
# (i,j,j,l)
function two_body_int(ops, ::Val{:abbc})
    return @tensor operator[-1 -2 -3; -4 -5 -6] := ops.n[-2; -5] * ops.c⁺c[-1 -3; -4 -6]
end
# (i,j,k,k)
function two_body_int(ops, ::Val{:abcc})
    return @tensor operator[-1 -2 -3; -4 -5 -6] := ops.c⁺c[-1 a; -4 -6] * ops.c⁺c[-2 -3; -5 a]
end
# (i,j,k,j)
function two_body_int(ops, ::Val{:abcb})
    return @tensor operator[-1 -2 -3; -4 -5 -6] := ops.c⁺c[-1 a; -4 -5] * ops.c⁺c[-2 -3; a -6]
end
# Four distinct sites: (i,j,k,l)
function two_body_int(ops, ::Val{:abcd})
    return @tensor operator[-1 -2 -3 -4; -5 -6 -7 -8] := ops.c⁺c[-1 -4; -5 -8] * ops.c⁺c[-2 -3; -6 -7]
end

# Caching constructed two-body operators
const two_body_cache = Dict{Symbol,Any}()
function two_body_int_cached(ops, (i,j,k,l)::NTuple{4,Int}, lattice)
    key = pattern_key(i,j,k,l)

    if haskey(two_body_cache, key)
        operator = two_body_cache[key]
    else
        operator = two_body_int(ops, Val(key))
        two_body_cache[key] = operator
    end
    sites = compute_sites((i,j,k,l), lattice, key)

    return operator, sites
end


###############################
# Three-body interaction term #
###############################

function three_body_int(ops, ::Val{key}) where key
    pattern = String(key)
    @assert length(pattern) == 6 "Three-body terms must have six indices."

    if all(pattern[1] == pattern[i] for i in 2:3)
        error("Three-body term of type V_iiilmn (first three equal) is not allowed.")
    elseif all(pattern[end] == pattern[end-i+1] for i in 1:3)
        error("Three-body term of type V_ijklll (last three equal) is not allowed.")
    end

    return _three_body_int(ops, Val(key))
end
function _three_body_int(ops, ::Val{:aabbaa})
    return @tensor operator[-1 -2; -3 -4] := 2 * ops.n_pair[-1; -3] * ops.n[-2; -4] # factor 2 to counter double counting
end
function _three_body_int(ops, ::Val{:abaaba})
    return @tensor operator[-1 -2; -3 -4] := 2 * ops.n_pair[-1; -3] * ops.n[-2; -4]
end
function _three_body_int(ops, ::Val{:abbbba})
    return @tensor operator[-1 -2; -3 -4] := 2 * ops.n[-1; -3] * ops.n_pair[-2; -4]
end
function _three_body_int(ops, ::Val{key}) where key
    error("Interaction terms with indices $key are not implemented")
end

# Caching constructed three-body operators
const three_body_cache = Dict{Symbol,Any}()
function three_body_int_cached(ops, (i,j,k,l,m,n)::NTuple{6,Int}, lattice)
    key = pattern_key(i,j,k,l,m,n)

    if haskey(three_body_cache, key)
        operator = three_body_cache[key]
    else
        operator = three_body_int(ops, Val(key))
        three_body_cache[key] = operator
    end
    sites = compute_sites((i,j,k,l), lattice, key)

    return operator, sites
end


############################
# Hamiltonian construction #
############################

# Build the symmetry-dependent operators
function build_ops(symm::SymmetryConfig)
    ps = symm.particle_symmetry
    ss = symm.spin_symmetry
    fill = symm.filling
    ops = (
        c⁺c      = c_plusmin(ps, ss; filling=fill),
        n_pair   = number_pair(ps, ss; filling=fill),
        n        = number_e(ps, ss; filling=fill),
        Sz       = Sz(ps, ss; filling=fill)
    )
    return ops
end

"""
    hamiltonian(calc::CalcConfig)

Constructs the many-body Hamiltonian for a lattice system with given hopping 
and interaction parameters, taking into account particle and spin symmetries.

# Arguments
- `calc::CalcConfig` : A configuration object containing:
    - `calc.symmetries` : SymmetryConfig, which includes:
        - `particle_symmetry`  - type of particle symmetry
        - `spin_symmetry`      - type of spin symmetry
        - `cell_width`         - number of unit cells in the system
        - `filling`            - particle filling
    - `calc.model` : Model parameters, which include:
        - `bands` - number of orbitals per unit cell
        - `t`     - hopping matrix or list of hopping terms
        - `U`     - two-body interaction tensor

# Returns
- `H` : The Hamiltonian as an Matrix Product Operator.

# Notes
- Lattice sites are represented using an `InfiniteChain` of length `cell_width * bands`.
- The resulting MPO can be used directly for DMRG, VUMPS, or other tensor network calculations.
"""
function hamiltonian(calc::CalcConfig{ModelParams{T}}) where {T}
    empty!(two_body_cache)
    empty!(three_body_cache)

    symm = calc.symmetries
    ops = build_ops(symm)
    cell_width = symm.cell_width

    bands = calc.model.bands
    t = calc.model.t
    U = calc.model.U
    V = calc.model.V
    lattice = InfiniteChain(cell_width * bands)

    H = @mpoham 0*ops.n{lattice[1]}       # Initialize MPO

    # --- Hopping ---
    for cell in 0:(cell_width-1)
        H += @mpoham sum(-t_ij * ops.c⁺c{lattice[i+cell*bands], lattice[j+cell*bands]} for ((i,j), t_ij) in t if i != j; init=0*ops.n{lattice[1]})
        H += @mpoham sum(-μ_i * ops.n{lattice[i+cell*bands]} for ((i,j), μ_i) in pairs(t) if i == j; init=0*ops.n{lattice[1]})
    end

    # --- 2-body Interaction ---
    for cell in 0:(cell_width-1)
        H += @mpoham sum(begin
            operator, indices = two_body_int_cached(ops, (i,j,k,l) .+ cell*bands, lattice)
            0.5 * U_ijkl * operator{indices...}
        end for ((i,j,k,l), U_ijkl) in collect(pairs(U)); init=0*ops.n{lattice[1]})
    end

    # --- 3-body Interaction ---
    for cell in 0:(cell_width-1)
        H += @mpoham sum(begin
            operator, indices = three_body_int_cached(ops, (i,j,k,l,m,n) .+ cell*bands, lattice)
            1/6 * V_ijklmn * operator{indices...}
        end for ((i,j,k,l,m,n), V_ijklmn) in collect(pairs(V)); init=0*ops.n{lattice[1]})
    end

    # --- Staggered magnetization field term: 2 * J_inter * Ms * (-1)^i Sz_i
    J_inter, Ms = calc.model.J_M0
    if Ms != 0.0 && J_inter != 0.0
        println("Using staggered magn field")
        H += @mpoham sum(2 * J_inter * Ms * (-1)^i * ops.Sz{lattice[i]} for i in 1:(cell_width * bands); init=0*ops.n{lattice[1]})
    end

    return H
end

function hamiltonian(calc::CalcConfig{HolsteinParams{T}}) where {T}
    empty!(two_body_cache)
    empty!(three_body_cache)

    symm  = calc.symmetries
    ops = build_ops(symm)

    ω0, g, cutoff = calc.model.W_G_cutoff
    cutoff = Int(cutoff)

    Ps  = hubbard_space(Trivial, U1Irrep; filling = symm.filling)
    Psb = holstein_space(Trivial, U1Irrep; cutoff=cutoff)

    # fixed geometry: site 1 = Hubbard, 2 = phonon, 3 = Hubbard, 4 = phonon
    spaces = [Ps, Psb, Ps, Psb]

    bmin   = boson_ann(Trivial, U1Irrep; cutoff=cutoff)
    bplus  = boson_cre(Trivial, U1Irrep; cutoff=cutoff)
    nb  = boson_number(Trivial, U1Irrep; cutoff=cutoff)

    # chemical potential
    μ = calc.model.t[(1,1)]

    H = InfiniteMPOHamiltonian(
        spaces, (1,) => -μ * ops.n
    )
    H += InfiniteMPOHamiltonian(
        spaces, (3,) => -μ * ops.n
    )

    # hopping: model.t[(1,2)] ↦ sites (1,3)
    t = calc.model.t[(1,2)]

    H += InfiniteMPOHamiltonian(
        spaces, (1,3) => -t * ops.c⁺c
    )
    H += InfiniteMPOHamiltonian(
        spaces, (3,1) => -t * ops.c⁺c
    )

    H += InfiniteMPOHamiltonian(
        spaces, (3,5) => -t * ops.c⁺c
    )
    H += InfiniteMPOHamiltonian(
        spaces, (5,3) => -t * ops.c⁺c
    )

    # Hubbard U (single-band, on-site)
    for ((i,j,k,l), U_ijkl) in pairs(calc.model.U)
        # on-site term only
        if i == j == k == l == 1
            H += InfiniteMPOHamiltonian(
                spaces, (1,) => U_ijkl * ops.n_pair
            )
            H += InfiniteMPOHamiltonian(
                spaces, (3,) => U_ijkl * ops.n_pair
            )
        end
    end

    # phonon energy
    H += InfiniteMPOHamiltonian(spaces, (2,) => ω0 * nb)
    H += InfiniteMPOHamiltonian(spaces, (4,) => ω0 * nb)

    # Holstein coupling
    H += InfiniteMPOHamiltonian(
        spaces, (1,2) => g * (ops.n - id(Ps)) ⊗ (bmin + bplus)
    )
    H += InfiniteMPOHamiltonian(
        spaces, (3,4) => g * (ops.n - id(Ps)) ⊗ (bmin + bplus)
    )
    return H
end

