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
function compute_sites(indices::NTuple{N,Int}, key::Symbol) where {N}
    letters = collect(string(key))
    unique_letters = unique(letters)

    selected_indices = [indices[findfirst(==(letter), letters)] for letter in unique_letters]

    return tuple((idx for idx in selected_indices)...)
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
function two_body_int_cached(ops, (i,j,k,l)::NTuple{4,Int})
    key = pattern_key(i,j,k,l)

    if haskey(two_body_cache, key)
        operator = two_body_cache[key]
    else
        operator = two_body_int(ops, Val(key))
        two_body_cache[key] = operator
    end
    sites = compute_sites((i,j,k,l), key)

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
function three_body_int_cached(ops, (i,j,k,l,m,n)::NTuple{6,Int})
    key = pattern_key(i,j,k,l,m,n)

    if haskey(three_body_cache, key)
        operator = three_body_cache[key]
    else
        operator = three_body_int(ops, Val(key))
        three_body_cache[key] = operator
    end
    sites = compute_sites((i,j,k,l), key)

    return operator, sites
end


############################
# Hamiltonian construction #
############################

# Build the symmetry-dependent operators
function build_ops(symm::SymmetryConfig, bands::Int64, max_b::Int64, nmodes::Int64)
    ps = symm.particle_symmetry
    ss = symm.spin_symmetry
    fill = symm.filling

    electron_space = hubbard_space(ps, ss; filling=fill)

    ops = (
        c⁺c      = c_plusmin(ps, ss; filling=fill),
        n_pair   = number_pair(ps, ss; filling=fill),
        n        = number_e(ps, ss; filling=fill)        
    )
    if ss !== SU2Irrep
        ops = merge(ops, (Sz = Sz(ps, ss; filling=fill),))
    end
    if ps === Trivial
        ops = merge(ops, (c⁺pair = create_pair_onesite(ps, ss; filling=fill), cpair = delete_pair_onesite(ps, ss; filling=fill)))
    end

    phonon_space = []
    if max_b > 0
        ops = merge(ops, (bmin = b_min(ps, ss, max_b; filling=fill),
                          bplus = b_plus(ps, ss, max_b; filling=fill),
                          nb = number_b(ps, ss, max_b; filling=fill)))

        phonon_space = boson_space(ps, ss, max_b; filling=fill)
        phonon_spaces = [phonon_space for _ in 1:nmodes]
    end

    electron_spaces = [electron_space for _ in 1:bands]
    spaces = append!(electron_spaces, phonon_spaces)

    return ops, repeat(spaces, symm.cell_width)
end

"""
    hamiltonian(calc::CalcConfig)

Constructs the many-body Hamiltonian for a system defined by configuration `calc`.

# Notes
- Lattice sites are represented using an `InfiniteChain` of length `cell_width * bands`.
- The resulting MPO can be used directly for DMRG, VUMPS, or other tensor network calculations.
"""
function hamiltonian(calc::CalcConfig{T}) where {T<:AbstractFloat}
    empty!(two_body_cache)
    empty!(three_body_cache)

    bands = calc.hubbard.bands
    t = calc.hubbard.t
    U = calc.hubbard.U

    idx = findfirst(t -> t isa HolsteinTerm, calc.terms)
    max_b = (idx === nothing ? 0 : calc.terms[idx].max_b)
    boson_site = Int(max_b>0)
    w = (idx === nothing ? T[] : calc.terms[idx].w)
    g = (idx === nothing ? T[] : calc.terms[idx].g)
    @assert size(g,2) == length(w) "w and g must have the same length (number of phonon modes)"
    boson_site = boson_site*length(w)
    period = bands + boson_site

    symm = calc.symmetries
    ops, spaces = build_ops(symm, bands, max_b, length(w))
    cell_width = symm.cell_width

    h::Vector{Pair{Tuple{Vararg{Int64}}, Any}} = [(1,) => 0*ops.n]  # Initialize MPO

    # --- Hopping ---
    for cell in 0:(cell_width-1)
        site(i) = i + cell*period + div(i-1, bands)*boson_site
        h = append!(h, [site.((i,j)) => -t_ij*ops.c⁺c for ((i,j), t_ij) in t if i != j])
        h = append!(h, [(site(i),) => -μ_i*ops.n for ((i,j), μ_i) in t if i == j])
    end

    # --- 2-body Interaction ---
    for cell in 0:(cell_width-1)
        site(i) = i + cell*period + div(i-1, bands)*boson_site
        h = append!(h, [
            begin 
                operator, indices = two_body_int_cached(ops, site.((i,j,k,l)))
                indices => 0.5 * U_ijkl * operator
            end for ((i,j,k,l), U_ijkl) in U
        ])
    end

    # --- Extra terms ---
    for term in calc.terms
        h = append!(h, hamiltonian_term(term, ops, cell_width, bands, boson_site))
    end

    return InfiniteMPOHamiltonian(spaces, h...)
end

# Three-body interaction term
function hamiltonian_term(
                    term::ThreeBodyTerm, 
                    ops, 
                    cell_width::Int64,
                    bands::Int64,
                    boson_site::Int64
                )
    V = term.V
    period = bands + boson_site

    h = []

    for cell in 0:(cell_width-1)
        site(i) = i + cell*period + div(i-1, bands)*boson_site
        h = append!(h, [
            begin
                operator, indices = three_body_int_cached(ops, site.((i,j,k,l,m,n)))
                indices => 1/6 * V_ijklmn * operator
            end for ((i,j,k,l,m,n), V_ijklmn) in V
        ])
    end

    return h
end
# Magnetic field term
function hamiltonian_term(
                    term::MagneticField, 
                    ops, 
                    cell_width::Int64,
                    bands::Int64,
                    boson_site::Int64
                )
    B = term.B
    period = bands + boson_site

    return [(i,) => -B*ops.Sz for i in 1:period*cell_width if (i%period != 0 || period==bands)]
end
# Staggered magnetic field term
function hamiltonian_term(
                    term::StaggeredField, 
                    ops, 
                    cell_width::Int64,
                    bands::Int64,
                    boson_site::Int64
                )
    J = term.J
    Ms = term.Ms
    period = bands + boson_site
    phase = (-1) .^ (div.(0:(period*cell_width-1), period))

    return [(i,) => 2*J*Ms * phase[i] * ops.Sz for i in 1:period*cell_width if (i%period != 0 || period==bands)]
end
# Holstein coupling term
function hamiltonian_term(
                    term::HolsteinTerm, 
                    ops, 
                    cell_width::Int64,
                    bands::Int64,
                    boson_site::Int64
                )

    w = term.w
    g = term.g
    mean_ne = term.mean_ne

    period = bands + boson_site

    electron_sites = [i for i in 1:cell_width*period if i%period == 1]

    phonon_sites_by_mode = [Int64[] for _ in 1:length(w)]
    for c in 0:(cell_width-1)
        base = c * period
        for m in 1:length(w)
            i  = base + bands + m
            append!(phonon_sites_by_mode[m], i)
        end
    end
    println(phonon_sites_by_mode)

    # onsite phonon terms: ω_m * nb on each phonon site of mode m
    h::Vector{Pair{Tuple{Vararg{Int64}}, Any}} = vcat([
        [(j,) => w[m] * ops.nb for j in phonon_sites_by_mode[m]]
        for m in 1:length(w)
    ]...)

    # coupling terms: g_m couples electrons to phonons of mode m in the same cell
    return append!(h, vcat([
        [
            (i,j) => g[m] * (ops.n - mean_ne*id(domain(ops.n))) ⊗ (ops.bmin + ops.bplus)
            for i in electron_sites
            for j in phonon_sites_by_mode[m]
            if j - period < i < j
        ]
        for m in 1:length(w)
    ]...))
end
# Bollmark term
function hamiltonian_term(
                    term::Bollmark, 
                    ops, 
                    cell_width::Int64,
                    bands::Int64,
                    boson_site::Int64
                )

    alpha = term.alpha
    a0, a01 = alpha[1], alpha[2]
    beta = term.beta
    b0, b01 = beta

    period = bands + boson_site
    electron_sites = [i for i in 1:period*cell_width if (i%period != 0 || period==bands)]

    hopping_onsite = ops.c⁺pair + ops.cpair
    hopping_pair = HubbardOperators.d_plus_u_plus(ComplexF64,Trivial,U1Irrep) + HubbardOperators.u_min_d_min(ComplexF64,Trivial,U1Irrep)

    h = Any[(i,) => -a0*hopping_onsite for i in electron_sites]

    h = append!(h, [
        (electron_sites[n], electron_sites[n+1]) => -a01*hopping_pair
        for n in 1:(length(electron_sites)-1)
    ])
    #h = append!(h, [(i,) => b0 * ops.n for i in electron_sites])  #This chemical potential renormalization would change the density in our system, Bollmark et al., Phys. Rev. X 13, 011039 (2023) also neglect this term.
    h = append!(h, [
        (electron_sites[n], electron_sites[n+1]) => b01 * ops.c⁺c
        for n in 1:(length(electron_sites)-1)
    ])
    h = append!(h, [
        (electron_sites[n+1], electron_sites[n]) => b01 * ops.c⁺c
        for n in 1:(length(electron_sites)-1)
    ])

    return h
end
