##########################
# Symmetry configuration #
##########################

"""
    SymmetryConfig(particle_symmetry, spin_symmetry; cell_width=1, filling=nothing)

Represents the symmetry configuration of a lattice system, including particle and spin symmetries,
unit cell width, and optional filling information.

# Fields
- `particle_symmetry` : Union{Type{Trivial}, Type{U1Irrep}, Type{SU2Irrep}}
    - The symmetry type for particle number. Use `Trivial` for no symmetry, `U1Irrep` for U(1) symmetry, or `SU2Irrep` for SU(2) symmetry.
- `spin_symmetry` : Union{Type{Trivial}, Type{U1Irrep}, Type{SU2Irrep}}
    - The symmetry type for spin degrees of freedom.
- `cell_width` : Int64
    - Number of sites in the unit cell. Must be a positive integer. Defaults to 1.
- `filling` : Union{Nothing, Tuple{Int64, Int64}}
    - Optional particle filling specified as a fraction `(numerator, denominator)`. Only allowed if `particle_symmetry` is `U1Irrep`. 
    Otherwise the filling is determined by the chemical potential.

# Constructor Behavior
- If `filling` is provided, the constructor checks that `particle_symmetry == U1Irrep`.
- `cell_width` must be positive.
- If `particle_symmetry` is `U1Irrep` and `filling` is specified, the constructor ensures that `cell_width` is a multiple of
  `filling[2] * (mod(filling[1], 2) + 1)` to accommodate the specified filling.

# Examples
    # Trivial particle and spin symmetry, default cell width
    cfg1 = SymmetryConfig(Trivial, Trivial)

    # U(1) particle symmetry with SU(2) spin symmetry, cell width 2, filling 1/2
    cfg2 = SymmetryConfig(U1Irrep, SU2Irrep, cell_width=2, filling=(1,2))
"""
struct SymmetryConfig
    particle_symmetry::Union{Type{Trivial},Type{U1Irrep},Type{SU2Irrep}}
    spin_symmetry::Union{Type{Trivial},Type{U1Irrep},Type{SU2Irrep}}
    cell_width::Int64
    filling::Union{Nothing,Tuple{Int64,Int64}}

    function SymmetryConfig(
                particle_symmetry::Union{Type{Trivial},Type{U1Irrep},Type{SU2Irrep}},
                spin_symmetry::Union{Type{Trivial},Type{U1Irrep},Type{SU2Irrep}},
                cell_width::Int64,
                filling::Union{Nothing,Tuple{Int64,Int64}}=nothing
            )
        @assert cell_width > 0 "Cell width must be a positive integer"

        if particle_symmetry == U1Irrep
            filling = filling === nothing ? (1, 1) : filling
            numerator, denominator = filling
            @assert numerator > 0 && denominator > 0 "Filling components must be positive integers"
            necessary_width = denominator * (mod(numerator, 2) + 1)
            @assert cell_width % necessary_width == 0 "Cell width ($cell_width) must be a multiple of $necessary_width \
                                                       to accommodate the specified filling ($numerator / $denominator)"
        elseif filling !== nothing
            error("Filling can only be specified when particle symmetry is U1Irrep, but got $(particle_symmetry).")
        end

        new(particle_symmetry, spin_symmetry, cell_width, filling)
    end
end


####################
# Model parameters #
####################

# Convert hopping matrix to dictionary representation
function hopping_matrix2dict(t::Union{Vector{T}, Matrix{T}}) where {T<:AbstractFloat}
    hopping = Dict{NTuple{2,Int},T}()
    bands = isa(t, Matrix) ? size(t,1) : 1
    num_sites = isa(t, Matrix) ? size(t,2) ÷ bands : length(t)

    @assert (isa(t, Matrix) && size(t,1) == bands) || isa(t, Vector) "First dimension of t ($(size(t,1))) must be equal to number of bands ($bands)"
    @assert (isa(t, Matrix) && size(t,2) % bands == 0) || isa(t, Vector) "Second dimension of t ($(size(t,2))) must be a multiple of number of bands ($bands)"
    @assert ishermitian(t[1:bands, 1:bands]) "t on-site matrix is not Hermitian"

    for i in 1:bands
        for j in 1:(num_sites*bands)
            if isa(t, Matrix)
                if t[i, j] != 0.0
                    hopping[(i, j)] = t[i, j]
                    hopping[(j, i)] = conj(t[i, j])
                end
            else
                if t[j] != 0.0
                    hopping[(1,j)] = t[j]
                    hopping[(j,1)] = conj(t[j])
                end
            end
        end
    end

    return hopping
end
# Add 2-body interaction term and its Hermitian conjugate
function addU!(U::Dict{NTuple{4,Int},T}, key::NTuple{4,Int}, val::T) where {T}
    if val != 0
        U[key] = val
        i,j,k,l = key
        U[(l,k,j,i)] = conj(val)          # Hermitian conjugate term
    end
end
# Add 3-body interaction term and its Hermitian conjugate
function addV!(V::Dict{NTuple{6,Int},T}, key::NTuple{6,Int}, val::T) where {T}
    if val != 0
        V[key] = val
        i,j,k,l,n,m = key
        V[(m,n,l,k,j,i)] = conj(val)          # Hermitian conjugate term
    end
end
# Make
function three_body_term(i::Int64, j::Int64, value::T, V::Dict{NTuple{6,Int},T}=Dict{NTuple{6,Int},T}()) where {T<:AbstractFloat}
    addV!(V, (i,i,j,j,i,i), value)
    addV!(V, (i,j,i,i,j,i), value)
    addV!(V, (j,i,i,i,i,j), value)
    addV!(V, (i,j,j,j,j,i), value)
    addV!(V, (j,i,j,j,i,j), value)
    addV!(V, (j,j,i,i,j,j), value)
    return V
end

"""
    ModelParams{T<:AbstractFloat}

Represents the Hamiltonian parameters for a lattice or multi-orbital system,
including hopping terms, two-body, and three-body interactions.

# Fields
- `bands::Int64`  
    Number of orbitals or bands per unit cell. Must be positive.
- `t::Dict{NTuple{2, Int64}, T}`  
    Hopping amplitudes between sites. Convention: `t[(i,i)] = μ_i` is the on-site potential,
    `t[(i,j)]` for `i ≠ j` is the hopping amplitude from site i to j.
- `U::Dict{NTuple{4, Int64}, T}`  
    Two-body interaction tensor. Entries `U[(i,j,k,l)]` correspond to the operator
    c⁺_i c⁺_j c_k c_l. Zero entries can be omitted from the dictionary.
- `V::Dict{NTuple{6, Int64}, T}`  
    Three-body interaction tensor. Entries `V[(i,j,k,l,m,n)]` correspond to the operator
    c⁺_i c⁺_j c⁺_k c_l c_m c_n. Zero entries can be omitted from the dictionary.

# Constructors

1. `ModelParams(bands::Int64, t::Dict{NTuple{2, Int64}, T}, U::Dict{NTuple{4,Int},T})`  
   Standard constructor with explicit number of bands, hopping dictionary, and interaction dictionary. `V` is empty by default.
2. `ModelParams(bands::Int64, t::Dict{NTuple{2, Int64}, T}, U::Dict{NTuple{4,Int},T}, V::Dict{NTuple{6, Int64},T})`  
   Full constructor including three-body interactions.
3. `ModelParams(t::Vector{T}, U::Dict{NTuple{4,Int},T})`  
   Single-band constructor from hopping vector and two-body interaction dictionary.
4. `ModelParams(t::Vector{T}, U::Vector{T})`  
   Single-band constructor from hopping vector and a vector of on-site interactions. Converts the vector automatically into the interaction dictionary.
5. `ModelParams(t::Matrix{T}, U::Matrix{T})`  
   Multi-band constructor from hopping matrix and two-body interaction matrix. Checks Hermiticity and dimensions, converts into internal dictionary format.
6. `ModelParams(t::Vector{T}, U::Vector{T}, V::Vector{T})`  
   Single-band constructor including three-body interactions. Converts vectors into dictionaries internally.
7. `ModelParams(t::Matrix{T}, U::Matrix{T}, V::Matrix{T})`  
   Multi-band constructor including three-body interactions. Converts matrices into dictionaries and checks Hermiticity.

# Notes
- Hopping and interaction arrays or vectors are automatically converted into the internal `Dict` representation.
- Hermiticity of on-site interaction matrices is asserted when applicable.
- The interaction dictionaries are structured for direct use in many-body Hamiltonian construction.
- Zero entries in `U` or `V` can be omitted from the dictionaries.
"""
struct ModelParams{T<:AbstractFloat}
    bands::Int64
    t::Dict{NTuple{2, Int64}, T}          # t_ii=µ_i, t_ij hopping i→j
    U::Dict{NTuple{4, Int64}, T}          # U_ijkl c⁺_i c⁺_j c_k c_l
    V::Dict{NTuple{6, Int64}, T}          # 3-body interaction V_ijklmn c⁺_i c⁺_j c⁺_k c_l c_m c_n
    J_M0::NTuple{2, T}                    # Staggered magnetic interaction J with initial magnetization M0

    function ModelParams(bands::Int64, t::Dict{NTuple{2,Int64}, T}, U::Dict{NTuple{4,Int},T}) where {T<:AbstractFloat}
        @assert bands > 0 "Number of bands must be a positive integer"
        new{T}(bands, t, U, Dict(), (0.0,0.0))
    end
    function ModelParams(bands::Int64, t::Dict{NTuple{2,Int64}, T}, 
                        U::Dict{NTuple{4,Int},T}, V::Dict{NTuple{6, Int64}, T}) where {T<:AbstractFloat}
        @assert bands > 0 "Number of bands must be a positive integer"
        new{T}(bands, t, U, V, (0.0,0.0))
    end
    function ModelParams(bands::Int64, t::Dict{NTuple{2,Int64}, T}, 
                        U::Dict{NTuple{4,Int},T}, J_M0::NTuple{2, T}) where {T<:AbstractFloat}
        @assert bands > 0 "Number of bands must be a positive integer"
        new{T}(bands, t, U, Dict(), J_M0)
    end
end
# Constructors
function ModelParams(t::Union{Vector{T}, Matrix{T}}, U::Dict{NTuple{4,Int},T}) where {T<:AbstractFloat}
    bands = isa(t, Matrix) ? size(t,1) : 1
    return ModelParams(bands, hopping_matrix2dict(t), U)
end
function ModelParams(t::Vector{T}, U::Vector{T}; J_M0::NTuple{2, T}=(0.0,0.0)) where {T<:AbstractFloat}
    interaction = Dict{NTuple{4,Int},T}()
    for (i, val) in enumerate(U)
        addU!(interaction, (1,i,i,1), val)
        addU!(interaction, (i,1,1,i), val)  # double counting: factor 1/2 added later
    end
    return ModelParams(1, hopping_matrix2dict(t), interaction, J_M0)
end
function ModelParams(t::Matrix{T}, U::Matrix{T}) where {T<:AbstractFloat}
    bands = size(t, 1)

    # --- basic checks ---
    @assert size(U, 1) == bands "First dimension of U ($(size(U,1))) must be equal to number of bands ($bands)"
    @assert size(U, 2) % bands == 0 "Second dimension of U ($(size(U,2))) must be multiple of number of bands ($bands)"
    @assert ishermitian(U[1:bands, 1:bands]) "U on-site matrix is not Hermitian"

    interaction = Dict{NTuple{4,Int},T}()
    for i in 1:bands
        for j in 1:size(U,2)
            addU!(interaction, (i,j,j,i), U[i,j])
            j>bands && addU!(interaction, (j,i,i,j), U[i,j])
        end
    end

    return ModelParams(bands, hopping_matrix2dict(t), interaction)
end
function ModelParams(t::Vector{T}, U::Vector{T}, V::Vector{T}) where {T<:AbstractFloat}
    model_base = ModelParams(t, U)
    threebody = Dict{NTuple{6,Int},T}()
    for (i, val) in enumerate(V)
        threebody = three_body_term(1, i+1, val, threebody)
    end

    return ModelParams(model_base.bands, model_base.t, model_base.U, threebody)
end
function ModelParams(t::Matrix{T}, U::Matrix{T}, V::Matrix{T}) where {T<:AbstractFloat}
    model_base = ModelParams(t, U)
    bands = model_base.bands
    @assert size(V, 1) == bands "First dimension of V ($(size(V,1))) must be equal to number of bands ($bands)"
    @assert size(V, 2) % bands == 0 "Second dimension of V ($(size(V,2))) must be multiple of number of bands ($bands)"
    @assert ishermitian(V[1:bands, 1:bands]) "V on-site matrix is not Hermitian"

    threebody = Dict{NTuple{6,Int},T}()
    for i in 1:bands
        for j in 1:size(V,2)
            threebody = three_body_term(i, j, V[i,j], threebody)
        end
    end

    return ModelParams(model_base.bands, model_base.t, model_base.U, threebody)
end

##########################
# Hubbard Holstein model #
##########################

struct HolsteinParams{T<:AbstractFloat}
    bands::Int64
    t::Dict{NTuple{2, Int64}, T}          # t_ii=µ_i, t_ij hopping i→j
    U::Dict{NTuple{4, Int64}, T}          # U_ijkl c⁺_i c⁺_j c_k c_l
    V::Dict{NTuple{6, Int64}, T}          # 3-body interaction V_ijklmn c⁺_i c⁺_j c⁺_k c_l c_m c_n
    W_G_cutoff::NTuple{3, T}              # phonon frequency omega, electron phonon coupling strength, cutoff: max allowed phonons per site

    function HolsteinParams(bands::Int64, t::Dict{NTuple{2,Int64}, T}, U::Dict{NTuple{4,Int},T}) where {T<:AbstractFloat}
        @assert bands > 0 "Number of bands must be a positive integer"
        new{T}(bands, t, U, Dict(), (0.0,0.0,0.0))
    end
    function HolsteinParams(bands::Int64, t::Dict{NTuple{2,Int64}, T}, U::Dict{NTuple{4,Int},T}, W_G_cutoff::NTuple{3, T}) where {T<:AbstractFloat}
        @assert bands > 0 "Number of bands must be a positive integer"
        new{T}(bands, t, U, Dict(), W_G_cutoff)
    end
end
# Constructors
function HolsteinParams(t::Vector{T}, U::Vector{T}; W_G_cutoff::NTuple{3, T}=(0.0,0.0,0.0)) where {T<:AbstractFloat}
    interaction = Dict{NTuple{4,Int},T}()
    for (i, val) in enumerate(U)
        addU!(interaction, (1,i,i,1), val)
        addU!(interaction, (i,1,1,i), val)  # double counting: factor 1/2 added later
    end
    return HolsteinParams(1, hopping_matrix2dict(t), interaction, W_G_cutoff)
end


######################
# Calculation set up #
######################

"""
    CalcConfig

Holds all configuration information for a lattice calculation, including
symmetries and model parameters.

# Fields
- `symmetries::SymmetryConfig`  
    Contains particle and spin symmetries, unit cell width, and optional filling.

- `model::ModelParams{Float64}`  
    Contains the Hamiltonian parameters: number of bands, hopping amplitudes, and two-body interactions.
"""
struct CalcConfig{M}
    symmetries::SymmetryConfig
    model::M
end