#############
# Operators #
#############

"""
    boson_space(ps::Type{<:Sector}, ss::Type{<:Sector}, cutoff::Int64; kwargs...)

The bosonic physical space compatible with a Hubbard space with particle symmetry `ps` 
and spin symmetry `ss`, truncated at a maximum of `cutoff` bosons per site.
"""
function boson_space(ps::Type{<:Sector}, ss::Type{<:Sector}, cutoff::Int64; kwargs...)
    space = unitspace(hubbard_space(ps, ss; kwargs...))
    return ⊕((space for _ in 0:cutoff)...)
end

"""
    b_plus(elt::Type{<:Number}, ps::Type{<:Sector}, ss::Type{<:Sector}, cutoff::Int64)

The truncated bosonic creation operator, with a maximum of `cutoff` bosons per site.
"""
b_plus(ps::Type{<:Sector}, ss::Type{<:Sector}, cutoff::Int64; kwargs...) = b_plus(ComplexF64, ps, ss, cutoff; kwargs...)
function b_plus(elt::Type{<:Number}, ps::Type{<:Sector}, ss::Type{<:Sector}, cutoff::Int64; kwargs...)
    pspace = boson_space(ps, ss, cutoff; kwargs...)

    b⁺ = zeros(elt, pspace ← pspace)
    I = sectortype(b⁺)
    charges = (0 for _ in 1:length(fieldtypes(I)[1].parameters))
    for i in 1:cutoff
        block(b⁺, I(charges...))[i + 1, i] = sqrt(i)
    end

    return b⁺
end

"""
    b_min(elt::Type{<:Number}, ps::Type{<:Sector}, ss::Type{<:Sector}, cutoff::Int64)

The truncated bosonic annihilation operator, with a maximum of `cutoff` bosons per site.
"""
b_min(ps::Type{<:Sector}, ss::Type{<:Sector}, cutoff::Int64; kwargs...) = b_min(ComplexF64, ps, ss, cutoff; kwargs...)
function b_min(elt::Type{<:Number}, ps::Type{<:Sector}, ss::Type{<:Sector}, cutoff::Int64; kwargs...)
    pspace = boson_space(ps, ss, cutoff; kwargs...)

    b⁻ = zeros(elt, pspace ← pspace)
    I = sectortype(b⁻)
    charges = (0 for _ in 1:length(fieldtypes(I)[1].parameters))
    for i in 1:cutoff
        block(b⁻, I(charges...))[i + 1, i] = sqrt(i)
    end

    return b⁻
end

"""
    number_b(elt::Type{<:Number}, ps::Type{<:Sector}, ss::Type{<:Sector}, cutoff::Int64)

The truncated bosonic number operator, with a maximum of `cutoff` bosons per site.
"""
number_b(ps::Type{<:Sector}, ss::Type{<:Sector}, cutoff::Int64; kwargs...) = number_b(ComplexF64, ps, ss, cutoff; kwargs...)
function number_b(elt::Type{<:Number}, ps::Type{<:Sector}, ss::Type{<:Sector}, cutoff::Int64; kwargs...)
    pspace = boson_space(ps, ss, cutoff; kwargs...)

    nb = zeros(elt, pspace ← pspace)
    I = sectortype(nb)
    charges = (0 for _ in 1:length(fieldtypes(I)[1].parameters))
    for i in 1:cutoff
        block(nb, I(charges...))[i + 1, i + 1] = i
    end

    return nb
end
