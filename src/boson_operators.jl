#############
# Operators #
#############

"""
    boson_space(ps::Type{<:Sector}, ss::Type{<:Sector}, cutoff::Int64; kwargs...)

The bosonic physical space compatible with a Hubbard space with particle symmetry `ps` 
and spin symmetry `ss`, truncated at a maximum of `cutoff` bosons per site.
"""
function boson_space(::Type{Trivial}, ::Type{U1Irrep}, cutoff::Int64; kwargs...)
    return Vect[FermionParity ⊠ U1Irrep]((0, 0) => cutoff + 1)
end
function boson_space(::Type{U1Irrep}, ::Type{U1Irrep}, cutoff::Int64; filling::Rational{Int}=1//1)
    P = numerator(filling); Q = denominator(filling)
    return Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((0, -P, 0) => cutoff + 1, (0, 2Q-P, 0) => cutoff + 1)
end

"""
    b_plus(elt::Type{<:Number}, ps::Type{<:Sector}, ss::Type{<:Sector}, cutoff::Int64)

The truncated bosonic creation operator, with a maximum of `cutoff` bosons per site.
"""
b_plus(ps::Type{<:Sector}, ss::Type{<:Sector}, cutoff::Int64; kwargs...) = b_plus(ComplexF64, ps, ss, cutoff; kwargs...)
function b_plus(elt::Type{<:Number}, ps::Type{<:Sector}, ss::Type{<:Sector}, cutoff::Int64; kwargs...)
    pspace = boson_space(ps, ss, cutoff; kwargs...)

    b⁺ = zeros(elt, pspace ← pspace)
    for s in sectors(pspace)
        for i in 1:cutoff
            block(b⁺, s)[i+1, i] = sqrt(i)
        end
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
    for s in sectors(pspace)
        for i in 1:cutoff
            block(b⁻, s)[i, i+1] = sqrt(i)
        end
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
    for s in sectors(pspace)
        for i in 1:cutoff
            block(nb, s)[i+1, i+1] = i
        end
    end

    return nb
end
