"""
  inv_power_expsum(α, K; M=256) -> (c, λ, err_est)

Best K-term exponential fit to `r^(-α)` on the integer grid
`r = 1, 2, ..., M`, via the matrix pencil / ESPRIT method.

Builds the model
    r^(-α)  ≈  Σ_{k=1}^{K}  c_k * λ_k^r

by:
 1. sampling `y_n = n^(-α)` for n = 1, ..., M;
 2. forming the Hankel matrix H[i,j] = y[i+j-1];
 3. taking its rank-K truncated SVD;
 4. extracting `λ_k` from the shift-invariance of the dominant left
    singular vectors (an eigenvalue problem of size K × K);
 5. recovering `c_k` by least squares against the original samples.

# Arguments
- `α::Real`: positive exponent in r^(-α).
- `K::Integer`: number of exponentials. Practical sweet spot is 4 – 16.
- `M::Integer = 256`: number of samples (and the upper end of the fit
  range). For an MPO on a chain of length N, set `M ≥ N`. Must satisfy
  `M ≥ 2K + 1`.

# Returns
A NamedTuple `(c, λ, err_est)`:
- `c`, `λ`: length-K complex vectors, sorted by `|λ|` descending
  (slowest-decaying first). For completely monotone targets like
  `r^(-α)`, the optimal fit has real positive `λ ∈ (0,1)` and real `c`,
  but ill-conditioning at large K can leave small imaginary residues
  and/or conjugate pairs. Take `real.(c)`, `real.(λ)` if you want to
  force real outputs and are willing to accept the resulting tiny error.
- `err_est`: max absolute error on the sample grid r = 1, ..., M.
"""
function inv_power_expsum(α::Real, K::Int; M::Int = max(256, 2K + 1))
    α > 0     || throw(ArgumentError("α must be positive"))
    K ≥ 1     || throw(ArgumentError("K must be ≥ 1"))
    M ≥ 2K + 1 || throw(ArgumentError("need M ≥ 2K + 1; got M = $M, K = $K"))

    # Samples y_n = n^(-α), n = 1, ..., M
    y = [Float64(n)^(-α) for n in 1:M]

    # Hankel data matrix of size (M-L) × (L+1), entries H[i,j] = y[i+j-1].
    # L = M ÷ 2 is the standard pencil parameter (balances row/col counts).
    L = M ÷ 2
    H = [y[i + j - 1] for i in 1:(M - L), j in 1:(L + 1)]

    # Truncated SVD; keep top-K left singular vectors.
    U_K, = MatrixAlgebraKit.svd_trunc(H; trunc = MatrixAlgebraKit.truncrank(K))

    # Shift-invariance:  rows of U_K span span{ (λ_k^i)_i }, so the
    # K × K operator Φ relating the row-shifted versions has eigvals λ_k.
    Φ = U_K[1:(end - 1), :] \ U_K[2:end, :]
    λ = MatrixAlgebraKit.eig_vals(Φ)

    # Recover c by least squares on the original samples.
    V = [λk^r for r in 1:M, λk in λ]
    c = V \ y

    err_est = maximum(abs.(V * c .- y))

    # Sort by |λ| descending: slowest-decaying components first.
    p = sortperm(λ, by = abs, rev = true)
    return c[p], λ[p], err_est
end

"""
    exponential_mpo(spaces, sites, O, λ) -> InfiniteMPOHamiltonian

Construct an MPO for a single exponentially decaying operator term.

Builds an MPO corresponding to an exponentially weighted operator string

    Σᵣ λʳ Oᵣ

where `O` acts on the specified `sites` and is propagated through the
chain by an auxiliary virtual channel carrying the factor `λ^r`.

Internally:
 1. instantiates the local operator `O` on `sites`;
 2. extracts its endpoint tensors;
 3. constructs a bond-dimension-3 MPO with an idle, propagating, and
    terminating virtual channel;
 4. inserts the transfer term weighted by `λ` and assembles the
    resulting MPO.

# Arguments
- `spaces`: physical spaces of the lattice.
- `sites`: site specification passed to
  `MPSKit.instantiate_operator`.
- `O`: local operator to embed in the MPO.
- `λ::Number`: exponential decay factor. Must satisfy `|λ| < 1`.

# Returns
An `InfiniteMPOHamiltonian` representing the exponentially decaying
operator term.

# Notes
- Complex `λ` values are supported.
- Useful for constructing long-range Hamiltonians from exponential
  decompositions, e.g. `inv_power_expsum(α, K)`.
"""
function exponential_mpo(spaces, sites, O, λ::Number)
    @assert abs(λ) < 1 "|λ| < 1 is required for convergence."

    mpo_sites, local_ops = MPSKit.instantiate_operator(spaces, (sites => O))
    i = first(mpo_sites)
    j = last(mpo_sites)
    L = first(local_ops)
    R = last(local_ops)

    T = scalartype(O)
    S = sectortype(space(O, 1))
    Vphys = typeof(space(O, 1))
    V0 = typeof(left_virtualspace(R))(one(S) => 1)
    V = SumSpace(V0, left_virtualspace(R), V0)

    Ws = map(eachindex(spaces)) do site
        W = MPSKit.jordanmpotensortype(Vphys, T)(
            undef, V ⊗ spaces[site] ← spaces[site] ⊗ V)

        W[2, 1, 1, 2] = λ * BraidingTensor{T}(eachspace(W)[2, 1, 1, 2])

        if site == mod1(i, length(spaces))
            W[1, 1, 1, 2] = L
        end
        if site == mod1(j, length(spaces))
            W[2, 1, 1, 3] = R
        end

        return W
    end

    return InfiniteMPOHamiltonian(Ws)
end