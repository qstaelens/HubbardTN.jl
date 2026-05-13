import MatrixAlgebraKit as MAK

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


Code by Claude-code
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
    U_K, = MAK.svd_trunc(H; trunc = MAK.truncrank(K))

    # Shift-invariance:  rows of U_K span span{ (λ_k^i)_i }, so the
    # K × K operator Φ relating the row-shifted versions has eigvals λ_k.
    Φ = U_K[1:(end - 1), :] \ U_K[2:end, :]
    λ = MAK.eig_vals(Φ)

    # Recover c by least squares on the original samples.
    V = [λk^r for r in 1:M, λk in λ]
    c = V \ y

    err_est = maximum(abs.(V * c .- y))

    # Sort by |λ| descending: slowest-decaying components first.
    p = sortperm(λ, by = abs, rev = true)
    return c[p], λ[p], err_est
end
