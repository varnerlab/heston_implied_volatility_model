"""
    LRTree.jl

Leisen-Reimer American option pricer (Peizer-Pratt method 2 inversion).

Why not CRR? The recombining CRR lattice has discrete node positions, so
bumping σ or S in finite-difference Greek calculations causes the strike K
to snap across node boundaries. The resulting V(σ) and V(S) are
piecewise-bilinear with kinks, which produces staircase/diamond-grid
aliasing in CRR-FD Vega. LR constructs the up/down probabilities from a
Peizer-Pratt inversion of the Black-Scholes d_1, d_2 so K falls at a fixed
lattice position by design at every (S, σ) configuration. Greeks are
smooth via FD at modest N (LR converges as O(1/N²) vs CRR's oscillatory
O(1/√N), so N≈201 here matches CRR accuracy at N≈1500). Requires N odd;
we bump even N up by one.
"""

function _peizer_pratt(z::Float64, n::Int)
    arg = (z / (n + 1/3 + 0.1/(n+1)))^2 * (n + 1/6)
    return 0.5 + sign(z) * sqrt(0.25 - 0.25 * exp(-arg))
end

"""
    lr_american_price(S, K, σ, r, T, N, otype; q=0.0) → Float64

Price an American option via the Leisen-Reimer lattice. `N` must be odd
(even values are silently bumped to N+1).
"""
function lr_american_price(S::Float64, K::Float64, σ::Float64,
                           r::Float64, T::Float64, N::Int,
                           otype::Symbol; q::Float64=0.0)
    N % 2 == 0 && (N += 1)
    Δt    = T / N
    disc  = exp(-r * Δt)
    σsqrtT = σ * sqrt(T)
    d1 = (log(S/K) + (r - q + 0.5*σ^2)*T) / σsqrtT
    d2 = d1 - σsqrtT
    p_bar = _peizer_pratt(d2, N)
    p     = _peizer_pratt(d1, N)
    R_over_Q = exp((r - q) * Δt)
    u = R_over_Q * p / p_bar
    d = R_over_Q * (1 - p) / (1 - p_bar)

    V = Vector{Float64}(undef, N + 1)
    @inbounds for j in 0:N
        S_T = S * u^j * d^(N-j)
        V[j+1] = otype === :call ? max(S_T - K, 0.0) : max(K - S_T, 0.0)
    end
    @inbounds for n in (N-1):-1:0
        for j in 0:n
            S_n  = S * u^j * d^(n-j)
            cont = disc * (p_bar * V[j+2] + (1 - p_bar) * V[j+1])
            intrinsic = otype === :call ? S_n - K : K - S_n
            V[j+1] = max(cont, intrinsic)
        end
    end
    return V[1]
end
