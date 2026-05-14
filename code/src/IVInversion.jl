"""
    IVInversion.jl

Black-Scholes implied volatility inversion via Brent's method.

Used by:
- Static-arbitrage check (B1) — invert CRR-priced surfaces back to IV
- SABR calibration (B2) — convert market mids to IV when only price is available
- Scenario template (A2) — handle ladder rows that ship price-only
"""

# Distributions is already a HestonIV dep; reuse its standard normal.
const _STDNORM = Normal(0.0, 1.0)

@inline _norm_cdf(x::Float64)::Float64 = cdf(_STDNORM, x)
@inline _norm_pdf(x::Float64)::Float64 = pdf(_STDNORM, x)

"""
    bs_price(S, K, σ, r, T, otype; q=0.0) → Float64

Black-Scholes-Merton price for European call (`:call`) or put (`:put`) with
continuous dividend yield `q`. Used as the inversion target.
"""
function bs_price(S::Float64, K::Float64, σ::Float64,
                  r::Float64, T::Float64, otype::Symbol;
                  q::Float64=0.0)::Float64
    @assert otype in (:call, :put) "otype must be :call or :put"
    if σ <= 0.0 || T <= 0.0
        intrinsic = otype === :call ? max(S - K, 0.0) : max(K - S, 0.0)
        return intrinsic * exp(-r * T)
    end
    sqrtT = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * σ * σ) * T) / (σ * sqrtT)
    d2 = d1 - σ * sqrtT
    if otype === :call
        return S * exp(-q * T) * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    else
        return K * exp(-r * T) * _norm_cdf(-d2) - S * exp(-q * T) * _norm_cdf(-d1)
    end
end

"""
    bs_vega(S, K, σ, r, T; q=0.0) → Float64

Black-Scholes vega (∂C/∂σ). Used only for the Newton fast-path inside the
inverter when a price is well-behaved.
"""
function bs_vega(S::Float64, K::Float64, σ::Float64,
                 r::Float64, T::Float64; q::Float64=0.0)::Float64
    if σ <= 0.0 || T <= 0.0
        return 0.0
    end
    sqrtT = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * σ * σ) * T) / (σ * sqrtT)
    return S * exp(-q * T) * _norm_pdf(d1) * sqrtT
end

"""
    bs_implied_vol(price, S, K, r, T, otype; q=0.0, σ_lo=1e-4, σ_hi=5.0,
                   tol=1e-7, max_iter=80) → Union{Float64,Missing}

Invert the Black-Scholes price to recover implied volatility using Brent's
method on `[σ_lo, σ_hi]`. Returns `missing` if the price is outside the
no-arbitrage band or the root is not bracketed.

Inputs and outputs are scalar Float64; vectorize at the caller. No exceptions
on bad inputs — return `missing` so calibration callers can drop the row.
"""
function bs_implied_vol(price::Float64, S::Float64, K::Float64,
                        r::Float64, T::Float64, otype::Symbol;
                        q::Float64=0.0,
                        σ_lo::Float64=1e-4, σ_hi::Float64=5.0,
                        tol::Float64=1e-7, max_iter::Int=80)::Union{Float64,Missing}
    @assert otype in (:call, :put) "otype must be :call or :put"
    if !(price > 0.0) || !(T > 0.0)
        return missing
    end
    # No-arbitrage band
    if otype === :call
        lower = max(S * exp(-q * T) - K * exp(-r * T), 0.0)
        upper = S * exp(-q * T)
    else
        lower = max(K * exp(-r * T) - S * exp(-q * T), 0.0)
        upper = K * exp(-r * T)
    end
    if price < lower - 1e-10 || price > upper + 1e-10
        return missing
    end

    f(σ) = bs_price(S, K, σ, r, T, otype; q=q) - price

    fa = f(σ_lo)
    fb = f(σ_hi)
    if fa * fb > 0.0
        # Not bracketed — widen σ_hi a few times, otherwise bail.
        b = σ_hi
        for _ in 1:4
            b *= 2.0
            fb = f(b)
            if fa * fb <= 0.0
                σ_hi = b
                break
            end
        end
        if fa * fb > 0.0
            return missing
        end
    end

    # Brent's method (Numerical Recipes formulation)
    a, b = σ_lo, σ_hi
    if abs(fa) < abs(fb)
        a, b = b, a
        fa, fb = fb, fa
    end
    c, fc = a, fa
    d = b - a
    e = d
    for _ in 1:max_iter
        if fb == 0.0 || abs(b - a) < tol
            return b
        end
        if fa != fc && fb != fc
            # Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc)) +
                b * fa * fc / ((fb - fa) * (fb - fc)) +
                c * fa * fb / ((fc - fa) * (fc - fb))
        else
            # Secant
            s = b - fb * (b - a) / (fb - fa)
        end
        cond1 = !((3a + b) / 4 < s < b) && !(b < s < (3a + b) / 4)
        cond2 = abs(s - b) >= abs(b - c) / 2
        if cond1 || cond2
            s = (a + b) / 2  # Bisection fallback
        end
        fs = f(s)
        d = c
        c, fc = b, fb
        if fa * fs < 0.0
            b, fb = s, fs
        else
            a, fa = s, fs
        end
        if abs(fa) < abs(fb)
            a, b = b, a
            fa, fb = fb, fa
        end
    end
    return abs(fb) < 1e-4 ? b : missing
end
