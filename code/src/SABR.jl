"""
    SABR.jl

Hagan-2002 lognormal-approximation SABR implied vol formula.

Used by `examples/calibrate_sabr_per_day.jl` to fit per-day per-ticker
per-maturity SABR baselines on the 15-date ladder corpus (Reviewer 1, 2, 3).

Convention: β fixed at 0.5 in the calibration driver (CIR-consistent with
the Heston backbone of the paper). Free parameters: (α, ρ, ν) with
constraints α > 0, |ρ| < 1, ν > 0.
"""

"""
    hagan_sabr_iv(F, K, T, α, β, ρ, ν) → Float64

Black-implied vol for a SABR(α, β, ρ, ν) model at forward `F`, strike `K`,
expiry `T` years. Handles the ATM singularity via an explicit ATM expansion.
"""
function hagan_sabr_iv(F::Float64, K::Float64, T::Float64,
                       α::Float64, β::Float64, ρ::Float64, ν::Float64)::Float64
    @assert F > 0 && K > 0 && T > 0
    @assert α > 0 && ν > 0
    @assert abs(ρ) < 1.0

    if abs(F - K) < 1e-8 * F
        # ATM expansion: σ_ATM = (α/F^(1-β)) · [1 + corrections·T]
        FmB = F^(1 - β)
        term = ((1 - β)^2 / 24) * α^2 / FmB^2 +
               (ρ * β * ν * α) / (4 * FmB) +
               ((2 - 3ρ^2) / 24) * ν^2
        return (α / FmB) * (1.0 + term * T)
    end

    logFK = log(F / K)
    FK_pow = (F * K)^((1 - β) / 2)
    z      = (ν / α) * FK_pow * logFK
    # Numerically stable x(z)
    denom = 1.0 - ρ
    xz    = log((sqrt(1.0 - 2.0 * ρ * z + z^2) + z - ρ) / denom)

    A_denom_inner = 1.0 +
                    ((1 - β)^2 / 24) * logFK^2 +
                    ((1 - β)^4 / 1920) * logFK^4
    A = α / (FK_pow * A_denom_inner)
    B = 1.0 +
        (((1 - β)^2 / 24) * α^2 / FK_pow^2 +
         (ρ * β * ν * α) / (4 * FK_pow) +
         ((2 - 3 * ρ^2) / 24) * ν^2) * T
    return A * (z / xz) * B
end
