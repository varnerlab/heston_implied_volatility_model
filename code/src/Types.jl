"""
    Types.jl

Core type definitions for the HestonIV pipeline.
"""

"""
    HestonParameters

Parameters for the Heston stochastic variance process:
    dv = κ(θ(t) - v)dt + σ_v √v dW_v

- `κ`: mean-reversion speed
- `σ_v`: vol-of-vol

Note: v₀ is not stored here — it is initialized per (ticker, strike, DTE) as
v₀ = θ(s₀, DTE, K/S₀, M₀), so the process starts at the mean-reversion target
for the current regime. This gives each contract its own initial IV automatically.
"""
struct HestonParameters
    κ::Float64
    σ_v::Float64
end

"""
    ThetaHybrid

Hybrid θ-function: θ(t) = θ_{s_t} · (1 + γ·M_t) · ψ(DTE, K/S_t)

- `θ_states`: vector of θ values, one per HMM state
- `β`: parameters for ψ(DTE, K/S):
    [β₁, β₂, β₃, β₄] → ψ = exp(β₁·ln(DTE) + β₂·ln(K/S) + β₃·ln(DTE)·ln(K/S) + β₄·(ln(K/S))²)
    β₁: term structure (DTE decay)
    β₂: skew (asymmetry)
    β₃: DTE × skew interaction (skew flattening at longer maturities)
    β₄: smile curvature (U-shape)
- `γ`: market mood sensitivity
"""
struct ThetaHybrid
    θ_states::Vector{Float64}
    β::Vector{Float64}  # [β₁, β₂, β₃, β₄]
    γ::Float64
end

"""
    OptionContract

Specification of an option contract.
"""
struct OptionContract
    K::Float64          # strike price
    DTE::Int            # days to expiration
    option_type::Symbol # :call or :put
    style::Symbol       # :american or :european
end

"""
    IVPoint

A single implied volatility observation at a specific (strike, DTE) point.
"""
struct IVPoint
    K::Float64
    DTE::Int
    σ_imp::Float64
    S::Float64          # underlying price at this point
end

"""
    ScenarioResult

Results from a scenario simulation run.

- `price_paths`: n_paths × n_steps matrix of underlying prices
- `variance_paths`: n_paths × n_steps matrix of variance process values
- `iv_paths`: n_paths × n_steps matrix of implied volatilities (√v_t)
- `hmm_states`: n_paths × n_steps matrix of HMM state indices
- `mood_paths`: n_paths × n_steps matrix of market mood values
- `option_prices`: n_paths vector of terminal option prices (or matrix for multiple contracts)
- `contracts`: the option contracts priced
"""
struct ScenarioResult
    price_paths::Matrix{Float64}
    variance_paths::Matrix{Float64}
    iv_paths::Matrix{Float64}
    hmm_states::Matrix{Int}
    mood_paths::Matrix{Float64}
    option_prices::Matrix{Float64}  # n_paths × n_contracts
    contracts::Vector{OptionContract}
end
