"""
    ThetaFunction.jl

Hybrid θ-function implementation:
    θ(t) = θ_{s_t} · (1 + γ·M_t) · ψ(DTE, K/S_t)
"""

"""
    ψ(β, DTE, moneyness) → Float64

Term-structure, skew, and smile adjustment:
    ψ = exp(β₁·ln(DTE) + β₂·ln(K/S) + β₃·ln(DTE)·ln(K/S) + β₄·(ln(K/S))²)

- β₁: term structure decay (positive = contango, IV decays toward expiration)
- β₂: skew (negative = put skew, OTM puts have higher IV)
- β₃: interaction (skew flattens at longer maturities)
- β₄: smile curvature (positive = U-shape, both wings have elevated IV)
"""
function ψ(β::Vector{Float64}, DTE::Float64, moneyness::Float64)::Float64
    log_DTE = log(max(DTE, 1.0))  # floor at 1 day to avoid log(0)
    log_m = log(moneyness)
    β₄ = length(β) >= 4 ? β[4] : 0.0  # backward compatible with 3-element β
    return exp(β[1] * log_DTE + β[2] * log_m + β[3] * log_DTE * log_m + β₄ * log_m^2)
end

"""
    compute_theta(θ_hybrid, s_t, DTE, moneyness, mood) → Float64

Compute the time-varying mean-reversion target:
    θ(t) = θ_{s_t} · (1 + γ·M_t) · ψ(DTE, K/S_t)

# Arguments
- `θ_hybrid::ThetaHybrid`: the hybrid θ-function parameters
- `s_t::Int`: current HMM state index (1-based)
- `DTE::Float64`: days to expiration
- `moneyness::Float64`: K/S_t ratio
- `mood::Float64`: aggregate market mood ∈ [0, 1]
"""
function compute_theta(θ_hybrid::ThetaHybrid, s_t::Int, DTE::Float64,
                       moneyness::Float64, mood::Float64)::Float64
    θ_base = θ_hybrid.θ_states[s_t]
    mood_factor = 1.0 + θ_hybrid.γ * mood
    ψ_factor = ψ(θ_hybrid.β, DTE, moneyness)
    return θ_base * mood_factor * ψ_factor
end

"""
    compute_mood(states, n_states, n_tail) → Float64

Compute aggregate market mood as the fraction of tickers currently in tail states.

# Arguments
- `states::Vector{Int}`: current HMM state index for each ticker
- `n_states::Int`: total number of HMM states
- `n_tail::Int`: number of states at each tail considered "extreme"

Returns a value in [0, 1] where 0 = no tickers in tail states, 1 = all tickers in tails.
"""
function compute_mood(states::AbstractVector{<:Integer}, n_states::Int, n_tail::Int)::Float64
    n_tickers = length(states)
    n_tickers == 0 && return 0.0
    count = 0
    for s in states
        if s <= n_tail || s > n_states - n_tail
            count += 1
        end
    end
    return count / n_tickers
end

"""
    compute_mood_path(state_matrix, n_states, n_tail) → Vector{Float64}

Compute market mood at each timestep from a matrix of HMM states.

# Arguments
- `state_matrix`: n_tickers × n_steps matrix of HMM state indices
- `n_states::Int`: total number of HMM states
- `n_tail::Int`: number of tail states at each end
"""
function compute_mood_path(state_matrix::Matrix{Int}, n_states::Int, n_tail::Int)::Vector{Float64}
    n_steps = size(state_matrix, 2)
    mood = Vector{Float64}(undef, n_steps)
    for t in 1:n_steps
        mood[t] = compute_mood(view(state_matrix, :, t), n_states, n_tail)
    end
    return mood
end

"""
    auto_calibrate_theta_states(market_model, prices; rf, dt) → Vector{Float64}

Compute θ_states from the empirical variance of returns in each HMM state.

Decodes the historical price series into HMM states using the market model's partition,
then computes mean(G²) for each state. States with no observations fall back to the
unconditional variance.

# Arguments
- `market_model`: a fitted JumpHiddenMarkovModel (e.g., the SPY market model)
- `prices::AbstractVector{Float64}`: historical close prices for the market ticker

# Keyword Arguments
- `rf::Float64`: risk-free rate (default: market_model.rf)
- `dt::Float64`: time step in years (default: market_model.dt)

# Returns
Vector of length N_states, where θ_states[s] = annualized realized variance when market
is in state s (suitable for IV = √θ). Converted from growth-rate units via multiplication
by dt.
"""
function auto_calibrate_theta_states(market_model, prices::AbstractVector{Float64};
                                     rf::Union{Float64,Nothing}=nothing,
                                     dt::Union{Float64,Nothing}=nothing)::Vector{Float64}
    rf_val = rf !== nothing ? rf : market_model.rf
    dt_val = dt !== nothing ? dt : market_model.dt

    G = JumpHMM.excess_growth_rates(prices; rf=rf_val, dt=dt_val)
    states = JumpHMM.assign_states(market_model.partition, G)

    N_states = market_model.partition.N
    unconditional_var = sum(G .^ 2) / length(G) * dt_val

    θ_states = Vector{Float64}(undef, N_states)
    for s in 1:N_states
        mask = states .== s
        n_obs = sum(mask)
        if n_obs > 0
            θ_states[s] = sum(G[mask] .^ 2) / n_obs * dt_val
        else
            θ_states[s] = unconditional_var
        end
    end

    return θ_states
end
