"""
    Calibration.jl

Calibrate Heston parameters (κ, σ_v, v₀) and ThetaHybrid parameters
(θ_{s_t}, β₁, β₂, β₃, γ) against historical option chain data.

Two-stage approach:
1. JumpHMM is already fitted on historical prices (external)
2. This module fits the variance process parameters to match observed IV
"""

using Optim
using Statistics

"""
    CalibrationData

Preprocessed calibration dataset: one row per (date, strike, DTE) observation.
"""
struct CalibrationData
    dates::Vector{Int}           # timestep index into the price series
    strikes::Vector{Float64}     # strike prices
    dtes::Vector{Int}            # days to expiration
    market_ivs::Vector{Float64}  # observed implied volatilities
    spot_prices::Vector{Float64} # underlying price at each observation
    hmm_states::Vector{Int}      # decoded HMM state at each observation
    moods::Vector{Float64}       # market mood at each observation
end

"""
    prepare_calibration_data(option_chains, prices, hmm_model; kwargs...) → CalibrationData

Prepare calibration data from an option chain DataFrame and historical prices.

# Arguments
- `option_chains::DataFrame`: must have columns :date_idx (Int), :strike (Float64),
  :dte (Int), :iv (Float64)
- `prices::Vector{Float64}`: historical price series for the underlying
- `hmm_model::JumpHMM.JumpHiddenMarkovModel`: fitted HMM for state decoding
- `n_tail::Int`: number of tail states for mood computation (default from model)
"""
function prepare_calibration_data(option_chains::DataFrame,
                                  prices::AbstractVector{Float64},
                                  hmm_model;
                                  n_tail::Union{Int,Nothing}=nothing)

    N_states = hmm_model.partition.N
    N_tail_actual = n_tail !== nothing ? n_tail : hmm_model.jump.N_tail
    rf = hmm_model.rf
    dt = hmm_model.dt

    # Decode HMM states from historical prices
    G = JumpHMM.excess_growth_rates(prices; rf=rf, dt=dt)
    state_sequence = JumpHMM.assign_states(hmm_model.partition, G)

    n_obs = nrow(option_chains)
    dates = Vector{Int}(undef, n_obs)
    strikes = Vector{Float64}(undef, n_obs)
    dtes = Vector{Int}(undef, n_obs)
    market_ivs = Vector{Float64}(undef, n_obs)
    spot_prices = Vector{Float64}(undef, n_obs)
    hmm_states_out = Vector{Int}(undef, n_obs)
    moods = Vector{Float64}(undef, n_obs)

    for i in 1:n_obs
        t_idx = option_chains[i, :date_idx]
        dates[i] = t_idx
        strikes[i] = option_chains[i, :strike]
        dtes[i] = option_chains[i, :dte]
        market_ivs[i] = option_chains[i, :iv]

        # Price at this date (date_idx is 1-based into original price series)
        spot_prices[i] = prices[t_idx]

        # HMM state (state_sequence is 1 shorter than prices due to differencing)
        state_idx = min(t_idx, length(state_sequence))
        hmm_states_out[i] = state_sequence[state_idx]

        # Single-asset mood: binary tail indicator at this state
        s = hmm_states_out[i]
        moods[i] = (s <= N_tail_actual || s > N_states - N_tail_actual) ? 1.0 : 0.0
    end

    return CalibrationData(dates, strikes, dtes, market_ivs,
                           spot_prices, hmm_states_out, moods)
end

"""
    calibrate(cal_data, N_states; kwargs...) → (HestonParameters, ThetaHybrid)

Calibrate the Heston + ThetaHybrid parameters to minimize IV prediction error.

Since v₀ = θ(t=0) (the process starts at equilibrium), the calibration objective
simplifies: the model-predicted IV for each observation is just √θ(s, DTE, K/S, M).
The κ parameter controls how quickly the process would mean-revert if perturbed,
and σ_v controls the stochastic spread around the target.

# Arguments
- `cal_data::CalibrationData`: preprocessed calibration data
- `N_states::Int`: number of HMM states

# Keyword Arguments
- `κ_init::Float64`: initial κ (default 5.0)
- `σv_init::Float64`: initial σ_v (default 0.3)
- `γ_init::Float64`: initial mood sensitivity (default 0.5)
- `method`: Optim method (default NelderMead())
- `maxiter::Int`: maximum iterations (default 5000)

# Returns
Tuple of (HestonParameters, ThetaHybrid) that minimize Σ(σ_model - IV_market)²
"""
function calibrate(cal_data::CalibrationData, N_states::Int;
                   κ_init::Float64=5.0,
                   σv_init::Float64=0.3,
                   γ_init::Float64=0.5,
                   method=NelderMead(),
                   maxiter::Int=5000)

    n_obs = length(cal_data.market_ivs)

    # Parameter vector layout:
    # [1]       = log(κ)          (ensures κ > 0)
    # [2]       = log(σ_v)        (ensures σ_v > 0)
    # [3]       = γ               (mood sensitivity, unconstrained)
    # [4:7]     = β₁, β₂, β₃, β₄ (ψ parameters, unconstrained)
    # [8:7+N]   = log(θ_states)   (ensures θ > 0 for each state)

    # Initialize θ_states: group observations by state and use mean IV² as θ
    θ_init = fill(0.04, N_states)
    for i in 1:n_obs
        s = cal_data.hmm_states[i]
        if 1 <= s <= N_states
            θ_init[s] = cal_data.market_ivs[i]^2
        end
    end

    n_params = 7 + N_states
    x0 = Vector{Float64}(undef, n_params)
    x0[1] = log(κ_init)
    x0[2] = log(σv_init)
    x0[3] = γ_init
    x0[4] = 0.0   # β₁ (DTE effect)
    x0[5] = 0.0   # β₂ (skew)
    x0[6] = 0.0   # β₃ (interaction)
    x0[7] = 0.0   # β₄ (smile curvature)
    x0[8:end] .= log.(max.(θ_init, 1e-8))

    function objective(x)
        κ = exp(x[1])
        σ_v = exp(x[2])
        γ = x[3]
        β = x[4:7]
        θ_states = exp.(x[8:end])

        θ_func = ThetaHybrid(θ_states, β, γ)

        total_error = 0.0
        for i in 1:n_obs
            s = cal_data.hmm_states[i]
            if s < 1 || s > N_states
                continue
            end
            DTE = Float64(cal_data.dtes[i])
            moneyness = cal_data.strikes[i] / cal_data.spot_prices[i]
            mood = cal_data.moods[i]

            # v₀ = θ(t=0), so the equilibrium IV is simply √θ
            θ_t = compute_theta(θ_func, s, DTE, moneyness, mood)
            σ_model = sqrt(max(θ_t, 1e-10))

            total_error += (σ_model - cal_data.market_ivs[i])^2
        end

        return total_error / n_obs
    end

    result = optimize(objective, x0, method,
                      Optim.Options(iterations=maxiter, show_trace=false))

    x_opt = Optim.minimizer(result)

    heston = HestonParameters(exp(x_opt[1]), exp(x_opt[2]))
    θ_func = ThetaHybrid(exp.(x_opt[8:end]), x_opt[4:7], x_opt[3])

    return heston, θ_func
end
