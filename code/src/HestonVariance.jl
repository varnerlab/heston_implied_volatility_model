"""
    HestonVariance.jl

Modified Heston variance process simulation:
    dv = κ(θ(t) - v)dt + σ_v √v dW_v

with reflecting boundary at v = 0 and time-varying θ(t).
"""

using Random

"""
    simulate_variance(params, θ_func, hmm_states, S_path, contract, mood_path; Δt, rng) → Vector{Float64}

Simulate the modified Heston variance process along a single JumpHMM price path.

The initial variance v₀ is set to θ(t=0) — the mean-reversion target at the
first timestep — so each (ticker, strike, DTE) triple starts at its own
equilibrium IV. This gives the initial IV surface smile, skew, and term
structure automatically.

The variance process is discretized via Euler-Maruyama with a reflecting boundary:
    v_{t+1} = |v_t + κ(θ_t - v_t)Δt + σ_v·√(max(v_t,0))·√Δt·Z|

# Arguments
- `params::HestonParameters`: κ, σ_v
- `θ_func::ThetaHybrid`: hybrid θ-function parameters
- `hmm_states::Vector{Int}`: HMM state sequence from JumpHMM simulation
- `S_path::Vector{Float64}`: underlying price path
- `contract::OptionContract`: option contract (provides K and DTE)
- `mood_path::Vector{Float64}`: market mood at each timestep
- `Δt::Float64`: time step in years (default 1/252)
- `rng`: random number generator

# Returns
Vector of variance values v_t at each timestep. Take √v_t to get implied volatility.
"""
function simulate_variance(params::HestonParameters, θ_func::ThetaHybrid,
                           hmm_states::Vector{Int}, S_path::Vector{Float64},
                           contract::OptionContract, mood_path::Vector{Float64};
                           Δt::Float64=1.0/252.0,
                           rng::AbstractRNG=Random.default_rng())::Vector{Float64}
    n_steps = length(hmm_states)
    v = Vector{Float64}(undef, n_steps)

    # Initialize v₀ = θ(t=0): start at equilibrium for this contract
    remaining_DTE_0 = Float64(contract.DTE)
    moneyness_0 = contract.K / S_path[1]
    v[1] = compute_theta(θ_func, hmm_states[1], remaining_DTE_0,
                         moneyness_0, mood_path[1])

    sqrt_Δt = sqrt(Δt)

    for t in 1:(n_steps - 1)
        # Remaining DTE decreases as we move forward
        remaining_DTE = max(contract.DTE - t + 1, 1)
        moneyness = contract.K / S_path[t]

        # Compute time-varying θ
        θ_t = compute_theta(θ_func, hmm_states[t], Float64(remaining_DTE),
                            moneyness, mood_path[t])

        # Euler-Maruyama step with reflecting boundary
        v_curr = max(v[t], 0.0)
        dv = params.κ * (θ_t - v_curr) * Δt + params.σ_v * sqrt(v_curr) * sqrt_Δt * randn(rng)
        v[t + 1] = abs(v_curr + dv)
    end

    return v
end

"""
    simulate_variance_ensemble(params, θ_func, hmm_states_matrix, S_paths_matrix,
                                contract, mood_path; n_var_paths, Δt, rng) → Matrix{Float64}

Simulate multiple variance paths for each synthetic price path.

For scenario analysis, each JumpHMM price path can generate multiple variance
realizations (since dW_v is independent of the price path noise).

# Arguments
- `hmm_states_matrix`: n_paths × n_steps matrix of HMM states
- `S_paths_matrix`: n_paths × n_steps matrix of underlying prices
- `mood_path`: n_steps vector of market mood (shared across paths for a given simulation)
- `n_var_paths::Int`: number of variance paths per price path

# Returns
(n_paths * n_var_paths) × n_steps matrix of variance values.
"""
function simulate_variance_ensemble(params::HestonParameters, θ_func::ThetaHybrid,
                                    hmm_states_matrix::Matrix{Int},
                                    S_paths_matrix::Matrix{Float64},
                                    contract::OptionContract,
                                    mood_path::Vector{Float64};
                                    n_var_paths::Int=1,
                                    Δt::Float64=1.0/252.0,
                                    rng::AbstractRNG=Random.default_rng())::Matrix{Float64}
    n_price_paths, n_steps = size(S_paths_matrix)
    total_paths = n_price_paths * n_var_paths
    V = Matrix{Float64}(undef, total_paths, n_steps)

    idx = 1
    for i in 1:n_price_paths
        states_i = view(hmm_states_matrix, i, :) |> collect
        S_i = view(S_paths_matrix, i, :) |> collect
        for _ in 1:n_var_paths
            V[idx, :] .= simulate_variance(params, θ_func, states_i, S_i,
                                           contract, mood_path; Δt=Δt, rng=rng)
            idx += 1
        end
    end

    return V
end

"""
    variance_to_iv(v) → Float64

Convert variance to implied volatility: σ_imp = √v.
"""
variance_to_iv(v::Float64)::Float64 = sqrt(max(v, 0.0))

"""
    variance_path_to_iv(v_path) → Vector{Float64}

Convert a variance path to an IV path.
"""
variance_path_to_iv(v_path::Vector{Float64})::Vector{Float64} = [variance_to_iv(v) for v in v_path]
