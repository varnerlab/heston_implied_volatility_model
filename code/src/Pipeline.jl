"""
    Pipeline.jl

End-to-end scenario generation:
    JumpHMM → synthetic prices/states → Heston dv → IV → CRR → American option prices
"""

using Random

"""
    run_scenario(portfolio, heston_params, θ_func, contracts; kwargs...) → ScenarioResult

Run a full scenario analysis pipeline:

1. Simulate correlated multi-asset paths via JumpHMM PortfolioModel
2. Compute aggregate market mood at each timestep
3. For each contract × path: simulate Heston variance → extract IV
4. Price each contract via CRR at the evaluation point

# Arguments
- `portfolio::PortfolioModel`: fitted JumpHMM portfolio model
- `heston_params::HestonParameters`: κ, σ_v, v₀ for the variance process
- `θ_func::ThetaHybrid`: hybrid θ-function parameters
- `contracts::Vector{OptionContract}`: option contracts to price
- `ticker::String`: which ticker's paths to use for pricing
- `P0::Float64`: initial price for reconstructing prices from growth rates
- `n_sim_steps::Int`: number of simulation steps (default 252 = 1 year)
- `n_paths::Int`: number of Monte Carlo paths (default 1000)
- `eval_step::Int`: timestep at which to evaluate option prices (default 1 = immediate)
- `r_f::Float64`: risk-free rate (default 0.05)
- `n_crr_steps::Int`: CRR tree depth (default 200)
- `seed`: optional RNG seed
"""
function run_scenario(portfolio, heston_params::HestonParameters,
                      θ_func::ThetaHybrid,
                      contracts::Vector{OptionContract},
                      ticker::String,
                      P0::Float64;
                      n_sim_steps::Int=252,
                      n_paths::Int=1000,
                      eval_step::Int=1,
                      r_f::Float64=0.05,
                      n_crr_steps::Int=200,
                      seed::Union{Int,Nothing}=nothing)

    # Step 1: Simulate multi-asset paths via JumpHMM
    sim_result = JumpHMM.simulate(portfolio, n_sim_steps;
                                  n_paths=n_paths, seed=seed)

    # Extract paths for the target ticker
    ticker_result = sim_result.results[ticker]
    n_actual_paths = length(ticker_result.paths)

    # Get N_tail from the marginal model for mood computation
    marginal = portfolio.marginals[ticker]
    N_states = marginal.partition.N
    N_tail = marginal.jump.N_tail
    rf_model = marginal.rf
    dt_model = marginal.dt

    # Build matrices from SimulationPath objects
    states_matrix = Matrix{Int}(undef, n_actual_paths, n_sim_steps)
    obs_matrix = Matrix{Float64}(undef, n_actual_paths, n_sim_steps)
    price_matrix = Matrix{Float64}(undef, n_actual_paths, n_sim_steps + 1)

    for i in 1:n_actual_paths
        path = ticker_result.paths[i]
        states_matrix[i, :] .= path.states
        obs_matrix[i, :] .= path.observations

        # Reconstruct prices from growth rates
        prices_i = JumpHMM.prices_from_growth_rates(path.observations, P0;
                                                     rf=rf_model, dt=dt_model)
        price_matrix[i, :] .= prices_i
    end

    # Step 2: Compute market mood at each timestep
    # For multi-asset mood, collect states across all tickers
    all_tickers = sim_result.tickers
    n_tickers = length(all_tickers)

    if n_tickers > 1
        # Multi-asset: mood is fraction of tickers in tail states (averaged across paths)
        # For each path, compute mood using all tickers' states at each timestep
        # Use per-path mood for the first ticker's paths as representative
        mood_paths = Matrix{Float64}(undef, n_actual_paths, n_sim_steps)
        for p in 1:n_actual_paths
            for t in 1:n_sim_steps
                tail_count = 0
                for tk in all_tickers
                    tk_path = sim_result.results[tk].paths[p]
                    s = tk_path.states[t]
                    if s <= N_tail || s > N_states - N_tail
                        tail_count += 1
                    end
                end
                mood_paths[p, t] = tail_count / n_tickers
            end
        end
    else
        # Single-asset: mood based on own state
        mood_paths = Matrix{Float64}(undef, n_actual_paths, n_sim_steps)
        for p in 1:n_actual_paths
            for t in 1:n_sim_steps
                s = states_matrix[p, t]
                mood_paths[p, t] = (s <= N_tail || s > N_states - N_tail) ? 1.0 : 0.0
            end
        end
    end

    # Step 3: Simulate Heston variance process — one per (path, contract)
    # Each contract gets its own v₀ = θ(s₀, DTE, K/S₀, M₀)
    n_contracts = length(contracts)
    option_prices = Matrix{Float64}(undef, n_actual_paths, n_contracts)

    # Store per-contract variance/IV paths (use first contract for the result matrices)
    variance_paths_all = Array{Float64}(undef, n_actual_paths, n_sim_steps, n_contracts)
    iv_paths_all = Array{Float64}(undef, n_actual_paths, n_sim_steps, n_contracts)

    rng = seed !== nothing ? Random.MersenneTwister(seed + 42) : Random.default_rng()

    for p in 1:n_actual_paths
        states_p = states_matrix[p, :]
        S_p = price_matrix[p, 1:n_sim_steps]
        mood_p = mood_paths[p, :]
        S_eval = price_matrix[p, eval_step]

        for c in 1:n_contracts
            contract = contracts[c]

            # Each contract gets its own variance path (different v₀ per strike/DTE)
            v_path = simulate_variance(heston_params, θ_func, states_p, S_p,
                                       contract, mood_p; Δt=dt_model, rng=rng)
            variance_paths_all[p, :, c] .= v_path
            iv_paths_all[p, :, c] .= variance_to_iv.(v_path)

            σ_eval = sqrt(max(v_path[eval_step], 0.0))
            option_prices[p, c] = price_contract(S_eval, contract, σ_eval, r_f;
                                                 n_steps=n_crr_steps)
        end
    end

    return ScenarioResult(
        price_matrix[:, 1:n_sim_steps],
        variance_paths_all[:, :, 1],   # first contract's variance for diagnostics
        iv_paths_all[:, :, 1],         # first contract's IV for diagnostics
        states_matrix,
        mood_paths,
        option_prices,
        contracts
    )
end

"""
    run_single_asset_scenario(model, heston_params, θ_func, contracts, P0; kwargs...) → ScenarioResult

Convenience wrapper for single-asset scenario analysis (no copula needed).
"""
function run_single_asset_scenario(model::JumpHMM.JumpHiddenMarkovModel,
                                   heston_params::HestonParameters,
                                   θ_func::ThetaHybrid,
                                   contracts::Vector{OptionContract},
                                   P0::Float64;
                                   n_sim_steps::Int=252,
                                   n_paths::Int=1000,
                                   eval_step::Int=1,
                                   r_f::Float64=0.05,
                                   n_crr_steps::Int=200,
                                   seed::Union{Int,Nothing}=nothing)

    N_states = model.partition.N
    N_tail = model.jump.N_tail
    dt_model = model.dt
    rf_model = model.rf

    # Simulate single-asset paths
    sim_result = JumpHMM.simulate(model, n_sim_steps; n_paths=n_paths, seed=seed)
    n_actual_paths = length(sim_result.paths)

    # Build matrices
    states_matrix = Matrix{Int}(undef, n_actual_paths, n_sim_steps)
    price_matrix = Matrix{Float64}(undef, n_actual_paths, n_sim_steps)
    mood_matrix = Matrix{Float64}(undef, n_actual_paths, n_sim_steps)

    for i in 1:n_actual_paths
        path = sim_result.paths[i]
        states_matrix[i, :] .= path.states
        prices_i = JumpHMM.prices_from_growth_rates(path.observations, P0;
                                                     rf=rf_model, dt=dt_model)
        price_matrix[i, :] .= prices_i[1:n_sim_steps]

        # Single-asset mood: binary tail indicator
        for t in 1:n_sim_steps
            s = path.states[t]
            mood_matrix[i, t] = (s <= N_tail || s > N_states - N_tail) ? 1.0 : 0.0
        end
    end

    # Simulate variance and price options — one variance path per (path, contract)
    n_contracts = length(contracts)
    variance_paths = Array{Float64}(undef, n_actual_paths, n_sim_steps, n_contracts)
    iv_paths = Array{Float64}(undef, n_actual_paths, n_sim_steps, n_contracts)
    option_prices = Matrix{Float64}(undef, n_actual_paths, n_contracts)

    rng = seed !== nothing ? Random.MersenneTwister(seed + 42) : Random.default_rng()

    for p in 1:n_actual_paths
        S_eval = price_matrix[p, eval_step]

        for c in 1:n_contracts
            v_path = simulate_variance(heston_params, θ_func,
                                       states_matrix[p, :], price_matrix[p, :],
                                       contracts[c], mood_matrix[p, :];
                                       Δt=dt_model, rng=rng)
            variance_paths[p, :, c] .= v_path
            iv_paths[p, :, c] .= variance_to_iv.(v_path)

            σ_eval = sqrt(max(v_path[eval_step], 0.0))
            option_prices[p, c] = price_contract(S_eval, contracts[c], σ_eval, r_f;
                                                 n_steps=n_crr_steps)
        end
    end

    return ScenarioResult(
        price_matrix, variance_paths[:, :, 1], iv_paths[:, :, 1],
        states_matrix, mood_matrix, option_prices, contracts
    )
end

"""
    summarize(result::ScenarioResult) → Dict

Summary statistics for a scenario run.
"""
function summarize(result::ScenarioResult)
    n_paths, n_contracts = size(result.option_prices)

    summaries = Dict{Symbol,Any}()
    summaries[:n_paths] = n_paths
    summaries[:n_steps] = size(result.price_paths, 2)

    # Per-contract statistics
    contract_stats = []
    for c in 1:n_contracts
        prices = result.option_prices[:, c]
        push!(contract_stats, Dict(
            :contract => result.contracts[c],
            :mean_price => mean(prices),
            :std_price => std(prices),
            :median_price => median(prices),
            :q05 => quantile(prices, 0.05),
            :q95 => quantile(prices, 0.95),
        ))
    end
    summaries[:contracts] = contract_stats

    # IV statistics at final step
    final_iv = result.iv_paths[:, end]
    summaries[:iv_mean] = mean(final_iv)
    summaries[:iv_std] = std(final_iv)

    # Mood statistics
    summaries[:mood_mean] = mean(result.mood_paths)
    summaries[:mood_max] = maximum(result.mood_paths)

    return summaries
end
