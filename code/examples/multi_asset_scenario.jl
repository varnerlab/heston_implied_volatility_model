"""
Multi-Asset Scenario Analysis via HybridSingleIndexModel

Uses the HybridSIM dependence model from JumpHMM.jl:
1. Fit a HybridSingleIndexModel with SPY as the market factor
2. Auto-calibrate θ_states from SPY's empirical variance per HMM state
3. Simulate correlated multi-asset paths
4. Price American options on a target semiconductor ticker
5. Compare calm vs stressed scenario pricing (binary mood from SPY state)

Data: SP500 daily OHLC from code/data/equity/ (2014-2024)
"""

using JumpHMM
using HestonIV
using JLD2
using DataFrames
using Statistics
using Random

# --- Step 1: Load equity price data ---
const DATA_PATH = joinpath(@__DIR__, "..", "data", "equity",
                           "SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2")

println("Loading equity price data...")
raw_data = JLD2.load(DATA_PATH)["dataset"]

# Extract close prices for SPY + semiconductor tickers
const TICKERS = ["SPY", "NVDA", "AMD", "MU", "INTC"]
const TARGET = "NVDA"

dfs = Dict{String,DataFrame}()
for p in raw_data
    if p.first in TICKERS
        dfs[p.first] = p.second
    end
end

prices_matrix = hcat([dfs[t].close for t in TICKERS]...)
spy_prices = dfs["SPY"].close
S0 = dfs[TARGET].close[end]

println("  → $(size(prices_matrix, 1)) trading days, $(length(TICKERS)) tickers")
println("  → Target: $TARGET, Spot: \$$(round(S0, digits=2))")

# --- Step 2: Fit HybridSIM portfolio with SPY as market ---
println("\nFitting HybridSingleIndexModel (SPY as market factor)...")
portfolio = fit(PortfolioModel, TICKERS, prices_matrix;
                dependence=HybridSingleIndexModel, market="SPY",
                N=50, rf=0.05)

# Report HybridSIM diagnostics
hsim = portfolio.dependence
non_market = hsim.tickers
println("  → Market model: $(hsim.market_model.partition.N) states, " *
        "N_tail=$(hsim.market_model.jump.N_tail)")
for (i, t) in enumerate(non_market)
    println("  → $t: α=$(round(hsim.α[i], digits=3)), " *
            "β=$(round(hsim.β[i], digits=3)), " *
            "r²=$(round(hsim.r²[i], digits=3))")
end

# --- Step 3: Auto-calibrate θ_states from SPY ---
println("\nAuto-calibrating θ_states from SPY empirical variance per state...")
θ_states = auto_calibrate_theta_states(hsim.market_model, spy_prices)

N_states = hsim.market_model.partition.N
N_tail = hsim.market_model.jump.N_tail
println("  → θ range: [$(round(minimum(θ_states), digits=4)), " *
        "$(round(maximum(θ_states), digits=4))]")
println("  → θ median: $(round(median(θ_states), digits=4)) " *
        "(≈ $(round(sqrt(median(θ_states))*100, digits=1))% IV)")
println("  → Tail states (bottom $N_tail): mean θ = " *
        "$(round(mean(θ_states[1:N_tail]), digits=4)) " *
        "(≈ $(round(sqrt(mean(θ_states[1:N_tail]))*100, digits=1))% IV)")
println("  → Tail states (top $N_tail): mean θ = " *
        "$(round(mean(θ_states[end-N_tail+1:end]), digits=4)) " *
        "(≈ $(round(sqrt(mean(θ_states[end-N_tail+1:end]))*100, digits=1))% IV)")

# --- Step 4: Set up Heston parameters ---
heston_params = HestonParameters(4.0, 0.35)  # κ, σ_v

# Use calibrated β from paper (NVDA single-DTE calibration)
θ_func = ThetaHybrid(
    θ_states,
    [0.20, -2.07, 0.34, 1.43],  # β₁ (DTE decay), β₂ (skew), β₃ (interaction), β₄ (curvature)
    0.5                          # γ (mood sensitivity)
)

# --- Step 5: Define contracts on target ticker ---
contracts = [
    OptionContract(round(S0 * 0.90), 80, :put, :american),    # 10% OTM put
    OptionContract(round(S0 * 0.95), 80, :put, :american),    # 5% OTM put
    OptionContract(round(S0),        80, :call, :american),    # ATM call
    OptionContract(round(S0 * 1.05), 80, :call, :american),   # 5% OTM call
    OptionContract(round(S0 * 1.10), 80, :call, :american),   # 10% OTM call
]

# --- Step 6: Run multi-asset scenario ---
println("\nRunning HybridSIM scenario (500 paths × 80 steps)...")
result = run_scenario(
    portfolio, heston_params, θ_func, contracts,
    TARGET, S0;
    n_sim_steps=80, n_paths=500,
    eval_step=5, r_f=0.05, seed=456
)

# --- Step 7: Analyze ---
println("\n" * "="^60)
println("HYBRID-SIM SCENARIO RESULTS — $TARGET (market: SPY)")
println("="^60)

stats = summarize(result)
println("\nPaths: $(stats[:n_paths]), Steps: $(stats[:n_steps])")
println("Mean IV: $(round(stats[:iv_mean]*100, digits=2))%")
println("IV Std: $(round(stats[:iv_std]*100, digits=2))%")
println("Mean mood (binary, SPY): $(round(stats[:mood_mean], digits=3))")
println("Max mood: $(round(stats[:mood_max], digits=3))")

println("\n--- Option Prices ---")
for cs in stats[:contracts]
    c = cs[:contract]
    type_str = c.option_type == :call ? "CALL" : "PUT "
    println("  $(type_str) K=$(c.K) DTE=$(c.DTE): " *
            "mean=\$$(round(cs[:mean_price], digits=2)) " *
            "std=\$$(round(cs[:std_price], digits=2)) " *
            "[$(round(cs[:q05], digits=2)), $(round(cs[:q95], digits=2))]")
end

# --- Step 8: Compare calm vs stressed scenarios ---
println("\n--- Tail Event Analysis (SPY binary mood) ---")
# Use mood at eval_step to classify paths
eval_mood = result.mood_paths[:, min(5, size(result.mood_paths, 2))]
stressed = eval_mood .> 0.0
calm = .!stressed

if sum(stressed) > 0 && sum(calm) > 0
    println("  Calm paths: $(sum(calm)), Stressed paths: $(sum(stressed))")
    for (i, c) in enumerate(contracts)
        type_str = c.option_type == :call ? "CALL" : "PUT "
        mean_calm = mean(result.option_prices[calm, i])
        mean_stress = mean(result.option_prices[stressed, i])
        ratio = round(mean_stress / max(mean_calm, 0.01), digits=2)
        println("  $(type_str) K=$(c.K) DTE=$(c.DTE): " *
                "calm=\$$(round(mean_calm, digits=2)) " *
                "stress=\$$(round(mean_stress, digits=2)) " *
                "ratio=$(ratio)x")
    end
else
    println("  (insufficient tail events for comparison)")
end
