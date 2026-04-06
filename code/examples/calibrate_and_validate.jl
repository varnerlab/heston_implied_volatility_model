"""
Calibration & Validation Against Real Option Chains

Calibrates the θ-function on NVDA options, then validates on AMD, MU, and INTC.
Produces IV surface plots comparing model vs market.

Data source: barchart.com option chains for semiconductor stocks
(stored in data/options/)
"""

using JumpHMM
using HestonIV
using CSV
using TOML
using DataFrames
using Statistics
using Optim
using Plots
using Random

# ============================================================================
# Step 1: Load option chain data
# ============================================================================

const DATA_DIR = joinpath(@__DIR__, "..", "data", "options")

"""Load a single ticker's option chain + metadata."""
function load_chain(ticker::String)
    csv_path = joinpath(DATA_DIR, "$(ticker).csv")
    toml_path = joinpath(DATA_DIR, "$(ticker).toml")

    df = CSV.read(csv_path, DataFrame)
    meta = TOML.parsefile(toml_path)["metadata"]

    S = parse(Float64, meta["underlying_share_price_mid"])
    DTE = parse(Int, meta["DTE"])
    atm_iv = parse(Float64, meta["atm_IV"])

    # Compute K/S moneyness
    df[!, :KoverS] = df.Strike ./ S

    # Filter to tradeable range:
    #   - IV > 0 (drop missing/zero IV)
    #   - Moneyness 0.70 to 1.30 (drop deep ITM/OTM with unreliable IV)
    #   - IV < 2.0 (drop obvious outliers)
    valid = df[(df.IV .> 0.01) .& (df.IV .< 2.0) .&
               (df.KoverS .> 0.70) .& (df.KoverS .< 1.30), :]

    return valid, S, DTE, atm_iv
end

println("Loading option chains...")
tickers = ["nvda", "amd", "mu", "intc"]
chains = Dict{String,Any}()
for t in tickers
    df, S, DTE, atm_iv = load_chain(t)
    chains[t] = (data=df, S=S, DTE=DTE, atm_iv=atm_iv)
    println("  $(uppercase(t)): $(nrow(df)) contracts, S=\$$(round(S,digits=2)), DTE=$DTE, ATM IV=$(round(atm_iv*100,digits=1))%")
end

# ============================================================================
# Step 2: Load pretrained JumpHMM (or build a fallback)
# ============================================================================

const PRETRAINED_PATH = joinpath(@__DIR__, "..", "data", "pretrained-portfolio-surrogate.jld2")

if isfile(PRETRAINED_PATH)
    using JLD2
    println("\nLoading pretrained JumpHMM portfolio...")
    portfolio_data = JLD2.load(PRETRAINED_PATH)
    marginals = portfolio_data["marginals"]
else
    println("\nPretrained model not found — using placeholder HMM states.")
    println("  (For real calibration, generate the pretrained model first)")
    marginals = nothing
end

# ============================================================================
# Step 3: Calibrate on NVDA
# ============================================================================

println("\n" * "="^60)
println("  CALIBRATING ON NVDA")
println("="^60)

nvda = chains["nvda"]
nvda_df = nvda.data
S_nvda = nvda.S
DTE_nvda = nvda.DTE

# For calibration, we need HMM state per observation.
# Since all observations are from the same date, they share one state.
# We'll calibrate the θ-function to match the IV smile directly.
#
# Key insight: at t=0, v₀ = θ(s₀, DTE, K/S, M₀)
# so the model IV = √θ(s₀, DTE, K/S, M₀)
# We just need to find (θ_base, β₁, β₂, β₃, γ) that best fit the observed IV smile.

# For a single-date calibration, we fix:
#   s₀ = some reference state (mid-range)
#   M₀ = 0 (no stress signal from a single snapshot)
# and optimize θ_base and β to match the IV curve

N_STATES = 50  # same as pretrained model

# β₁ is fixed from the empirical term structure of equity IV (contango):
#   σ_ATM(T) ∝ T^(β₁/2)  →  β₁ = +0.20  →  σ ∝ DTE^(+0.10)
# Longer DTE = higher IV. As a contract ages (DTE shrinks), IV decays.
# This is the normal (non-stressed) term structure. We fix β₁ because
# single-DTE snapshots cannot identify it; multi-expiration data would
# allow calibration (future work).
const β₁_FIXED = 0.20

function calibrate_single_date(df, S, DTE; β₁=β₁_FIXED)
    n_obs = nrow(df)
    strikes = df.Strike
    ivs = df.IV
    moneyness = strikes ./ S
    log_DTE = log(max(DTE, 1.0))

    # Parameter vector: [log(θ_base), β₂, β₃, β₄]
    # β₁ is fixed; we optimize θ_base (level), β₂ (skew), β₃ (interaction), β₄ (curvature)
    # The IV model is: σ = √(θ_base · ψ(DTE, K/S))
    #   where ψ = exp(β₁·ln(DTE) + β₂·ln(K/S) + β₃·ln(DTE)·ln(K/S) + β₄·(ln(K/S))²)

    x0 = [log(mean(ivs)^2), 0.0, 0.0, 0.0]

    function objective(x)
        θ_base = exp(x[1])
        β₂, β₃, β₄ = x[2], x[3], x[4]

        err = 0.0
        for i in 1:n_obs
            log_m = log(moneyness[i])
            ψ_val = exp(β₁ * log_DTE + β₂ * log_m + β₃ * log_DTE * log_m + β₄ * log_m^2)
            σ_model = sqrt(max(θ_base * ψ_val, 1e-10))
            err += (σ_model - ivs[i])^2
        end
        return err / n_obs
    end

    result = optimize(objective, x0, NelderMead(),
                      Optim.Options(iterations=10000, g_tol=1e-12))

    x_opt = Optim.minimizer(result)
    θ_base = exp(x_opt[1])
    β = [β₁, x_opt[2], x_opt[3], x_opt[4]]
    rmse = sqrt(Optim.minimum(result))

    # Compute model IVs
    model_ivs = Float64[]
    for i in 1:n_obs
        log_m = log(moneyness[i])
        ψ_val = exp(β[1] * log_DTE + β[2] * log_m + β[3] * log_DTE * log_m + β[4] * log_m^2)
        push!(model_ivs, sqrt(max(θ_base * ψ_val, 1e-10)))
    end

    return θ_base, β, rmse, model_ivs
end

θ_base_nvda, β_nvda, rmse_nvda, model_ivs_nvda = calibrate_single_date(
    nvda_df, S_nvda, DTE_nvda
)

println("\nCalibrated parameters (β₁ = $β₁_FIXED fixed):")
println("  θ_base = $(round(θ_base_nvda, digits=6)) (√θ = $(round(sqrt(θ_base_nvda)*100, digits=1))% IV)")
println("  β₁ (DTE, fixed)   = $(β_nvda[1])")
println("  β₂ (skew)         = $(round(β_nvda[2], digits=4))")
println("  β₃ (interaction)  = $(round(β_nvda[3], digits=4))")
println("  β₄ (curvature)    = $(round(β_nvda[4], digits=4))")
println("  RMSE = $(round(rmse_nvda*100, digits=2))% IV")

# Show term structure effect:
println("\n  Term structure (β₁ effect on ATM IV):")
for dte_ex in [7, 14, 30, 60, 90, 180]
    iv_at_dte = sqrt(θ_base_nvda * exp(β_nvda[1] * log(dte_ex))) * 100
    println("    DTE=$dte_ex → $(round(iv_at_dte, digits=1))%")
end

# ============================================================================
# Step 4: Validate on other tickers
# ============================================================================

println("\n" * "="^60)
println("  CROSS-TICKER VALIDATION")
println("="^60)

# Apply the same β (smile shape) but re-estimate θ_base per ticker
# This tests if the smile SHAPE generalizes
results = Dict{String,Any}()

for t in tickers
    ch = chains[t]
    df, S, DTE = ch.data, ch.S, ch.DTE

    # Re-optimize just θ_base with fixed β from NVDA
    n_obs = nrow(df)
    moneyness = df.Strike ./ S
    ivs = df.IV

    log_DTE_t = log(max(DTE, 1.0))

    function obj_base(x)
        θ_b = exp(x[1])
        err = 0.0
        for i in 1:n_obs
            log_m = log(moneyness[i])
            ψ_val = exp(β_nvda[1] * log_DTE_t + β_nvda[2] * log_m + β_nvda[3] * log_DTE_t * log_m + β_nvda[4] * log_m^2)
            σ_model = sqrt(max(θ_b * ψ_val, 1e-10))
            err += (σ_model - ivs[i])^2
        end
        return err / n_obs
    end

    res = optimize(obj_base, [log(mean(ivs)^2)], NelderMead(),
                   Optim.Options(iterations=5000))

    θ_base_t = exp(Optim.minimizer(res)[1])
    rmse_t = sqrt(Optim.minimum(res))

    # Model IVs
    model_ivs = Float64[]
    for i in 1:n_obs
        log_m = log(moneyness[i])
        ψ_val = exp(β_nvda[1] * log_DTE_t + β_nvda[2] * log_m + β_nvda[3] * log_DTE_t * log_m + β_nvda[4] * log_m^2)
        push!(model_ivs, sqrt(max(θ_base_t * ψ_val, 1e-10)))
    end

    results[t] = (θ_base=θ_base_t, rmse=rmse_t, model_ivs=model_ivs)
    println("  $(uppercase(t)): θ_base=$(round(θ_base_t, digits=4)) ($(round(sqrt(θ_base_t)*100,digits=1))% IV), RMSE=$(round(rmse_t*100, digits=2))%")
end

# ============================================================================
# Step 5: Generate plots
# ============================================================================

println("\nGenerating plots...")
const PLOT_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(PLOT_DIR)

# --- Plot 1: NVDA calibration fit ---
nvda_calls = nvda_df[nvda_df.Type .== "Call", :]
nvda_puts = nvda_df[nvda_df.Type .== "Put", :]
model_calls = model_ivs_nvda[nvda_df.Type .== "Call"]
model_puts = model_ivs_nvda[nvda_df.Type .== "Put"]

p1 = plot(title="NVDA IV Smile — Model vs Market (DTE=$DTE_nvda)",
          xlabel="Strike", ylabel="Implied Volatility",
          legend=:topright, size=(800, 500), dpi=150)
scatter!(p1, nvda_calls.Strike, nvda_calls.IV .* 100,
         label="Market Calls", marker=:circle, ms=3, color=:blue, alpha=0.6)
scatter!(p1, nvda_puts.Strike, nvda_puts.IV .* 100,
         label="Market Puts", marker=:diamond, ms=3, color=:red, alpha=0.6)
scatter!(p1, nvda_calls.Strike, model_calls .* 100,
         label="Model Calls", marker=:xcross, ms=4, color=:dodgerblue)
scatter!(p1, nvda_puts.Strike, model_puts .* 100,
         label="Model Puts", marker=:xcross, ms=4, color=:salmon)
vline!(p1, [S_nvda], label="Spot (\$$(round(S_nvda,digits=0)))", ls=:dash, color=:gray)
savefig(p1, joinpath(PLOT_DIR, "nvda_calibration.png"))
println("  → saved nvda_calibration.png")

# --- Plot 2: All tickers — model vs market IV by moneyness ---
p2 = plot(title="IV Smile Fit — All Tickers (β shared from NVDA)",
          xlabel="Moneyness (K/S)", ylabel="Implied Volatility (%)",
          legend=:topright, size=(800, 500), dpi=150)

colors_market = [:blue, :red, :green, :purple]
colors_model = [:dodgerblue, :salmon, :limegreen, :mediumpurple]
for (i, t) in enumerate(tickers)
    ch = chains[t]
    df = ch.data
    m = df.Strike ./ ch.S

    scatter!(p2, m, df.IV .* 100,
             label="$(uppercase(t)) market", marker=:circle, ms=2,
             color=colors_market[i], alpha=0.5)

    model_iv = results[t].model_ivs
    plot!(p2, sort(m), model_iv[sortperm(m)] .* 100,
          label="$(uppercase(t)) model", lw=2, color=colors_model[i])
end
vline!(p2, [1.0], label="ATM", ls=:dash, color=:gray)
savefig(p2, joinpath(PLOT_DIR, "all_tickers_smile.png"))
println("  → saved all_tickers_smile.png")

# --- Plot 3: Residuals ---
p3 = plot(title="IV Residuals (Model - Market)",
          xlabel="Moneyness (K/S)", ylabel="IV Residual (%)",
          legend=:topright, size=(800, 400), dpi=150)
hline!(p3, [0.0], color=:gray, ls=:dash, label=nothing)

for (i, t) in enumerate(tickers)
    ch = chains[t]
    df = ch.data
    m = df.Strike ./ ch.S
    residuals = (results[t].model_ivs .- df.IV) .* 100
    scatter!(p3, m, residuals,
             label="$(uppercase(t))", marker=:circle, ms=3,
             color=colors_market[i], alpha=0.6)
end
savefig(p3, joinpath(PLOT_DIR, "iv_residuals.png"))
println("  → saved iv_residuals.png")

# --- Plot 4: Scenario — forward IV dynamics for NVDA ---
println("\nRunning forward scenario for NVDA...")

# Build a ThetaHybrid from calibrated parameters
θ_states_cal = fill(θ_base_nvda, N_STATES)
# Elevate tail states
N_TAIL = 5
for s in 1:N_TAIL
    θ_states_cal[s] = θ_base_nvda * 2.0   # 2x vol in bearish tail
end
for s in (N_STATES - N_TAIL + 1):N_STATES
    θ_states_cal[s] = θ_base_nvda * 1.5   # 1.5x vol in bullish tail
end

θ_func_cal = ThetaHybrid(θ_states_cal, β_nvda, 0.5)
heston_cal = HestonParameters(5.0, 0.3)

# If we have the pretrained model, use it; otherwise build a quick one
if marginals !== nothing && haskey(marginals, "NVDA")
    nvda_model = marginals["NVDA"]
else
    # Quick synthetic model for demo
    Random.seed!(42)
    synth_prices = Float64[S_nvda]
    for _ in 1:500
        push!(synth_prices, synth_prices[end] * exp(0.0003 + 0.02 * randn()))
    end
    nvda_model = fit(JumpHiddenMarkovModel, synth_prices; N=N_STATES, rf=0.05)
    nvda_model = tune(nvda_model, synth_prices; n_paths=100)
end

contracts_nvda = [
    OptionContract(round(S_nvda * 0.90), DTE_nvda, :put, :american),
    OptionContract(round(S_nvda * 0.95), DTE_nvda, :put, :american),
    OptionContract(round(S_nvda),        DTE_nvda, :call, :american),
    OptionContract(round(S_nvda * 1.05), DTE_nvda, :call, :american),
    OptionContract(round(S_nvda * 1.10), DTE_nvda, :call, :american),
]

result = run_single_asset_scenario(
    nvda_model, heston_cal, θ_func_cal, contracts_nvda, S_nvda;
    n_sim_steps=DTE_nvda, n_paths=500, eval_step=5, r_f=0.05, seed=789
)

# Plot IV path distribution
p4 = plot(title="NVDA Forward IV Paths ($(500) scenarios, DTE=$DTE_nvda)",
          xlabel="Trading Days", ylabel="Implied Volatility (%)",
          legend=:topright, size=(800, 500), dpi=150)

# Plot a sample of paths
for i in 1:min(50, size(result.iv_paths, 1))
    plot!(p4, 1:DTE_nvda, result.iv_paths[i, :] .* 100,
          color=:blue, alpha=0.08, label=nothing)
end

# Mean and percentiles
iv_mean = mean(result.iv_paths, dims=1)[1, :] .* 100
iv_q05 = [quantile(result.iv_paths[:, t], 0.05) for t in 1:DTE_nvda] .* 100
iv_q95 = [quantile(result.iv_paths[:, t], 0.95) for t in 1:DTE_nvda] .* 100

plot!(p4, 1:DTE_nvda, iv_mean, lw=3, color=:black, label="Mean IV")
plot!(p4, 1:DTE_nvda, iv_q05, lw=2, ls=:dash, color=:red, label="5th pctl")
plot!(p4, 1:DTE_nvda, iv_q95, lw=2, ls=:dash, color=:green, label="95th pctl")
hline!(p4, [nvda.atm_iv * 100], ls=:dot, color=:orange, label="Market ATM IV")

savefig(p4, joinpath(PLOT_DIR, "nvda_iv_paths.png"))
println("  → saved nvda_iv_paths.png")

# --- Summary ---
println("\n" * "="^60)
println("  CALIBRATION SUMMARY")
println("="^60)
println("\nShared smile parameters (calibrated on NVDA):")
println("  β₁ (DTE decay, fixed) = $(round(β_nvda[1], digits=4))")
println("  β₂ (skew)             = $(round(β_nvda[2], digits=4))")
println("  β₃ (DTE×skew interact) = $(round(β_nvda[3], digits=4))")
println("  β₄ (smile curvature)  = $(round(β_nvda[4], digits=4))")
println("\nPer-ticker θ_base (equivalent ATM IV):")
for t in tickers
    θ_b = results[t].θ_base
    println("  $(uppercase(t)): θ=$(round(θ_b, digits=4)) → IV=$(round(sqrt(θ_b)*100, digits=1))%")
end
println("\nCross-ticker RMSE:")
for t in tickers
    println("  $(uppercase(t)): $(round(results[t].rmse*100, digits=2))% IV")
end

stats = summarize(result)
println("\nForward scenario (NVDA, eval_step=5):")
for cs in stats[:contracts]
    c = cs[:contract]
    type_str = c.option_type == :call ? "CALL" : "PUT "
    println("  $(type_str) K=$(c.K) DTE=$(c.DTE): \$$(round(cs[:mean_price],digits=2)) ± \$$(round(cs[:std_price],digits=2))")
end
