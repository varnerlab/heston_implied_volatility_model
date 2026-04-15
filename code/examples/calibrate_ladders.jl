"""
Joint Calibration of IV Model Parameters from Multi-Ticker Ladder Data

Loads options ladder data for 23 tickers (captured 2026-04-14), filters to
tradeable observations, and jointly estimates:
  - Shared smile-shape parameters (beta1-beta5)
  - Per-ticker baseline volatility (theta_base)

Model:
    sigma_model(ticker, K, DTE) = sqrt(theta_base[ticker] * psi(beta, DTE, K/S))

where psi = exp(beta1*ln(DTE) + beta2*ln(K/S) + beta3*ln(DTE)*ln(K/S)
                + beta4*(ln(K/S))^2 + beta5*(ln(DTE))^2)

This tests whether the psi surface (smile shape) is universal across sectors:
tech, financials, energy, healthcare, and ETFs.
"""

using CSV
using DataFrames
using Statistics
using Optim
using Plots
using Printf

# ============================================================================
# Step 1: Load and filter ladder data
# ============================================================================

const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladders")
const PLOT_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(PLOT_DIR)

# Sector classification for analysis
const SECTORS = Dict(
    "AAPL" => "Tech", "AMD" => "Tech", "INTC" => "Tech", "MSFT" => "Tech",
    "MU" => "Tech", "NVDA" => "Tech",
    "BAC" => "Financials", "GS" => "Financials", "JPM" => "Financials",
    "WFC" => "Financials",
    "CVX" => "Energy", "OXY" => "Energy", "XOM" => "Energy",
    "ABBV" => "Healthcare", "JNJ" => "Healthcare", "LLY" => "Healthcare",
    "MRNA" => "Healthcare",
    "TGT" => "Retail", "UPS" => "Retail", "WMT" => "Retail",
    "IWM" => "ETF", "QQQ" => "ETF", "SPY" => "ETF",
)

"""Load a single ladder CSV and return a filtered DataFrame."""
function load_ladder(filepath::String)
    df = CSV.read(filepath, DataFrame)

    # Extract ticker from the underlying column
    ticker = string(df.underlying[1])

    # Spot price = most recent underlying close
    S = df.und_close[1]

    # Compute moneyness
    df[!, :ticker] .= ticker
    df[!, :S] .= S
    df[!, :moneyness] = df.strike ./ S

    # Filter to usable observations
    valid = df[
        .!ismissing.(df.implied_vol) .&
        .!isnan.(coalesce.(df.implied_vol, NaN)) .&
        (coalesce.(df.implied_vol, 0.0) .> 0.01) .&
        (coalesce.(df.implied_vol, 999.0) .< 2.0) .&
        (df.bid .> 0) .&
        (df.moneyness .>= 0.80) .&
        (df.moneyness .<= 1.20) .&
        (df.actual_dte .> 0),
    :]

    return valid
end

"""Load all ladder CSVs from a directory."""
function load_all_ladders(dir::String)
    files = filter(f -> endswith(f, ".csv"), readdir(dir; join=true))
    frames = DataFrame[]
    for f in files
        df = load_ladder(f)
        if nrow(df) > 0
            push!(frames, df)
        end
    end
    return vcat(frames...)
end

println("Loading ladder data from $(LADDER_DIR)...")
all_data = load_all_ladders(LADDER_DIR)
tickers = sort(unique(all_data.ticker))
n_tickers = length(tickers)
ticker_idx = Dict(t => i for (i, t) in enumerate(tickers))

println("  Loaded $(nrow(all_data)) observations across $n_tickers tickers")
println("  Tickers: $(join(tickers, ", "))")
println("\n  Per-ticker summary:")
for t in tickers
    slice = all_data[all_data.ticker .== t, :]
    dtes = sort(unique(slice.actual_dte))
    S = slice.S[1]
    sector = get(SECTORS, t, "Other")
    n_calls = sum(slice.type .== "call")
    n_puts = sum(slice.type .== "put")
    @printf("    %-5s [%-11s]: %4d obs (%d C, %d P), S=\$%7.2f, DTEs=%s\n",
            t, sector, nrow(slice), n_calls, n_puts, S, join(dtes, ","))
end

# ============================================================================
# Step 2: Joint optimization — shared beta + per-ticker theta_base
# ============================================================================

println("\n" * "="^70)
println("  JOINT CALIBRATION: shared beta + per-ticker theta_base")
println("="^70)

# Precompute arrays for speed
const N_OBS = nrow(all_data)
const obs_iv = Float64.(all_data.implied_vol)
const obs_moneyness = Float64.(all_data.moneyness)
const obs_dte = Float64.(all_data.actual_dte)
const obs_ticker_idx = [ticker_idx[t] for t in all_data.ticker]
const obs_log_dte = log.(max.(obs_dte, 1.0))
const obs_log_m = log.(obs_moneyness)

"""Evaluate psi(beta, log_dte, log_m)."""
function eval_psi(beta::AbstractVector, log_dte::Float64, log_m::Float64)::Float64
    return exp(beta[1] * log_dte + beta[2] * log_m + beta[3] * log_dte * log_m +
               beta[4] * log_m^2 + beta[5] * log_dte^2)
end

"""
    joint_objective(x) -> Float64

Parameter vector layout:
  x[1:5]              = beta1 through beta5 (shared smile shape)
  x[6:5+n_tickers]    = log(theta_base) for each ticker (ensures theta > 0)
"""
function joint_objective(x)
    beta = @view x[1:5]
    log_theta = @view x[6:end]

    total_err = 0.0
    for i in 1:N_OBS
        t_idx = obs_ticker_idx[i]
        theta_base = exp(log_theta[t_idx])
        psi_val = eval_psi(beta, obs_log_dte[i], obs_log_m[i])
        sigma_model = sqrt(max(theta_base * psi_val, 1e-10))
        total_err += (sigma_model - obs_iv[i])^2
    end
    return total_err / N_OBS
end

# Initialize: beta from prior work, theta_base from per-ticker mean IV^2
beta_init = [0.3, -1.0, 0.1, 2.0, -0.15]  # literature/prior estimates
theta_init = Vector{Float64}(undef, n_tickers)
for (i, t) in enumerate(tickers)
    slice_iv = obs_iv[obs_ticker_idx .== i]
    theta_init[i] = mean(slice_iv)^2
end

x0 = vcat(beta_init, log.(theta_init))
n_params = length(x0)
println("\n  Parameters: $n_params ($n_tickers theta_base + 5 beta)")
println("  Observations: $N_OBS")
println("  Observations per parameter: $(round(N_OBS / n_params, digits=0))")

println("\n  Pass 1: Nelder-Mead (100k iterations)...")
result1 = optimize(joint_objective, x0, NelderMead(),
                   Optim.Options(iterations=100_000, show_trace=false))
println("    MSE = $(round(Optim.minimum(result1), digits=8))")

println("  Pass 2: Nelder-Mead polish (100k iterations)...")
result2 = optimize(joint_objective, Optim.minimizer(result1), NelderMead(),
                   Optim.Options(iterations=100_000, show_trace=false))
println("    MSE = $(round(Optim.minimum(result2), digits=8))")

x_opt = Optim.minimizer(result2)
beta_opt = x_opt[1:5]
theta_opt = exp.(x_opt[6:end])

# Compute model IVs for all observations
model_ivs = Vector{Float64}(undef, N_OBS)
for i in 1:N_OBS
    t_idx = obs_ticker_idx[i]
    psi_val = eval_psi(beta_opt, obs_log_dte[i], obs_log_m[i])
    model_ivs[i] = sqrt(max(theta_opt[t_idx] * psi_val, 1e-10))
end
overall_rmse = sqrt(mean((model_ivs .- obs_iv).^2))

# ============================================================================
# Step 3: Results — parameters
# ============================================================================

println("\n" * "="^70)
println("  CALIBRATED PARAMETERS")
println("="^70)

println("\n  Shared smile-shape (beta):")
println("    beta1 (DTE decay)         = $(round(beta_opt[1], digits=4))")
println("    beta2 (skew)              = $(round(beta_opt[2], digits=4))")
println("    beta3 (DTE x skew)        = $(round(beta_opt[3], digits=4))")
println("    beta4 (smile curvature)   = $(round(beta_opt[4], digits=4))")
println("    beta5 (DTE curvature)     = $(round(beta_opt[5], digits=4))")
println("    Overall RMSE = $(round(overall_rmse * 100, digits=2))% IV")

println("\n  Interpretation:")
println("    beta1 = $(round(beta_opt[1], digits=3)): ",
    beta_opt[1] > 0 ? "IV increases with DTE (contango)" :
                       "IV decreases with DTE (backwardation)")
println("    beta2 = $(round(beta_opt[2], digits=3)): ",
    beta_opt[2] < 0 ? "negative skew (OTM puts > OTM calls)" :
                       "positive skew (unusual)")
println("    beta3 = $(round(beta_opt[3], digits=3)): ",
    beta_opt[3] > 0 ? "skew flattens at longer maturities" :
                       "skew steepens at longer maturities")
println("    beta4 = $(round(beta_opt[4], digits=3)): ",
    beta_opt[4] > 0 ? "U-shaped smile (both wings elevated)" :
                       "inverted smile (unusual)")
println("    beta5 = $(round(beta_opt[5], digits=3)): ",
    beta_opt[5] < 0 ? "U-shaped ATM term structure" :
                       "monotonic ATM term structure")

println("\n  Per-ticker theta_base (sorted by IV level):")
ticker_order = sortperm(theta_opt; rev=true)
for idx in ticker_order
    t = tickers[idx]
    theta = theta_opt[idx]
    iv_level = sqrt(theta) * 100
    sector = get(SECTORS, t, "Other")
    @printf("    %-5s [%-11s]: theta=%.4f  (%.1f%% IV)\n", t, sector, theta, iv_level)
end

# ============================================================================
# Step 4: Per-ticker and per-DTE validation
# ============================================================================

println("\n" * "="^70)
println("  PER-TICKER FIT QUALITY")
println("="^70)

all_data[!, :model_iv] = model_ivs
all_data[!, :residual] = model_ivs .- obs_iv

println("\n  Ticker       N     RMSE(%)  Bias(%)  Sector")
println("  " * "-"^55)
for t in tickers
    mask = all_data.ticker .== t
    res = all_data.residual[mask]
    rmse_t = sqrt(mean(res.^2)) * 100
    bias_t = mean(res) * 100
    sector = get(SECTORS, t, "Other")
    @printf("  %-5s    %5d    %5.2f    %+5.2f    %s\n",
            t, sum(mask), rmse_t, bias_t, sector)
end

println("\n" * "="^70)
println("  PER-DTE FIT QUALITY (all tickers pooled)")
println("="^70)
println("\n  DTE     N     RMSE(%)  Bias(%)")
println("  " * "-"^35)
for dte in sort(unique(all_data.actual_dte))
    mask = all_data.actual_dte .== dte
    res = all_data.residual[mask]
    @printf("  %3d   %5d    %5.2f    %+5.2f\n",
            dte, sum(mask), sqrt(mean(res.^2)) * 100, mean(res) * 100)
end

# ============================================================================
# Step 5: Sector-level analysis
# ============================================================================

println("\n" * "="^70)
println("  SECTOR-LEVEL ANALYSIS")
println("="^70)

all_data[!, :sector] = [get(SECTORS, t, "Other") for t in all_data.ticker]

println("\n  Sector         N     RMSE(%)  Bias(%)  Mean IV(%)")
println("  " * "-"^55)
for sector in sort(unique(all_data.sector))
    mask = all_data.sector .== sector
    res = all_data.residual[mask]
    mean_iv = mean(obs_iv[mask]) * 100
    @printf("  %-12s %5d    %5.2f    %+5.2f     %5.1f\n",
            sector, sum(mask), sqrt(mean(res.^2)) * 100, mean(res) * 100, mean_iv)
end

# ============================================================================
# Step 6: Comparison with prior NVDA-only estimates
# ============================================================================

println("\n" * "="^70)
println("  COMPARISON WITH PRIOR ESTIMATES")
println("="^70)

# Old estimates from fit_term_structure.jl (NVDA-only, April 6 data)
println("\n  Parameter       Old (NVDA-only)    New (23-ticker joint)")
println("  " * "-"^55)
old_beta = [0.3, -1.0, 0.1, 2.0, -0.15]  # approximate prior values
labels = ["beta1 (DTE)", "beta2 (skew)", "beta3 (interact)", "beta4 (curve)", "beta5 (DTE^2)"]
for (i, label) in enumerate(labels)
    @printf("  %-18s  %+8.4f           %+8.4f\n", label, old_beta[i], beta_opt[i])
end

# ============================================================================
# Step 7: Diagnostic plots
# ============================================================================

println("\nGenerating diagnostic plots...")

# --- Plot 1: Representative IV smiles (6-panel: 2 per row, 3 rows) ---
representative = ["SPY", "NVDA", "JPM", "XOM", "LLY", "WMT"]
# Use whatever is available
representative = filter(t -> t in tickers, representative)

p_panels = []
for t in representative
    slice = all_data[all_data.ticker .== t, :]
    # Pick a mid-range DTE for the smile plot
    avail_dtes = sort(unique(slice.actual_dte))
    target_dte = avail_dtes[max(1, length(avail_dtes) ÷ 2)]
    dte_slice = slice[slice.actual_dte .== target_dte, :]

    S = dte_slice.S[1]
    sector = get(SECTORS, t, "Other")

    p = plot(title="$t [$sector] DTE=$target_dte",
             xlabel="Moneyness (K/S)", ylabel="IV (%)",
             legend=:topright, titlefontsize=9)

    calls = dte_slice[dte_slice.type .== "call", :]
    puts = dte_slice[dte_slice.type .== "put", :]

    scatter!(p, calls.moneyness, Float64.(calls.implied_vol) .* 100,
             label="Calls", marker=:circle, ms=3, color=:blue, alpha=0.6)
    scatter!(p, puts.moneyness, Float64.(puts.implied_vol) .* 100,
             label="Puts", marker=:diamond, ms=3, color=:red, alpha=0.6)

    # Model curve
    m_range = range(0.85, 1.15, length=80)
    log_dte_val = log(max(target_dte, 1.0))
    t_idx = ticker_idx[t]
    iv_curve = [sqrt(max(theta_opt[t_idx] * eval_psi(beta_opt, log_dte_val, log(m)), 1e-10)) * 100
                for m in m_range]
    plot!(p, m_range, iv_curve, label="Model", lw=2, color=:black)
    vline!(p, [1.0], label=nothing, ls=:dash, color=:gray, alpha=0.5)

    push!(p_panels, p)
end

p1 = plot(p_panels..., layout=(2, 3), size=(1200, 700), dpi=150,
          plot_title="IV Smile Fit — Representative Tickers (shared beta)")
savefig(p1, joinpath(PLOT_DIR, "ladder_smile_panels.png"))
savefig(p1, joinpath(PLOT_DIR, "ladder_smile_panels.pdf"))
println("  -> saved ladder_smile_panels.png/pdf")

# --- Plot 2: ATM term structure (all tickers overlaid) ---
p2 = plot(title="ATM IV Term Structure — All Tickers",
          xlabel="Days to Expiration", ylabel="ATM IV (%)",
          legend=:outerright, size=(900, 500), dpi=150)

sector_colors = Dict("Tech" => :blue, "Financials" => :red, "Energy" => :green,
                     "Healthcare" => :purple, "Retail" => :orange, "ETF" => :black)

# Market ATM points per ticker
for t in tickers
    slice = all_data[(all_data.ticker .== t) .& (abs.(all_data.moneyness .- 1.0) .< 0.05), :]
    if nrow(slice) == 0
        continue
    end
    atm_by_dte = combine(groupby(slice, :actual_dte),
                         :implied_vol => (x -> mean(Float64.(x))) => :mean_iv)
    sector = get(SECTORS, t, "Other")
    c = get(sector_colors, sector, :gray)
    scatter!(p2, atm_by_dte.actual_dte, atm_by_dte.mean_iv .* 100,
             label=nothing, marker=:circle, ms=3, color=c, alpha=0.4)
end

# Model ATM curves for a few representative tickers
dte_range = 1:80
for t in ["SPY", "NVDA", "JPM", "XOM"]
    if !(t in tickers)
        continue
    end
    t_idx = ticker_idx[t]
    iv_atm = [sqrt(max(theta_opt[t_idx] * eval_psi(beta_opt, log(d), 0.0), 1e-10)) * 100
              for d in dte_range]
    sector = get(SECTORS, t, "Other")
    c = get(sector_colors, sector, :gray)
    plot!(p2, dte_range, iv_atm, label=t, lw=2, color=c)
end

# Add sector legend entries
for (sector, c) in sort(collect(sector_colors))
    scatter!(p2, [], [], label=sector, color=c, marker=:circle, ms=4)
end

savefig(p2, joinpath(PLOT_DIR, "ladder_atm_term_structure.png"))
savefig(p2, joinpath(PLOT_DIR, "ladder_atm_term_structure.pdf"))
println("  -> saved ladder_atm_term_structure.png/pdf")

# --- Plot 3: Residuals by moneyness and DTE ---
p3 = plot(title="IV Residuals (Model - Market) by Moneyness",
          xlabel="Moneyness (K/S)", ylabel="IV Residual (%)",
          legend=:outerright, size=(900, 450), dpi=150)
hline!(p3, [0.0], color=:gray, ls=:dash, label=nothing)

for sector in sort(unique(all_data.sector))
    mask = all_data.sector .== sector
    c = get(sector_colors, sector, :gray)
    scatter!(p3, all_data.moneyness[mask],
             all_data.residual[mask] .* 100,
             label=sector, marker=:circle, ms=2, color=c, alpha=0.3)
end
savefig(p3, joinpath(PLOT_DIR, "ladder_residuals_moneyness.png"))
savefig(p3, joinpath(PLOT_DIR, "ladder_residuals_moneyness.pdf"))
println("  -> saved ladder_residuals_moneyness.png/pdf")

# --- Plot 4: Per-ticker theta_base bar chart ---
sorted_idx = sortperm(theta_opt; rev=true)
sorted_tickers = tickers[sorted_idx]
sorted_theta = theta_opt[sorted_idx]
sorted_iv = sqrt.(sorted_theta) .* 100
sorted_colors = [get(sector_colors, get(SECTORS, t, "Other"), :gray) for t in sorted_tickers]

p4 = bar(sorted_tickers, sorted_iv,
         title="Per-Ticker Baseline IV (theta_base)",
         xlabel="Ticker", ylabel="Baseline IV (%)",
         legend=false, size=(900, 450), dpi=150,
         color=sorted_colors, alpha=0.8,
         xrotation=45)

# Add sector legend
for (sector, c) in sort(collect(sector_colors))
    bar!(p4, [], [], label=sector, color=c)
end
plot!(p4, legend=:topright)

savefig(p4, joinpath(PLOT_DIR, "ladder_theta_base_bar.png"))
savefig(p4, joinpath(PLOT_DIR, "ladder_theta_base_bar.pdf"))
println("  -> saved ladder_theta_base_bar.png/pdf")

# --- Plot 5: IV surface heatmap (using SPY as example) ---
p5 = plot(title="Model IV Surface (SPY, shared beta)",
          xlabel="Moneyness (K/S)", ylabel="Days to Expiration",
          size=(700, 500), dpi=150)

spy_idx = ticker_idx["SPY"]
m_grid = range(0.85, 1.15, length=60)
dte_grid = range(2, 70, length=60)
iv_surface = [sqrt(max(theta_opt[spy_idx] * eval_psi(beta_opt, log(d), log(m)), 1e-10)) * 100
              for d in dte_grid, m in m_grid]

heatmap!(p5, collect(m_grid), collect(dte_grid), iv_surface,
         colorbar_title="IV (%)", color=:viridis)

# Overlay SPY market points
spy_data = all_data[all_data.ticker .== "SPY", :]
scatter!(p5, spy_data.moneyness, Float64.(spy_data.actual_dte),
         marker=:circle, ms=2, color=:white, alpha=0.6, label="Market obs")

savefig(p5, joinpath(PLOT_DIR, "ladder_iv_surface_spy.png"))
savefig(p5, joinpath(PLOT_DIR, "ladder_iv_surface_spy.pdf"))
println("  -> saved ladder_iv_surface_spy.png/pdf")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^70)
println("  SUMMARY")
println("="^70)
println("\n  Joint calibration on $n_tickers tickers, $N_OBS observations")
println("  Overall RMSE: $(round(overall_rmse * 100, digits=2))% IV")
println("\n  Shared beta (smile shape):")
for (i, label) in enumerate(labels)
    @printf("    %-18s = %+.4f\n", label, beta_opt[i])
end
println("\n  These beta values plug directly into ThetaHybrid for scenario generation.")
println("  Per-ticker theta_base provides the volatility level for each asset.")
