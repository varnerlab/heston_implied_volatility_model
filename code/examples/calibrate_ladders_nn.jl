"""
Neural Network Calibration of IV Model from Multi-Ticker Ladder Data

Replaces the parametric psi(beta, DTE, K/S) with a learned neural network:

    sigma_model(ticker, K, DTE) = sqrt(theta_base[ticker] * psi_NN(ln(DTE), ln(K/S)))

where psi_NN is a small MLP: 2 inputs -> 32 -> 32 -> 1 (with exp output to ensure psi > 0).

The shared psi_NN learns the smile/term-structure geometry from data without imposing
a functional form. Per-ticker theta_base (also learned) sets the volatility level.

Data: 23 tickers, ~18k observations from ladder CSVs (2026-04-14).
"""

using CSV
using DataFrames
using Statistics
using Flux
using Plots
using Printf
using Random
using LinearAlgebra

# ============================================================================
# Step 1: Load and filter ladder data (same as parametric version)
# ============================================================================

const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladder")
const PLOT_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(PLOT_DIR)

const SECTORS = Dict(
    "AAPL" => "Tech", "AMD" => "Tech", "AVGO" => "Tech", "GOOG" => "Tech",
    "INTC" => "Tech", "META" => "Tech", "MSFT" => "Tech", "MU" => "Tech",
    "NVDA" => "Tech", "QCOM" => "Tech",
    "BAC" => "Financials", "GS" => "Financials", "JPM" => "Financials",
    "WFC" => "Financials",
    "CVX" => "Energy", "OXY" => "Energy", "XOM" => "Energy",
    "ABBV" => "Healthcare", "AMGN" => "Healthcare", "BMY" => "Healthcare",
    "JNJ" => "Healthcare", "LLY" => "Healthcare", "MRNA" => "Healthcare",
    "PFE" => "Healthcare", "UNH" => "Healthcare",
    "TGT" => "Retail", "UPS" => "Retail", "WMT" => "Retail",
    "IWM" => "ETF", "QQQ" => "ETF", "SPY" => "ETF",
)

function load_ladder(filepath::String)
    df = CSV.read(filepath, DataFrame)
    ticker = string(df.underlying[1])
    S = df.und_close[1]
    df[!, :ticker] .= ticker
    df[!, :S] .= S
    df[!, :moneyness] = df.strike ./ S

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

function load_all_ladders(dir::String)
    files = String[]
    for (root, _, fs) in walkdir(dir)
        for f in fs
            endswith(f, ".csv") && push!(files, joinpath(root, f))
        end
    end
    frames = DataFrame[]
    for f in files
        df = load_ladder(f)
        nrow(df) > 0 && push!(frames, df)
    end
    return vcat(frames...)
end

println("Loading ladder data...")
all_data = load_all_ladders(LADDER_DIR)
tickers = sort(unique(all_data.ticker))
n_tickers = length(tickers)
ticker_idx = Dict(t => i for (i, t) in enumerate(tickers))

println("  $(nrow(all_data)) observations, $n_tickers tickers")
for t in tickers
    n = sum(all_data.ticker .== t)
    sector = get(SECTORS, t, "Other")
    @printf("    %-5s [%-11s]: %4d obs\n", t, sector, n)
end

# ============================================================================
# Step 2: Prepare training data
# ============================================================================

println("\nPreparing training data...")

# NN inputs: [ln(DTE), ln(K/S)] — standardized for stable training
log_dte_raw = log.(max.(Float64.(all_data.actual_dte), 1.0))
log_m_raw = log.(Float64.(all_data.moneyness))

# Standardize inputs (zero mean, unit variance)
mu_dte, sigma_dte = mean(log_dte_raw), std(log_dte_raw)
mu_m, sigma_m = mean(log_m_raw), std(log_m_raw)

log_dte_std = (log_dte_raw .- mu_dte) ./ sigma_dte
log_m_std = (log_m_raw .- mu_m) ./ sigma_m

# Input matrix: 2 x N_obs
X = Float32.(hcat(log_dte_std, log_m_std)')  # 2 x N

# Target: market IV
target_iv = Float32.(all_data.implied_vol)

# Ticker indices for each observation
obs_tidx = Int32[ticker_idx[t] for t in all_data.ticker]

N_OBS = length(target_iv)
println("  Input shape: $(size(X))")
println("  Standardization: ln(DTE) ~ N($(round(mu_dte, digits=2)), $(round(sigma_dte, digits=2)))")
println("  Standardization: ln(K/S) ~ N($(round(mu_m, digits=4)), $(round(sigma_m, digits=4)))")

# ============================================================================
# Step 3: Define model — psi_NN + per-ticker log(theta_base)
# ============================================================================

println("\nBuilding model...")

# psi network: 2 -> 16 -> 16 -> 1, output through exp() to ensure psi > 0
Random.seed!(42)
psi_nn = Chain(
    Dense(2 => 16, tanh),
    Dense(16 => 16, tanh),
    Dense(16 => 1),
)

# Per-ticker log(theta_base) — initialized from mean IV^2 per ticker
log_theta_init = Float32[]
for t in tickers
    mask = all_data.ticker .== t
    mean_iv = mean(Float64.(all_data.implied_vol[mask]))
    push!(log_theta_init, Float32(log(mean_iv^2)))
end
log_theta = log_theta_init  # mutable vector, will be wrapped as a Flux parameter

# Count parameters
n_nn_params = length(Flux.destructure(psi_nn)[1])
n_theta_params = n_tickers
n_total_params = n_nn_params + n_theta_params
println("  psi_NN parameters: $n_nn_params")
println("  theta_base parameters: $n_theta_params")
println("  Total parameters: $n_total_params")
println("  Observations per parameter: $(round(N_OBS / n_total_params, digits=0))")

# ============================================================================
# Step 4: Training loop
# ============================================================================

println("\n" * "="^70)
println("  TRAINING NEURAL NETWORK PSI")
println("="^70)

# Batch computation of model IVs
function compute_model_ivs(psi_nn, log_theta, X, obs_tidx)
    # psi_NN output: 1 x N_obs
    log_psi = psi_nn(X)  # 1 x N

    # Gather theta_base for each observation
    theta_base = exp.(log_theta[obs_tidx])  # N

    # sigma_model = sqrt(theta_base * exp(log_psi))
    # = sqrt(exp(log(theta_base) + log_psi))
    # = exp(0.5 * (log(theta_base) + log_psi))
    log_theta_obs = log_theta[obs_tidx]  # N
    model_iv = exp.(Float32(0.5) .* (log_theta_obs .+ vec(log_psi)))

    return model_iv
end

function loss_fn(psi_nn, log_theta, X, target_iv, obs_tidx)
    model_iv = compute_model_ivs(psi_nn, log_theta, X, obs_tidx)
    return Flux.mse(model_iv, target_iv)
end

# Wrap both components in a single model for Flux
model = (psi_nn = psi_nn, log_theta = log_theta)

# Optimizer: Adam with learning rate scheduling
opt_state = Flux.setup(Adam(1e-3), model)

n_epochs = 2000
best_loss = Inf
best_model_state = nothing
patience = 200
no_improve = 0

println("\n  Training for up to $n_epochs epochs (patience=$patience)...")
println("  Epoch      Loss       RMSE(%)    Best RMSE(%)")
println("  " * "-"^50)

for epoch in 1:n_epochs
    # Compute loss and gradients
    l, grads = Flux.withgradient(model) do m
        model_iv = compute_model_ivs(m.psi_nn, m.log_theta, X, obs_tidx)
        Flux.mse(model_iv, target_iv)
    end

    # Update parameters
    Flux.update!(opt_state, model, grads[1])

    rmse_pct = sqrt(l) * 100

    if l < best_loss
        global best_loss = l
        global best_model_state = Flux.state(model)
        global no_improve = 0
    else
        global no_improve += 1
    end

    if epoch % 100 == 0 || epoch == 1
        best_rmse = sqrt(best_loss) * 100
        @printf("  %5d    %.6f    %5.2f      %5.2f\n", epoch, l, rmse_pct, best_rmse)
    end

    # Early stopping
    if no_improve >= patience
        println("  Early stopping at epoch $epoch (no improvement for $patience epochs)")
        break
    end

    # Learning rate decay at epoch milestones
    if epoch == 500
        Flux.adjust!(opt_state, 5e-4)
        println("  [LR -> 5e-4]")
    elseif epoch == 1000
        Flux.adjust!(opt_state, 2e-4)
        println("  [LR -> 2e-4]")
    elseif epoch == 1500
        Flux.adjust!(opt_state, 1e-4)
        println("  [LR -> 1e-4]")
    end
end

# Restore best model
Flux.loadmodel!(model, best_model_state)
theta_opt = exp.(Float64.(model.log_theta))

# Final model IVs
model_ivs = Float64.(compute_model_ivs(model.psi_nn, model.log_theta, X, obs_tidx))
overall_rmse = sqrt(mean((model_ivs .- Float64.(target_iv)).^2))

println("\n  Final RMSE: $(round(overall_rmse * 100, digits=2))% IV")

# ============================================================================
# Step 5: Results
# ============================================================================

println("\n" * "="^70)
println("  CALIBRATED PARAMETERS")
println("="^70)

println("\n  psi_NN: learned ($(n_nn_params) parameters, not interpretable as betas)")
println("  Overall RMSE = $(round(overall_rmse * 100, digits=2))% IV")

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
# Step 6: Per-ticker and per-DTE validation
# ============================================================================

all_data[!, :model_iv] = model_ivs
all_data[!, :residual] = model_ivs .- Float64.(target_iv)

println("\n" * "="^70)
println("  PER-TICKER FIT QUALITY")
println("="^70)
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

# Sector-level
all_data[!, :sector] = [get(SECTORS, t, "Other") for t in all_data.ticker]
println("\n" * "="^70)
println("  SECTOR-LEVEL ANALYSIS")
println("="^70)
println("\n  Sector         N     RMSE(%)  Bias(%)  Mean IV(%)")
println("  " * "-"^55)
for sector in sort(unique(all_data.sector))
    mask = all_data.sector .== sector
    res = all_data.residual[mask]
    mean_iv = mean(Float64.(target_iv[mask])) * 100
    @printf("  %-12s %5d    %5.2f    %+5.2f     %5.1f\n",
            sector, sum(mask), sqrt(mean(res.^2)) * 100, mean(res) * 100, mean_iv)
end

# ============================================================================
# Step 7: Comparison with parametric model
# ============================================================================

println("\n" * "="^70)
println("  COMPARISON: PARAMETRIC vs NEURAL NETWORK")
println("="^70)

# Parametric results (from calibrate_ladders.jl run)
parametric_rmse = 9.36  # % IV, from the prior run
println("\n  Parametric psi (5 betas):  RMSE = $(parametric_rmse)% IV")
println("  Neural network psi (NN):   RMSE = $(round(overall_rmse * 100, digits=2))% IV")
improvement = parametric_rmse - overall_rmse * 100
println("  Improvement:               $(round(improvement, digits=2))% IV ($(round(improvement/parametric_rmse*100, digits=1))% relative)")

# ============================================================================
# Step 8: Diagnostic plots
# ============================================================================

println("\nGenerating diagnostic plots...")

sector_colors = Dict("Tech" => :blue, "Financials" => :red, "Energy" => :green,
                     "Healthcare" => :purple, "Retail" => :orange, "ETF" => :black)

# Helper: evaluate psi_NN on a grid (for plotting model curves)
function eval_model_iv_grid(psi_nn, log_theta_t, m_range, dte_val)
    log_dte_val = (log(max(dte_val, 1.0)) - mu_dte) / sigma_dte
    log_m_vals = [(log(m) - mu_m) / sigma_m for m in m_range]
    x_grid = Float32.(hcat(fill(log_dte_val, length(m_range)), log_m_vals)')
    log_psi = vec(psi_nn(x_grid))
    ivs = exp.(Float32(0.5) .* (Float32(log_theta_t) .+ log_psi))
    return Float64.(ivs) .* 100
end

# --- Plot 1: Representative IV smiles (6-panel) ---
representative = filter(t -> t in tickers, ["SPY", "NVDA", "JPM", "XOM", "LLY", "WMT"])

p_panels = []
for t in representative
    slice = all_data[all_data.ticker .== t, :]
    avail_dtes = sort(unique(slice.actual_dte))
    target_dte = avail_dtes[max(1, length(avail_dtes) ÷ 2)]
    dte_slice = slice[slice.actual_dte .== target_dte, :]

    sector = get(SECTORS, t, "Other")
    t_idx = ticker_idx[t]

    p = plot(title="$t [$sector] DTE=$target_dte",
             xlabel="K/S", ylabel="IV (%)",
             legend=:topright, titlefontsize=9)

    calls = dte_slice[dte_slice.type .== "call", :]
    puts = dte_slice[dte_slice.type .== "put", :]

    scatter!(p, calls.moneyness, Float64.(calls.implied_vol) .* 100,
             label="Calls", marker=:circle, ms=3, color=:blue, alpha=0.6)
    scatter!(p, puts.moneyness, Float64.(puts.implied_vol) .* 100,
             label="Puts", marker=:diamond, ms=3, color=:red, alpha=0.6)

    m_range = range(0.85, 1.15, length=80)
    iv_curve = eval_model_iv_grid(model.psi_nn, model.log_theta[t_idx], m_range, target_dte)
    plot!(p, m_range, iv_curve, label="NN Model", lw=2, color=:black)
    vline!(p, [1.0], label=nothing, ls=:dash, color=:gray, alpha=0.5)

    push!(p_panels, p)
end

p1 = plot(p_panels..., layout=(2, 3), size=(1200, 700), dpi=150,
          plot_title="IV Smile Fit — Neural Network psi (shared)")
savefig(p1, joinpath(PLOT_DIR, "ladder_nn_smile_panels.png"))
savefig(p1, joinpath(PLOT_DIR, "ladder_nn_smile_panels.pdf"))
println("  -> saved ladder_nn_smile_panels.png/pdf")

# --- Plot 2: ATM term structure ---
p2 = plot(title="ATM IV Term Structure — NN Model",
          xlabel="Days to Expiration", ylabel="ATM IV (%)",
          legend=:outerright, size=(900, 500), dpi=150)

for t in tickers
    slice = all_data[(all_data.ticker .== t) .& (abs.(all_data.moneyness .- 1.0) .< 0.05), :]
    nrow(slice) == 0 && continue
    atm_by_dte = combine(groupby(slice, :actual_dte),
                         :implied_vol => (x -> mean(Float64.(x))) => :mean_iv)
    sector = get(SECTORS, t, "Other")
    c = get(sector_colors, sector, :gray)
    scatter!(p2, atm_by_dte.actual_dte, atm_by_dte.mean_iv .* 100,
             label=nothing, marker=:circle, ms=3, color=c, alpha=0.4)
end

dte_range = 2:80
for t in ["SPY", "NVDA", "JPM", "XOM"]
    t in tickers || continue
    t_idx = ticker_idx[t]
    iv_atm = eval_model_iv_grid(model.psi_nn, model.log_theta[t_idx], [1.0], 2)[1]  # placeholder
    # Compute full curve
    iv_curve = Float64[]
    for d in dte_range
        iv_val = eval_model_iv_grid(model.psi_nn, model.log_theta[t_idx], [1.0], Float64(d))[1]
        push!(iv_curve, iv_val)
    end
    sector = get(SECTORS, t, "Other")
    c = get(sector_colors, sector, :gray)
    plot!(p2, dte_range, iv_curve, label=t, lw=2, color=c)
end

for (sector, c) in sort(collect(sector_colors))
    scatter!(p2, [], [], label=sector, color=c, marker=:circle, ms=4)
end

savefig(p2, joinpath(PLOT_DIR, "ladder_nn_atm_term_structure.png"))
savefig(p2, joinpath(PLOT_DIR, "ladder_nn_atm_term_structure.pdf"))
println("  -> saved ladder_nn_atm_term_structure.png/pdf")

# --- Plot 3: Residuals by moneyness ---
p3 = plot(title="IV Residuals (NN Model - Market) by Moneyness",
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
savefig(p3, joinpath(PLOT_DIR, "ladder_nn_residuals_moneyness.png"))
savefig(p3, joinpath(PLOT_DIR, "ladder_nn_residuals_moneyness.pdf"))
println("  -> saved ladder_nn_residuals_moneyness.png/pdf")

# --- Plot 4: Per-ticker theta_base bar chart ---
sorted_idx = sortperm(theta_opt; rev=true)
sorted_tickers = tickers[sorted_idx]
sorted_iv = sqrt.(theta_opt[sorted_idx]) .* 100
sorted_colors = [get(sector_colors, get(SECTORS, t, "Other"), :gray) for t in sorted_tickers]

p4 = bar(sorted_tickers, sorted_iv,
         title="Per-Ticker Baseline IV — NN Model",
         xlabel="Ticker", ylabel="Baseline IV (%)",
         legend=false, size=(900, 450), dpi=150,
         color=sorted_colors, alpha=0.8, xrotation=45)
for (sector, c) in sort(collect(sector_colors))
    bar!(p4, [], [], label=sector, color=c)
end
plot!(p4, legend=:topright)

savefig(p4, joinpath(PLOT_DIR, "ladder_nn_theta_base_bar.png"))
savefig(p4, joinpath(PLOT_DIR, "ladder_nn_theta_base_bar.pdf"))
println("  -> saved ladder_nn_theta_base_bar.png/pdf")

# --- Plot 5: Learned psi surface heatmap ---
m_grid = range(0.85, 1.15, length=80)
dte_grid = range(2, 70, length=80)

# Evaluate psi_NN on the grid (ticker-independent — psi is shared)
psi_surface = Matrix{Float64}(undef, length(dte_grid), length(m_grid))
for (j, m) in enumerate(m_grid)
    for (i, d) in enumerate(dte_grid)
        log_dte_s = Float32((log(d) - mu_dte) / sigma_dte)
        log_m_s = Float32((log(m) - mu_m) / sigma_m)
        x_in = Float32[log_dte_s, log_m_s]
        psi_surface[i, j] = exp(Float64(model.psi_nn(x_in)[1]))
    end
end

p5 = heatmap(collect(m_grid), collect(dte_grid), psi_surface,
             title="Learned psi Surface (shared across tickers)",
             xlabel="Moneyness (K/S)", ylabel="Days to Expiration",
             colorbar_title="psi", color=:viridis,
             size=(700, 500), dpi=150)
savefig(p5, joinpath(PLOT_DIR, "ladder_nn_psi_surface.png"))
savefig(p5, joinpath(PLOT_DIR, "ladder_nn_psi_surface.pdf"))
println("  -> saved ladder_nn_psi_surface.png/pdf")

# --- Plot 6: IV surface for SPY (NN model) ---
spy_idx = ticker_idx["SPY"]
iv_surface_spy = Matrix{Float64}(undef, length(dte_grid), length(m_grid))
for (j, m) in enumerate(m_grid)
    for (i, d) in enumerate(dte_grid)
        log_dte_s = Float32((log(d) - mu_dte) / sigma_dte)
        log_m_s = Float32((log(m) - mu_m) / sigma_m)
        x_in = Float32[log_dte_s, log_m_s]
        psi_val = exp(Float64(model.psi_nn(x_in)[1]))
        iv_surface_spy[i, j] = sqrt(theta_opt[spy_idx] * psi_val) * 100
    end
end

p6 = heatmap(collect(m_grid), collect(dte_grid), iv_surface_spy,
             title="SPY IV Surface — NN Model",
             xlabel="Moneyness (K/S)", ylabel="Days to Expiration",
             colorbar_title="IV (%)", color=:viridis,
             size=(700, 500), dpi=150)

spy_data = all_data[all_data.ticker .== "SPY", :]
scatter!(p6, spy_data.moneyness, Float64.(spy_data.actual_dte),
         marker=:circle, ms=2, color=:white, alpha=0.6, label="Market obs")

savefig(p6, joinpath(PLOT_DIR, "ladder_nn_iv_surface_spy.png"))
savefig(p6, joinpath(PLOT_DIR, "ladder_nn_iv_surface_spy.pdf"))
println("  -> saved ladder_nn_iv_surface_spy.png/pdf")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^70)
println("  SUMMARY")
println("="^70)
println("\n  Neural network psi calibration on $n_tickers tickers, $N_OBS observations")
println("  NN architecture: 2 -> 16 -> 16 -> 1 (tanh, exp output)")
println("  Total parameters: $n_total_params ($n_nn_params NN + $n_theta_params theta)")
println("  Overall RMSE: $(round(overall_rmse * 100, digits=2))% IV")
println("\n  vs parametric (5 betas): ~$(parametric_rmse)% IV RMSE")
println("  The learned psi surface captures smile/term-structure geometry")
println("  without imposing a functional form.")
