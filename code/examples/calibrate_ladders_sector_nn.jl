"""
Sector-Specific Neural Network Calibration of IV Model

Trains a separate psi_NN per sector (Tech, Financials, Energy, Healthcare, Retail, ETF),
each learning the smile/term-structure geometry for its sector:

    sigma_model(ticker, K, DTE) = sqrt(theta_base[ticker] * psi_NN[sector](ln(DTE), ln(K/S)))

Larger sectors (ETF, Tech) get 2->16->16->1 networks; smaller sectors (Energy, Retail)
get 2->8->8->1 to avoid overfitting on fewer observations.

Comparison: parametric (5 betas) vs shared NN vs sector-specific NN.
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
# Step 1: Load and filter ladder data
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
all_data[!, :sector] = [get(SECTORS, t, "Other") for t in all_data.ticker]
tickers = sort(unique(all_data.ticker))
n_tickers = length(tickers)
ticker_idx = Dict(t => i for (i, t) in enumerate(tickers))
sectors = sort(unique(all_data.sector))

println("  $(nrow(all_data)) observations, $n_tickers tickers, $(length(sectors)) sectors\n")
for sector in sectors
    mask = all_data.sector .== sector
    sec_tickers = sort(unique(all_data.ticker[mask]))
    @printf("  %-12s: %5d obs  (%s)\n", sector, sum(mask), join(sec_tickers, ", "))
end

# ============================================================================
# Step 2: Global input standardization (same scale across all sectors)
# ============================================================================

println("\nStandardizing inputs (global)...")
log_dte_all = log.(max.(Float64.(all_data.actual_dte), 1.0))
log_m_all = log.(Float64.(all_data.moneyness))

const MU_DTE = mean(log_dte_all)
const SIGMA_DTE = std(log_dte_all)
const MU_M = mean(log_m_all)
const SIGMA_M = std(log_m_all)

println("  ln(DTE): mu=$(round(MU_DTE, digits=3)), sigma=$(round(SIGMA_DTE, digits=3))")
println("  ln(K/S): mu=$(round(MU_M, digits=4)), sigma=$(round(SIGMA_M, digits=4))")

# ============================================================================
# Step 3: Train one psi_NN per sector
# ============================================================================

println("\n" * "="^70)
println("  TRAINING SECTOR-SPECIFIC PSI NETWORKS")
println("="^70)

# Architecture sizing: larger sectors get bigger networks
function make_psi_nn(n_obs::Int)
    if n_obs >= 2000
        return Chain(Dense(2 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 1))
    else
        return Chain(Dense(2 => 8, tanh), Dense(8 => 8, tanh), Dense(8 => 1))
    end
end

function compute_model_ivs(psi_nn, log_theta, X, obs_tidx)
    log_psi = psi_nn(X)
    log_theta_obs = log_theta[obs_tidx]
    return exp.(Float32(0.5) .* (log_theta_obs .+ vec(log_psi)))
end

"""Train a sector model. Returns (trained_model, training_history)."""
function train_sector(sector_name::AbstractString, sector_data::DataFrame,
                      sector_tickers::AbstractVector{<:AbstractString},
                      sector_ticker_idx::Dict{<:AbstractString,Int};
                      n_epochs::Int=2000, patience::Int=200, seed::Int=42)

    n_obs = nrow(sector_data)
    n_sector_tickers = length(sector_tickers)

    # Prepare inputs
    log_dte = Float32.((log.(max.(Float64.(sector_data.actual_dte), 1.0)) .- MU_DTE) ./ SIGMA_DTE)
    log_m = Float32.((log.(Float64.(sector_data.moneyness)) .- MU_M) ./ SIGMA_M)
    X = hcat(log_dte, log_m)'  # 2 x N
    target_iv = Float32.(sector_data.implied_vol)
    obs_tidx = Int32[sector_ticker_idx[t] for t in sector_data.ticker]

    # Build model
    Random.seed!(seed)
    psi_nn = make_psi_nn(n_obs)
    n_nn_params = length(Flux.destructure(psi_nn)[1])

    # Initialize log_theta from per-ticker mean IV^2
    log_theta = Float32[]
    for t in sector_tickers
        mask = sector_data.ticker .== t
        push!(log_theta, Float32(log(mean(Float64.(sector_data.implied_vol[mask]))^2)))
    end

    n_total = n_nn_params + n_sector_tickers
    arch = n_obs >= 2000 ? "2->16->16->1" : "2->8->8->1"

    println("\n  [$sector_name] $n_obs obs, $n_sector_tickers tickers, arch=$arch")
    println("    $n_nn_params NN + $n_sector_tickers theta = $n_total params ($(round(n_obs/n_total, digits=0)) obs/param)")

    # Training
    model = (psi_nn = psi_nn, log_theta = log_theta)
    opt_state = Flux.setup(Adam(1e-3), model)

    best_loss = Inf
    best_state = nothing
    no_improve = 0
    history = Float64[]

    for epoch in 1:n_epochs
        l, grads = Flux.withgradient(model) do m
            model_iv = compute_model_ivs(m.psi_nn, m.log_theta, X, obs_tidx)
            Flux.mse(model_iv, target_iv)
        end

        Flux.update!(opt_state, model, grads[1])
        push!(history, l)

        if l < best_loss
            best_loss = l
            best_state = Flux.state(model)
            no_improve = 0
        else
            no_improve += 1
        end

        if epoch % 500 == 0 || epoch == 1
            @printf("    epoch %4d: RMSE=%.2f%%  (best=%.2f%%)\n",
                    epoch, sqrt(l)*100, sqrt(best_loss)*100)
        end

        no_improve >= patience && break

        # LR schedule
        if epoch == 500
            Flux.adjust!(opt_state, 5e-4)
        elseif epoch == 1000
            Flux.adjust!(opt_state, 2e-4)
        elseif epoch == 1500
            Flux.adjust!(opt_state, 1e-4)
        end
    end

    # Restore best
    Flux.loadmodel!(model, best_state)

    # Compute final model IVs
    final_ivs = Float64.(compute_model_ivs(model.psi_nn, model.log_theta, X, obs_tidx))
    rmse = sqrt(mean((final_ivs .- Float64.(target_iv)).^2))
    println("    FINAL RMSE: $(round(rmse*100, digits=2))%")

    return model, final_ivs, rmse, history
end

# Train each sector
sector_models = Dict{String,Any}()
sector_results = Dict{String,Any}()

for sector in sectors
    mask = all_data.sector .== sector
    sector_data = all_data[mask, :]
    sector_tickers = sort(unique(sector_data.ticker))
    sector_ticker_idx = Dict(t => i for (i, t) in enumerate(sector_tickers))

    model, model_ivs, rmse, history = train_sector(
        sector, sector_data, sector_tickers, sector_ticker_idx
    )

    sector_models[sector] = (
        model = model,
        tickers = sector_tickers,
        ticker_idx = sector_ticker_idx,
    )
    sector_results[sector] = (
        model_ivs = model_ivs,
        rmse = rmse,
        history = history,
        data = sector_data,
    )
end

# ============================================================================
# Step 4: Assemble global results
# ============================================================================

# Write model IVs back to full DataFrame
all_data[!, :model_iv] .= 0.0
all_data[!, :residual] .= 0.0

for sector in sectors
    global_mask = all_data.sector .== sector
    all_data.model_iv[global_mask] .= sector_results[sector].model_ivs
    all_data.residual[global_mask] .= sector_results[sector].model_ivs .- Float64.(all_data.implied_vol[global_mask])
end

overall_rmse = sqrt(mean(all_data.residual.^2))

# ============================================================================
# Step 5: Results summary
# ============================================================================

println("\n" * "="^70)
println("  RESULTS: SECTOR-SPECIFIC NN")
println("="^70)

println("\n  Overall RMSE: $(round(overall_rmse * 100, digits=2))% IV")

println("\n  Per-sector RMSE:")
println("  Sector         N     RMSE(%)  Bias(%)  Arch")
println("  " * "-"^55)
for sector in sectors
    mask = all_data.sector .== sector
    res = all_data.residual[mask]
    n_obs = sum(mask)
    arch = n_obs >= 2000 ? "2->16->16->1" : "2->8->8->1"
    @printf("  %-12s %5d    %5.2f    %+5.2f    %s\n",
            sector, n_obs, sqrt(mean(res.^2))*100, mean(res)*100, arch)
end

# Per-ticker results
println("\n  Per-ticker RMSE:")
println("  Ticker       N     RMSE(%)  Bias(%)  Sector")
println("  " * "-"^55)
for t in tickers
    mask = all_data.ticker .== t
    res = all_data.residual[mask]
    sector = get(SECTORS, t, "Other")
    @printf("  %-5s    %5d    %5.2f    %+5.2f    %s\n",
            t, sum(mask), sqrt(mean(res.^2))*100, mean(res)*100, sector)
end

# Per-DTE results
println("\n  Per-DTE RMSE (all tickers pooled):")
println("  DTE     N     RMSE(%)  Bias(%)")
println("  " * "-"^35)
for dte in sort(unique(all_data.actual_dte))
    mask = all_data.actual_dte .== dte
    res = all_data.residual[mask]
    @printf("  %3d   %5d    %5.2f    %+5.2f\n",
            dte, sum(mask), sqrt(mean(res.^2))*100, mean(res)*100)
end

# Per-ticker theta_base
println("\n  Per-ticker theta_base (sorted by IV level):")
theta_all = Dict{String,Float64}()
for sector in sectors
    sm = sector_models[sector]
    for t in sm.tickers
        tidx = sm.ticker_idx[t]
        theta_all[t] = exp(Float64(sm.model.log_theta[tidx]))
    end
end

sorted_tickers = sort(collect(keys(theta_all)), by=t -> -theta_all[t])
for t in sorted_tickers
    theta = theta_all[t]
    sector = get(SECTORS, t, "Other")
    @printf("    %-5s [%-11s]: theta=%.4f  (%.1f%% IV)\n",
            t, sector, theta, sqrt(theta)*100)
end

# ============================================================================
# Step 6: Three-way comparison
# ============================================================================

println("\n" * "="^70)
println("  THREE-WAY COMPARISON")
println("="^70)

# Prior results (hardcoded from previous runs)
parametric_overall = 9.36
shared_nn_overall = 8.03
sector_nn_overall = overall_rmse * 100

println("\n  Model                   Overall RMSE(%)")
println("  " * "-"^45)
@printf("  Parametric (5 betas)        %5.2f\n", parametric_overall)
@printf("  Shared NN (1 network)       %5.2f\n", shared_nn_overall)
@printf("  Sector NN (6 networks)      %5.2f\n", sector_nn_overall)
println()
@printf("  Shared NN improvement:      %.2f%% IV (%.1f%% relative vs parametric)\n",
        parametric_overall - shared_nn_overall,
        (parametric_overall - shared_nn_overall) / parametric_overall * 100)
@printf("  Sector NN improvement:      %.2f%% IV (%.1f%% relative vs parametric)\n",
        parametric_overall - sector_nn_overall,
        (parametric_overall - sector_nn_overall) / parametric_overall * 100)

# Per-sector comparison
println("\n  Per-sector RMSE comparison:")
println("  Sector         Parametric  Shared NN  Sector NN")
println("  " * "-"^55)

# Parametric and shared NN per-sector (from prior runs)
parametric_sector = Dict("ETF" => 8.93, "Energy" => 7.66, "Financials" => 6.85,
                         "Healthcare" => 10.78, "Retail" => 8.62, "Tech" => 11.65)
shared_nn_sector = Dict("ETF" => 6.89, "Energy" => 6.44, "Financials" => 6.44,
                        "Healthcare" => 9.83, "Retail" => 6.94, "Tech" => 11.23)

for sector in sectors
    mask = all_data.sector .== sector
    sector_rmse = sqrt(mean(all_data.residual[mask].^2)) * 100
    p_rmse = get(parametric_sector, sector, NaN)
    s_rmse = get(shared_nn_sector, sector, NaN)
    @printf("  %-12s    %5.2f      %5.2f      %5.2f\n",
            sector, p_rmse, s_rmse, sector_rmse)
end

# ============================================================================
# Step 7: Diagnostic plots
# ============================================================================

println("\nGenerating diagnostic plots...")

sector_colors = Dict("Tech" => :blue, "Financials" => :red, "Energy" => :green,
                     "Healthcare" => :purple, "Retail" => :orange, "ETF" => :black)

# Helper: evaluate sector psi_NN on a grid
function eval_sector_iv_grid(sector::String, ticker::String,
                             m_range, dte_val::Float64)
    sm = sector_models[sector]
    psi_nn = sm.model.psi_nn
    tidx = sm.ticker_idx[ticker]
    log_theta_t = sm.model.log_theta[tidx]

    log_dte_s = Float32((log(max(dte_val, 1.0)) - MU_DTE) / SIGMA_DTE)
    log_m_vals = [Float32((log(m) - MU_M) / SIGMA_M) for m in m_range]
    x_grid = hcat(fill(log_dte_s, length(m_range)), log_m_vals)' |> Matrix{Float32}
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
    iv_curve = eval_sector_iv_grid(sector, t, m_range, Float64(target_dte))
    plot!(p, m_range, iv_curve, label="Sector NN", lw=2, color=:black)
    vline!(p, [1.0], label=nothing, ls=:dash, color=:gray, alpha=0.5)

    push!(p_panels, p)
end

p1 = plot(p_panels..., layout=(2, 3), size=(1200, 700), dpi=150,
          plot_title="IV Smile Fit — Sector-Specific NN")
savefig(p1, joinpath(PLOT_DIR, "ladder_sector_nn_smile_panels.png"))
savefig(p1, joinpath(PLOT_DIR, "ladder_sector_nn_smile_panels.pdf"))
println("  -> saved ladder_sector_nn_smile_panels.png/pdf")

# --- Plot 2: ATM term structure ---
p2 = plot(title="ATM IV Term Structure — Sector NN",
          xlabel="Days to Expiration", ylabel="ATM IV (%)",
          legend=:outerright, size=(900, 500), dpi=150)

# Market ATM points
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

# Model ATM curves
dte_range = 2:80
for t in ["SPY", "NVDA", "JPM", "XOM"]
    t in tickers || continue
    sector = get(SECTORS, t, "Other")
    c = get(sector_colors, sector, :gray)
    iv_curve = [eval_sector_iv_grid(sector, t, [1.0], Float64(d))[1] for d in dte_range]
    plot!(p2, dte_range, iv_curve, label=t, lw=2, color=c)
end

for (sector, c) in sort(collect(sector_colors))
    scatter!(p2, [], [], label=sector, color=c, marker=:circle, ms=4)
end

savefig(p2, joinpath(PLOT_DIR, "ladder_sector_nn_atm_term_structure.png"))
savefig(p2, joinpath(PLOT_DIR, "ladder_sector_nn_atm_term_structure.pdf"))
println("  -> saved ladder_sector_nn_atm_term_structure.png/pdf")

# --- Plot 3: Residuals by moneyness ---
p3 = plot(title="IV Residuals — Sector-Specific NN",
          xlabel="Moneyness (K/S)", ylabel="IV Residual (%)",
          legend=:outerright, size=(900, 450), dpi=150)
hline!(p3, [0.0], color=:gray, ls=:dash, label=nothing)

for sector in sectors
    mask = all_data.sector .== sector
    c = get(sector_colors, sector, :gray)
    scatter!(p3, all_data.moneyness[mask],
             all_data.residual[mask] .* 100,
             label=sector, marker=:circle, ms=2, color=c, alpha=0.3)
end
savefig(p3, joinpath(PLOT_DIR, "ladder_sector_nn_residuals.png"))
savefig(p3, joinpath(PLOT_DIR, "ladder_sector_nn_residuals.pdf"))
println("  -> saved ladder_sector_nn_residuals.png/pdf")

# --- Plot 4: Per-sector psi surfaces (2x3 grid) ---
m_grid = range(0.85, 1.15, length=60)
dte_grid = range(2, 70, length=60)

p_psi_panels = []
for sector in sectors
    sm = sector_models[sector]
    psi_nn = sm.model.psi_nn

    psi_surface = Matrix{Float64}(undef, length(dte_grid), length(m_grid))
    for (j, m) in enumerate(m_grid)
        for (i, d) in enumerate(dte_grid)
            log_dte_s = Float32((log(d) - MU_DTE) / SIGMA_DTE)
            log_m_s = Float32((log(m) - MU_M) / SIGMA_M)
            psi_surface[i, j] = exp(Float64(psi_nn(Float32[log_dte_s, log_m_s])[1]))
        end
    end

    c = get(sector_colors, sector, :gray)
    p = heatmap(collect(m_grid), collect(dte_grid), psi_surface,
                title="$sector psi", xlabel="K/S", ylabel="DTE",
                colorbar_title="psi", color=:viridis,
                titlefontsize=9)
    push!(p_psi_panels, p)
end

p4 = plot(p_psi_panels..., layout=(2, 3), size=(1200, 700), dpi=150,
          plot_title="Learned psi Surfaces by Sector")
savefig(p4, joinpath(PLOT_DIR, "ladder_sector_nn_psi_surfaces.png"))
savefig(p4, joinpath(PLOT_DIR, "ladder_sector_nn_psi_surfaces.pdf"))
println("  -> saved ladder_sector_nn_psi_surfaces.png/pdf")

# --- Plot 5: Three-way RMSE comparison bar chart ---
comparison_sectors = sectors
n_sec = length(comparison_sectors)
x_pos = 1:n_sec

# Build grouped bar chart manually (no StatsPlots dependency)
bar_width = 0.25
p5 = plot(title="RMSE Comparison by Sector",
          ylabel="RMSE (% IV)", legend=:topright,
          size=(800, 450), dpi=150, xrotation=20)

x_base = collect(1:n_sec)
param_vals = [parametric_sector[s] for s in comparison_sectors]
shared_vals = [shared_nn_sector[s] for s in comparison_sectors]
sector_vals = [sqrt(mean(all_data.residual[all_data.sector .== s].^2))*100 for s in comparison_sectors]

bar!(p5, x_base .- bar_width, param_vals, bar_width=bar_width,
     label="Parametric", color=:lightgray)
bar!(p5, x_base, shared_vals, bar_width=bar_width,
     label="Shared NN", color=:steelblue)
bar!(p5, x_base .+ bar_width, sector_vals, bar_width=bar_width,
     label="Sector NN", color=:darkblue)
plot!(p5, xticks=(x_base, comparison_sectors))
savefig(p5, joinpath(PLOT_DIR, "ladder_three_way_comparison.png"))
savefig(p5, joinpath(PLOT_DIR, "ladder_three_way_comparison.pdf"))
println("  -> saved ladder_three_way_comparison.png/pdf")

# --- Plot 6: Per-ticker theta_base bar chart ---
sorted_t = sort(tickers, by=t -> -theta_all[t])
sorted_iv = [sqrt(theta_all[t]) * 100 for t in sorted_t]
sorted_colors = [get(sector_colors, get(SECTORS, t, "Other"), :gray) for t in sorted_t]

p6 = bar(sorted_t, sorted_iv,
         title="Per-Ticker Baseline IV — Sector NN",
         xlabel="Ticker", ylabel="Baseline IV (%)",
         legend=false, size=(900, 450), dpi=150,
         color=sorted_colors, alpha=0.8, xrotation=45)
for (sector, c) in sort(collect(sector_colors))
    bar!(p6, [], [], label=sector, color=c)
end
plot!(p6, legend=:topright)
savefig(p6, joinpath(PLOT_DIR, "ladder_sector_nn_theta_base.png"))
savefig(p6, joinpath(PLOT_DIR, "ladder_sector_nn_theta_base.pdf"))
println("  -> saved ladder_sector_nn_theta_base.png/pdf")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^70)
println("  SUMMARY")
println("="^70)
println("\n  Sector-specific NN calibration on $n_tickers tickers, $(nrow(all_data)) observations")
println("  6 independent psi networks (one per sector)")
println("  Overall RMSE: $(round(overall_rmse * 100, digits=2))% IV")
println("\n  Three-way comparison:")
@printf("    Parametric:    %5.2f%% IV\n", parametric_overall)
@printf("    Shared NN:     %5.2f%% IV\n", shared_nn_overall)
@printf("    Sector NN:     %5.2f%% IV  (%.1f%% improvement vs parametric)\n",
        sector_nn_overall,
        (parametric_overall - sector_nn_overall) / parametric_overall * 100)

include(joinpath(@__DIR__, "..", "scripts", "promote_figures.jl"))
promote_figures()
