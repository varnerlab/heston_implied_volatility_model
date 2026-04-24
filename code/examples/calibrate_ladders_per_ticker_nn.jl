"""
Per-Ticker Neural Network Calibration of IV Model

For each ticker with at least MIN_OBS_PER_TICKER observations in the corpus,
trains a dedicated psi NN:

    sigma_model(K, DTE) = sqrt(theta_base[ticker] * psi_ticker[ticker](ln(DTE), ln(K/S)))

Tickers below the threshold fall back to the sector NN result so the paper
can still report something for them. This script trains both models on the
same data so the per-ticker vs sector comparison is self-consistent.

Outputs:
  - Per-ticker in-sample RMSE table (parametric hardcoded / sector NN / per-ticker NN)
  - Figures in code/figures/, promoted to paper/sections/figures/
"""

using CSV
using DataFrames
using Statistics
using Flux
using Plots
using Printf
using Random
using LinearAlgebra

const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladder")
const PLOT_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(PLOT_DIR)

const MIN_OBS_PER_TICKER = 2000

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

# ============================================================================
# Step 1: Load ladder data (all full-coverage day folders under LADDER_DIR)
# ============================================================================

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
        # Skip auxiliary data folders that don't hold per-ticker ladder CSVs.
        occursin("VIX-data", root) && continue
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
sectors = sort(unique(all_data.sector))

println("  $(nrow(all_data)) observations, $(length(tickers)) tickers, $(length(sectors)) sectors\n")

# ============================================================================
# Step 2: Standardize inputs (global, same as sector NN for comparability)
# ============================================================================

const MU_DTE = mean(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const SIGMA_DTE = std(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const MU_M = mean(log.(Float64.(all_data.moneyness)))
const SIGMA_M = std(log.(Float64.(all_data.moneyness)))

# ============================================================================
# Step 3: Per-ticker obs counts + qualification split
# ============================================================================

obs_by_ticker = Dict(t => sum(all_data.ticker .== t) for t in tickers)
qualified = sort([t for t in tickers if obs_by_ticker[t] >= MIN_OBS_PER_TICKER])
unqualified = sort([t for t in tickers if obs_by_ticker[t] < MIN_OBS_PER_TICKER])

println("Qualification threshold: N >= $MIN_OBS_PER_TICKER")
println("  Qualified ($(length(qualified))):   ", join(qualified, ", "))
println("  Unqualified ($(length(unqualified))): ", join(unqualified, ", "))
println()

# ============================================================================
# Shared training helpers
# ============================================================================

function compute_model_ivs(psi_nn, log_theta, X, obs_tidx)
    log_psi = psi_nn(X)
    log_theta_obs = log_theta[obs_tidx]
    return exp.(Float32(0.5) .* (log_theta_obs .+ vec(log_psi)))
end

# Scalar theta variant (per-ticker NNs have only one ticker, so log_theta is a scalar)
function compute_model_ivs_scalar(psi_nn, log_theta_scalar::AbstractVector{Float32}, X)
    log_psi = psi_nn(X)
    return exp.(Float32(0.5) .* (log_theta_scalar[1] .+ vec(log_psi)))
end

"""
Train a network with Adam + LR schedule + early stopping.
Returns the final (model, best_loss) after loading the best state.
"""
function train_with_schedule!(model, Xs, ys, loss_fn;
                              n_epochs::Int=2000, patience::Int=200,
                              lr_init::Float64=1e-3)
    opt_state = Flux.setup(Adam(Float32(lr_init)), model)
    best_loss = Inf
    best_state = nothing
    no_improve = 0

    for epoch in 1:n_epochs
        l, grads = Flux.withgradient(model) do m
            loss_fn(m, Xs, ys)
        end
        Flux.update!(opt_state, model, grads[1])

        if l < best_loss
            best_loss = l
            best_state = Flux.state(model)
            no_improve = 0
        else
            no_improve += 1
        end
        no_improve >= patience && break

        epoch == 500  && Flux.adjust!(opt_state, 5f-4)
        epoch == 1000 && Flux.adjust!(opt_state, 2f-4)
        epoch == 1500 && Flux.adjust!(opt_state, 1f-4)
    end

    Flux.loadmodel!(model, best_state)
    return model, best_loss
end

# ============================================================================
# Step 4: Train sector NNs (for the comparison baseline)
# ============================================================================

println("="^70)
println("  TRAINING SECTOR NNS (baseline for comparison)")
println("="^70)

sector_models = Dict{String,Any}()
sector_data_cache = Dict{String,Any}()

for sector in sectors
    mask = all_data.sector .== sector
    sector_data = all_data[mask, :]
    sector_tickers = sort(unique(sector_data.ticker))
    sector_ticker_idx = Dict(t => i for (i, t) in enumerate(sector_tickers))
    n_obs = nrow(sector_data)

    log_dte_s = Float32.((log.(max.(Float64.(sector_data.actual_dte), 1.0)) .- MU_DTE) ./ SIGMA_DTE)
    log_m_s = Float32.((log.(Float64.(sector_data.moneyness)) .- MU_M) ./ SIGMA_M)
    Xs = hcat(log_dte_s, log_m_s)'
    ys = Float32.(sector_data.implied_vol)
    tidx_s = Int32[sector_ticker_idx[t] for t in sector_data.ticker]

    Random.seed!(42)
    psi_nn = n_obs >= 2000 ?
        Chain(Dense(2 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 1)) :
        Chain(Dense(2 => 8, tanh),  Dense(8 => 8, tanh),   Dense(8 => 1))
    log_theta = Float32[Float32(log(mean(Float64.(sector_data.implied_vol[sector_data.ticker .== t]))^2))
                        for t in sector_tickers]
    model = (psi_nn = psi_nn, log_theta = log_theta)

    sector_loss(m, X, y) = Flux.mse(compute_model_ivs(m.psi_nn, m.log_theta, X, tidx_s), y)
    train_with_schedule!(model, Xs, ys, sector_loss)

    final_ivs = Float64.(compute_model_ivs(model.psi_nn, model.log_theta, Xs, tidx_s))
    rmse_sector = sqrt(mean((final_ivs .- Float64.(ys)).^2))
    @printf("  [%-11s] %5d obs, %2d tickers  -> RMSE %5.2f%%\n",
            sector, n_obs, length(sector_tickers), rmse_sector*100)

    sector_models[sector] = (model = model, tickers = sector_tickers, ticker_idx = sector_ticker_idx)
    sector_data_cache[sector] = (data = sector_data, model_ivs = final_ivs)
end

# Per-ticker sector-NN RMSE (will be the baseline)
sector_rmse_by_ticker = Dict{String,Float64}()
for sector in sectors
    sc = sector_data_cache[sector]
    for t in sector_models[sector].tickers
        mask = sc.data.ticker .== t
        sector_rmse_by_ticker[t] = sqrt(mean((sc.model_ivs[mask] .- Float64.(sc.data.implied_vol[mask])).^2))
    end
end

# ============================================================================
# Step 5: Train per-ticker NNs (2->8->8->1) for qualified tickers
# ============================================================================

println("\n" * "="^70)
println("  TRAINING PER-TICKER NNS (N >= $MIN_OBS_PER_TICKER)")
println("="^70)

per_ticker_models = Dict{String,Any}()
per_ticker_rmse = Dict{String,Float64}()
per_ticker_preds = Dict{String,Vector{Float64}}()

for t in qualified
    mask = all_data.ticker .== t
    td = all_data[mask, :]
    n_obs = nrow(td)

    log_dte_t = Float32.((log.(max.(Float64.(td.actual_dte), 1.0)) .- MU_DTE) ./ SIGMA_DTE)
    log_m_t = Float32.((log.(Float64.(td.moneyness)) .- MU_M) ./ SIGMA_M)
    Xt = hcat(log_dte_t, log_m_t)'
    yt = Float32.(td.implied_vol)

    Random.seed!(42)
    # Adaptive arch: data-rich tickers (N >= 5000) get the same 2->16->16->1
    # network the sector NN uses; smaller-N tickers stay on 2->8->8->1 to
    # avoid overfitting.
    psi_nn = n_obs >= 5000 ?
        Chain(Dense(2 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 1)) :
        Chain(Dense(2 => 8, tanh),  Dense(8 => 8, tanh),   Dense(8 => 1))
    arch_label = n_obs >= 5000 ? "2->16->16->1" : "2->8->8->1"
    n_nn_params = length(Flux.destructure(psi_nn)[1])
    log_theta = Float32[Float32(log(mean(Float64.(td.implied_vol))^2))]
    model = (psi_nn = psi_nn, log_theta = log_theta)

    obs_per_param = round(n_obs / (n_nn_params + 1), digits=1)
    ticker_loss(m, X, y) = Flux.mse(compute_model_ivs_scalar(m.psi_nn, m.log_theta, X), y)
    train_with_schedule!(model, Xt, yt, ticker_loss)

    preds = Float64.(compute_model_ivs_scalar(model.psi_nn, model.log_theta, Xt))
    rmse_t = sqrt(mean((preds .- Float64.(yt)).^2))

    per_ticker_models[t] = (model = model, mask = mask)
    per_ticker_rmse[t] = rmse_t
    per_ticker_preds[t] = preds

    sector_rmse = sector_rmse_by_ticker[t] * 100
    delta = sector_rmse - rmse_t*100
    @printf("  %-5s [%-11s]  N=%5d  arch=%s (%d params, %.1f obs/param)  sector=%.2f%%  per-ticker=%.2f%%  delta=%+.2f\n",
            t, get(SECTORS, t, "Other"), n_obs, arch_label, n_nn_params, obs_per_param,
            sector_rmse, rmse_t*100, delta)
end

# ============================================================================
# Step 6: Assemble the comparison DataFrame
# ============================================================================

println("\n" * "="^70)
println("  PER-TICKER COMPARISON")
println("="^70)

rows = NamedTuple[]
for t in tickers
    is_q = t in Set(qualified)
    sec_rmse = sector_rmse_by_ticker[t] * 100
    pt_rmse  = is_q ? per_ticker_rmse[t]*100 : NaN
    delta    = is_q ? sec_rmse - pt_rmse : NaN
    push!(rows, (
        ticker = t, sector = get(SECTORS, t, "Other"),
        n_obs = obs_by_ticker[t], qualified = is_q,
        sector_rmse = sec_rmse, per_ticker_rmse = pt_rmse, improvement = delta,
    ))
end
cmp_df = DataFrame(rows)

# Table: qualified tickers sorted by improvement (descending)
q_sorted = sort(filter(r -> r.qualified, cmp_df), :improvement, rev=true)
println("\n  Qualified tickers (sorted by improvement over sector NN):")
println("  Ticker  Sector       N_obs   Sector%  Per-ticker%  Delta%")
println("  " * "-"^65)
for r in eachrow(q_sorted)
    @printf("  %-5s   %-11s  %5d     %5.2f       %5.2f     %+5.2f\n",
            r.ticker, r.sector, r.n_obs, r.sector_rmse, r.per_ticker_rmse, r.improvement)
end

# Unqualified tickers: stay sector-pooled
uq_sorted = sort(filter(r -> !r.qualified, cmp_df), :n_obs, rev=true)
if nrow(uq_sorted) > 0
    println("\n  Unqualified tickers (stay on sector NN, below N=$MIN_OBS_PER_TICKER):")
    println("  Ticker  Sector       N_obs   Sector%")
    println("  " * "-"^45)
    for r in eachrow(uq_sorted)
        @printf("  %-5s   %-11s  %5d     %5.2f\n",
                r.ticker, r.sector, r.n_obs, r.sector_rmse)
    end
end

# Overall headline: compute a combined RMSE using per-ticker for qualified + sector for unqualified
combined_residuals_sq = Float64[]
for t in tickers
    mask = all_data.ticker .== t
    target = Float64.(all_data.implied_vol[mask])
    if t in Set(qualified)
        preds = per_ticker_preds[t]
        append!(combined_residuals_sq, (preds .- target).^2)
    else
        sector = get(SECTORS, t, "Other")
        sc = sector_data_cache[sector]
        sc_mask = sc.data.ticker .== t
        append!(combined_residuals_sq, (sc.model_ivs[sc_mask] .- Float64.(sc.data.implied_vol[sc_mask])).^2)
    end
end
combined_rmse = sqrt(mean(combined_residuals_sq)) * 100

# Sector-only overall (for comparison)
sector_residuals_sq = Float64[]
for sector in sectors
    sc = sector_data_cache[sector]
    append!(sector_residuals_sq, (sc.model_ivs .- Float64.(sc.data.implied_vol)).^2)
end
sector_only_rmse = sqrt(mean(sector_residuals_sq)) * 100

println("\n  Overall in-sample RMSE:")
@printf("    Sector NN only:                           %5.2f%%\n", sector_only_rmse)
@printf("    Per-ticker NN (qualified) + sector NN:    %5.2f%%  (delta %+.2f)\n",
        combined_rmse, sector_only_rmse - combined_rmse)
@printf("    Qualified coverage: %d of %d tickers (%.0f%% of obs)\n",
        length(qualified), length(tickers),
        100 * sum(r.n_obs for r in eachrow(cmp_df) if r.qualified) / nrow(all_data))

# ============================================================================
# Step 7: Figures
# ============================================================================

println("\nGenerating figures...")

sector_colors = Dict("Tech" => :blue, "Financials" => :red, "Energy" => :green,
                     "Healthcare" => :purple, "Retail" => :orange, "ETF" => :black)

# --- Figure 1: per-ticker improvement bar chart ---
q_sorted_by_delta = sort(filter(r -> r.qualified, cmp_df), :improvement, rev=true)
tickers_plot = [r.ticker for r in eachrow(q_sorted_by_delta)]
delta_plot = [r.improvement for r in eachrow(q_sorted_by_delta)]
colors_plot = [get(sector_colors, r.sector, :gray) for r in eachrow(q_sorted_by_delta)]

p1 = bar(tickers_plot, delta_plot,
         title="Per-Ticker NN Improvement Over Sector NN (in-sample RMSE reduction)",
         xlabel="Ticker", ylabel="Delta RMSE (% IV), positive = per-ticker better",
         legend=false, size=(1000, 450), dpi=150,
         color=colors_plot, alpha=0.8, xrotation=45)
hline!(p1, [0.0], color=:black, ls=:dash, label=nothing, alpha=0.5)
for (sector, c) in sort(collect(sector_colors))
    bar!(p1, [], [], label=sector, color=c)
end
plot!(p1, legend=:topright)
savefig(p1, joinpath(PLOT_DIR, "ladder_per_ticker_nn_improvement.png"))
savefig(p1, joinpath(PLOT_DIR, "ladder_per_ticker_nn_improvement.pdf"))
println("  -> saved ladder_per_ticker_nn_improvement.png/pdf")

# --- Figure 2: smile-fit panels for representative qualified tickers ---
function eval_per_ticker_iv_grid(ticker::String, m_range, dte_val::Float64)
    pm = per_ticker_models[ticker]
    psi_nn = pm.model.psi_nn
    log_theta = pm.model.log_theta[1]
    log_dte_s = Float32((log(max(dte_val, 1.0)) - MU_DTE) / SIGMA_DTE)
    log_m_vals = [Float32((log(m) - MU_M) / SIGMA_M) for m in m_range]
    x_grid = hcat(fill(log_dte_s, length(m_range)), log_m_vals)' |> Matrix{Float32}
    log_psi = vec(psi_nn(x_grid))
    ivs = exp.(Float32(0.5) .* (Float32(log_theta) .+ log_psi))
    return Float64.(ivs) .* 100
end

function eval_sector_iv_grid(ticker::String, m_range, dte_val::Float64)
    sector = get(SECTORS, ticker, "Other")
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

# Pick a spread of qualified tickers across sectors for the panel
panel_candidates = ["SPY", "NVDA", "MSFT", "LLY", "GS", "AVGO"]
panel_tickers = [t for t in panel_candidates if t in Set(qualified)]

p_panels = []
for t in panel_tickers
    slice = all_data[all_data.ticker .== t, :]
    avail_dtes = sort(unique(slice.actual_dte))
    target_dte = avail_dtes[max(1, length(avail_dtes) ÷ 2)]
    dte_slice = slice[slice.actual_dte .== target_dte, :]
    sector = get(SECTORS, t, "Other")

    p = plot(title="$t [$sector] DTE=$target_dte",
             xlabel="K/S", ylabel="IV (%)", legend=:topright, titlefontsize=9)

    calls = dte_slice[dte_slice.type .== "call", :]
    puts = dte_slice[dte_slice.type .== "put", :]
    scatter!(p, calls.moneyness, Float64.(calls.implied_vol) .* 100,
             label="Calls", marker=:circle, ms=3, color=:blue, alpha=0.5)
    scatter!(p, puts.moneyness, Float64.(puts.implied_vol) .* 100,
             label="Puts", marker=:diamond, ms=3, color=:red, alpha=0.5)

    m_range = range(0.85, 1.15, length=80)
    sec_curve = eval_sector_iv_grid(t, m_range, Float64(target_dte))
    pt_curve = eval_per_ticker_iv_grid(t, m_range, Float64(target_dte))
    plot!(p, m_range, sec_curve, label="Sector NN", lw=2, color=:gray, ls=:dash)
    plot!(p, m_range, pt_curve, label="Per-ticker NN", lw=2, color=:black)
    vline!(p, [1.0], label=nothing, ls=:dash, color=:gray, alpha=0.3)
    push!(p_panels, p)
end
p2 = plot(p_panels..., layout=(2, 3), size=(1200, 700), dpi=150,
          plot_title="IV Smile Fit — Sector NN vs Per-Ticker NN")
savefig(p2, joinpath(PLOT_DIR, "ladder_per_ticker_nn_smile_panels.png"))
savefig(p2, joinpath(PLOT_DIR, "ladder_per_ticker_nn_smile_panels.pdf"))
println("  -> saved ladder_per_ticker_nn_smile_panels.png/pdf")

# --- Figure 3: residual distribution, sector vs per-ticker on qualified tickers ---
sec_resid_pooled = Float64[]
pt_resid_pooled = Float64[]
for t in qualified
    mask = all_data.ticker .== t
    target = Float64.(all_data.implied_vol[mask])
    sector = get(SECTORS, t, "Other")
    sc = sector_data_cache[sector]
    sc_mask = sc.data.ticker .== t
    append!(sec_resid_pooled, (sc.model_ivs[sc_mask] .- target) .* 100)
    append!(pt_resid_pooled, (per_ticker_preds[t] .- target) .* 100)
end

p3 = histogram(sec_resid_pooled, bins=-15:0.5:15, alpha=0.5, label="Sector NN",
               color=:gray, normalize=:pdf, size=(900, 450), dpi=150,
               title="Residual Distribution — Qualified Tickers",
               xlabel="Residual (% IV)", ylabel="Density")
histogram!(p3, pt_resid_pooled, bins=-15:0.5:15, alpha=0.5, label="Per-ticker NN",
           color=:steelblue, normalize=:pdf)
vline!(p3, [0.0], color=:black, ls=:dash, label=nothing)
savefig(p3, joinpath(PLOT_DIR, "ladder_per_ticker_nn_residuals.png"))
savefig(p3, joinpath(PLOT_DIR, "ladder_per_ticker_nn_residuals.pdf"))
println("  -> saved ladder_per_ticker_nn_residuals.png/pdf")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^70)
println("  SUMMARY")
println("="^70)
@printf("  Corpus: %d obs, %d tickers (%d qualified for per-ticker NN)\n",
        nrow(all_data), length(tickers), length(qualified))
@printf("  Sector NN overall:                        %5.2f%% IV\n", sector_only_rmse)
@printf("  Per-ticker (qualified) + sector NN combo: %5.2f%% IV  (%.2f improvement)\n",
        combined_rmse, sector_only_rmse - combined_rmse)
improved = count(r -> r.qualified && r.improvement > 0, eachrow(cmp_df))
regressed = count(r -> r.qualified && r.improvement < 0, eachrow(cmp_df))
@printf("  Tickers improved over sector NN: %d  |  regressed: %d\n", improved, regressed)

include(joinpath(@__DIR__, "..", "scripts", "promote_figures.jl"))
promote_figures()
