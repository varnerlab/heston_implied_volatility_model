"""
Trigger ROC analysis for the LoRA daily refit.

For every (ticker, date) cell in the 31×15 corpus, we already have the base
ψ-vs-market RMSE from the L1 sweep scan. We treat each cell as a binary
classification: does it need a refit (base RMSE above the "refit-needed"
threshold) or not?

We then evaluate a streaming trigger that sees dates chronologically and
must decide (per ticker) whether to refit today based on:
- absolute threshold:        today's base RMSE > τ_abs
- relative-drift threshold:  z-score of today's RMSE over the trailing-4
  non-flagged dates > τ_z, computed via robust median + 1.4826 × MAD

Flagged dates are excluded from the trailing window for future decisions
(the "contamination fix" identified by the LLY 04-28 prototype, where the
04-23 INTC blowup contaminated the rolling baseline and hid 04-28's drift).

We sweep τ_z from 0 to 6 with τ_abs fixed at 10% IV (a practical floor)
and plot TPR vs FPR.

Output:
  code/figures/lora_trigger_roc.csv     — per-cell trigger decisions
  code/figures/lora_trigger_roc.pdf     — ROC + per-ticker timelines

Run:
    julia --project=. examples/lora_trigger_roc.jl
"""

using CSV
using DataFrames
using Dates
using Flux
using JLD2
using Plots
using Plots.PlotMeasures
using Printf
using Statistics

using HestonIV
include(joinpath(@__DIR__, "..", "src", "ScenarioTemplate.jl"))
using .ScenarioTemplate

const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladder")
const NN_CACHE   = joinpath(@__DIR__, "..", "figures",
                            "calibrate_ladders_per_ticker_nn_cache.jld2")
const FIG_DIR    = joinpath(@__DIR__, "..", "figures")

const ALL_DAYS = [
    "options-04-14-2026", "options-04-15-2026", "options-04-16-2026",
    "options-04-17-2026", "options-04-21-2026", "options-04-22-2026",
    "options-04-23-2026", "options-04-24-2026", "options-04-27-2026",
    "options-04-28-2026", "options-04-29-2026", "options-05-01-2026",
    "options-05-06-2026", "options-05-08-2026", "options-05-11-2026",
]

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

# Ground truth: a cell "needs refit" if base RMSE exceeds this
const REFIT_THRESHOLD = 0.12     # 12% IV (~1.5× the corpus median 8.8%)
const TAU_ABS         = 0.10     # absolute floor for the streaming trigger
const TRAIL_LEN       = 4        # trailing window length

# ============================================================================
# Reload base-ψ RMSE per (ticker, date) — replays the L1 scan exactly
# ============================================================================

function _load_ladder(filepath)
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
        (df.actual_dte .> 0), :]
    return valid
end

function _load_all_ladders(dir)
    files = String[]
    for (root, _, fs) in walkdir(dir)
        occursin("VIX-data", root) && continue
        for f in fs
            endswith(f, ".csv") && occursin("_dte_ladder_", f) &&
                push!(files, joinpath(root, f))
        end
    end
    frames = DataFrame[]
    for f in files
        df = _load_ladder(f)
        nrow(df) > 0 && push!(frames, df)
    end
    return vcat(frames...)
end

println("Loading ladder corpus...")
all_data = _load_all_ladders(LADDER_DIR)
const MU_DTE    = mean(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const SIGMA_DTE = std(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const MU_M      = mean(log.(Float64.(all_data.moneyness)))
const SIGMA_M   = std(log.(Float64.(all_data.moneyness)))

function load_slice(ticker, date_dir)
    full = joinpath(LADDER_DIR, date_dir)
    isdir(full) || return DataFrame()
    files = filter(f -> startswith(f, ticker * "_") && endswith(f, ".csv") &&
                        occursin("_dte_ladder_", f), readdir(full))
    isempty(files) && return DataFrame()
    return _load_ladder(joinpath(full, files[1]))
end

println("Restoring base ψ_NN per ticker...")
nn_cache = JLD2.load(NN_CACHE)

per_ticker_base = Dict{String, Any}()
for ticker in sort(collect(keys(SECTORS)))
    sector = SECTORS[ticker]
    psi, log_theta, src = ScenarioTemplate._restore_nn(nn_cache, ticker;
        use_per_ticker=true, sector=sector)
    per_ticker_base[ticker] = (psi=psi, log_theta=log_theta, source=src)
end

println("Computing base RMSE for every (ticker, date) cell...")
cells = NamedTuple[]
for ticker in sort(collect(keys(SECTORS)))
    b = per_ticker_base[ticker]
    for d in ALL_DAYS
        slice = load_slice(ticker, d)
        nrow(slice) < 10 && continue
        z_dte = Float32.((log.(max.(Float64.(slice.actual_dte), 1.0)) .- MU_DTE) ./ SIGMA_DTE)
        z_m   = Float32.((log.(Float64.(slice.moneyness)) .- MU_M) ./ SIGMA_M)
        X = vcat(z_dte', z_m')
        # base IV
        h1 = tanh.(b.psi.layers[1].weight * X .+ b.psi.layers[1].bias)
        h2 = tanh.(b.psi.layers[2].weight * h1 .+ b.psi.layers[2].bias)
        log_psi = vec(b.psi.layers[3].weight * h2 .+ b.psi.layers[3].bias)
        σ = exp.(Float32(0.5) .* (Float32(b.log_theta) .+ log_psi))
        rmse = sqrt(mean((Float64.(σ) .- Float64.(slice.implied_vol)).^2))
        push!(cells, (ticker=ticker, date=d, sector=SECTORS[ticker],
                      psi_source=String(b.source), n_obs=nrow(slice),
                      base_rmse=rmse))
    end
end
cells_df = DataFrame(cells)
@printf("  %d cells scanned; %d above the %.0f%% refit-needed threshold\n",
        nrow(cells_df), sum(cells_df.base_rmse .> REFIT_THRESHOLD),
        100*REFIT_THRESHOLD)

# ============================================================================
# Streaming per-ticker trigger with flagged-day exclusion
# ============================================================================

"""
    streaming_trigger(rmse_seq, τ_abs, τ_z; trail_len=4)

Process the per-ticker chronological RMSE sequence and flag dates where:
- today's RMSE > τ_abs (absolute floor), AND
- today's z-score over the trailing TRAIL_LEN non-flagged dates > τ_z,
  with z computed as (rmse - median(trail)) / max(1.4826·MAD(trail), 1e-4).
Flagged dates are excluded from the trailing baseline used for future decisions.
"""
function streaming_trigger(rmse_seq::Vector{Float64}, τ_abs::Float64, τ_z::Float64;
                           trail_len::Int=TRAIL_LEN)
    n = length(rmse_seq)
    flags = falses(n)
    z_scores = fill(NaN, n)
    trail = Float64[]
    for i in 1:n
        r_today = rmse_seq[i]
        if length(trail) >= trail_len
            m = median(trail[end-trail_len+1:end])
            mad_v = median(abs.(trail[end-trail_len+1:end] .- m))
            scaled = max(1.4826 * mad_v, 1e-4)
            z = (r_today - m) / scaled
            z_scores[i] = z
            if r_today > τ_abs && z > τ_z
                flags[i] = true
                # don't push into trail (contamination fix)
                continue
            end
        end
        push!(trail, r_today)
    end
    return flags, z_scores
end

# ============================================================================
# ROC sweep over τ_z (τ_abs fixed)
# ============================================================================

println("Sweeping τ_z thresholds for ROC...")

# Build per-ticker chronological sequences
tickers = sort(unique(cells_df.ticker))
ticker_dates = Dict{String, Vector{String}}()
ticker_rmse  = Dict{String, Vector{Float64}}()
ticker_truth = Dict{String, Vector{Bool}}()
for t in tickers
    sub = sort(cells_df[cells_df.ticker .== t, :], :date)
    ticker_dates[t] = String.(sub.date)
    ticker_rmse[t]  = collect(Float64.(sub.base_rmse))
    ticker_truth[t] = collect(sub.base_rmse .> REFIT_THRESHOLD)
end

τ_zs = collect(0.0:0.25:6.0)
tpr_arr = Float64[]
fpr_arr = Float64[]
prec_arr = Float64[]

for τ_z in τ_zs
    tp, fp, tn, fn = 0, 0, 0, 0
    for t in tickers
        flags, _ = streaming_trigger(ticker_rmse[t], TAU_ABS, τ_z)
        truth = ticker_truth[t]
        for i in 1:length(flags)
            if truth[i] && flags[i]
                tp += 1
            elseif truth[i] && !flags[i]
                fn += 1
            elseif !truth[i] && flags[i]
                fp += 1
            else
                tn += 1
            end
        end
    end
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    push!(tpr_arr, tpr); push!(fpr_arr, fpr); push!(prec_arr, prec)
end

roc_df = DataFrame(tau_z=τ_zs, tpr=tpr_arr, fpr=fpr_arr, precision=prec_arr)
@printf("\nROC sweep (τ_abs = %.0f%% IV fixed, τ_z varied):\n", 100*TAU_ABS)
@printf("  τ_z     TPR     FPR     Precision\n")
@printf("  -----   -----   -----   ---------\n")
for r in eachrow(roc_df)
    @printf("  %5.2f   %5.3f   %5.3f      %5.3f\n", r.tau_z, r.tpr, r.fpr, r.precision)
end

function compute_auroc(fpr_arr, tpr_arr)
    idx = sortperm(fpr_arr)
    fpr_s = fpr_arr[idx]
    tpr_s = tpr_arr[idx]
    auc = 0.0
    for i in 2:length(fpr_s)
        auc += 0.5 * (tpr_s[i] + tpr_s[i-1]) * (fpr_s[i] - fpr_s[i-1])
    end
    return auc
end
auroc = compute_auroc(fpr_arr, tpr_arr)
@printf("\nAUROC = %.4f   (TAU_ABS = %.0f%% IV)\n", auroc, 100*TAU_ABS)

# Optimal operating point: maximize Youden's J = TPR - FPR
J = tpr_arr .- fpr_arr
opt_i = argmax(J)
@printf("Youden-optimal τ_z = %.2f   →   TPR = %.3f   FPR = %.3f   Precision = %.3f\n",
        τ_zs[opt_i], tpr_arr[opt_i], fpr_arr[opt_i], prec_arr[opt_i])

# ============================================================================
# Persist + plots
# ============================================================================

mkpath(FIG_DIR)
CSV.write(joinpath(FIG_DIR, "lora_trigger_roc_sweep.csv"), roc_df)

# Per-cell streaming decisions at the Youden-optimal τ_z
opt_τ_z = τ_zs[opt_i]
flag_rows = NamedTuple[]
for t in tickers
    flags, zs = streaming_trigger(ticker_rmse[t], TAU_ABS, opt_τ_z)
    for i in 1:length(flags)
        truth = ticker_rmse[t][i] > REFIT_THRESHOLD
        push!(flag_rows, (
            ticker=t, date=ticker_dates[t][i],
            base_rmse=ticker_rmse[t][i],
            z_score=zs[i],
            truth_needs_refit=truth,
            flag_triggered=flags[i],
            correct = (truth == flags[i]),
        ))
    end
end
flag_df = DataFrame(flag_rows)
CSV.write(joinpath(FIG_DIR, "lora_trigger_roc_cells.csv"), flag_df)
@printf("\n[csv] wrote -> %s\n", joinpath(FIG_DIR, "lora_trigger_roc_sweep.csv"))
@printf("[csv] wrote -> %s\n", joinpath(FIG_DIR, "lora_trigger_roc_cells.csv"))

# Figure: ROC + per-ticker timeline summary
println("Rendering ROC figure...")
p_roc = plot(fpr_arr, tpr_arr;
             marker=:circle, ms=4, lw=2,
             xlabel="False positive rate",
             ylabel="True positive rate",
             title=@sprintf("Trigger ROC (τ_abs = %.0f%% IV; AUROC = %.3f)",
                            100*TAU_ABS, auroc),
             titlefontsize=11, label="MAD trigger (flagged-day exclusion)",
             color=RGB(0.10, 0.35, 0.65),
             framestyle=:box, grid=true, gridalpha=0.25,
             legend=:bottomright, xlims=(0, 1), ylims=(0, 1),
             aspect_ratio=:equal)
plot!(p_roc, [0, 1], [0, 1]; color=:gray, ls=:dash, alpha=0.6, label="chance")
scatter!(p_roc, [fpr_arr[opt_i]], [tpr_arr[opt_i]];
         color=:red, ms=8, label=@sprintf("Youden-optimal τ_z = %.2f", opt_τ_z))

# Per-ticker timeline: for the 6 most-flagged tickers
flag_per_ticker = combine(groupby(flag_df, :ticker), :flag_triggered => sum => :n_flags)
sort!(flag_per_ticker, :n_flags, rev=true)
top_tickers = String.(flag_per_ticker.ticker[1:min(6, nrow(flag_per_ticker))])

p_time = plot(layout=(2, 3), size=(1400, 700), dpi=180,
              left_margin=8mm, right_margin=4mm,
              top_margin=4mm, bottom_margin=6mm)
for (j, t) in enumerate(top_tickers)
    sub = flag_df[flag_df.ticker .== t, :]
    if nrow(sub) == 0; continue; end
    dates_idx = 1:nrow(sub)
    truth_pts = findall(sub.truth_needs_refit)
    flag_pts  = findall(sub.flag_triggered)
    plot!(p_time[j], dates_idx, 100 .* sub.base_rmse;
          color=:black, lw=2, label="base RMSE",
          xlabel="fold index (chronological)", ylabel="RMSE (% IV)",
          title=String(t), titlefontsize=10, legend=false,
          framestyle=:box, grid=true, gridalpha=0.25)
    hline!(p_time[j], [100*REFIT_THRESHOLD]; color=:gray, ls=:dash, alpha=0.5, label="")
    if !isempty(truth_pts)
        scatter!(p_time[j], truth_pts, 100 .* sub.base_rmse[truth_pts];
                 color=RGB(0.78, 0.20, 0.20), ms=8, markerstrokewidth=0,
                 label="truth (refit needed)")
    end
    if !isempty(flag_pts)
        scatter!(p_time[j], flag_pts, 100 .* sub.base_rmse[flag_pts];
                 color=RGB(0.20, 0.55, 0.30), ms=4, markershape=:cross,
                 markerstrokewidth=2, label="trigger fired")
    end
end

p_grid = plot(p_roc, p_time, layout=grid(1, 2, widths=[0.36, 0.64]),
              size=(2100, 700), dpi=180)
fig_pdf = joinpath(FIG_DIR, "lora_trigger_roc.pdf")
fig_png = joinpath(FIG_DIR, "lora_trigger_roc.png")
savefig(p_grid, fig_pdf); savefig(p_grid, fig_png)
@printf("[fig] wrote -> %s / .png\n", fig_pdf)

println("\nDone.")
