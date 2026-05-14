"""
No-arbitrage audit on the LoRA-adapted surfaces.

For each of the 20 (ticker, date) cells in `lora_sweep_adapters.jld2`,
re-run the B1 static-arbitrage check on both the base ψ_NN surface and the
V1-adapted surface. Compare butterfly and calendar violation rates to
answer: does the rank-2 daily adapter amplify arbitrage pathologies?

Grid: 20×20, K/S in [0.80, 1.20] log-spaced, DTE in [7, 365] log-spaced.
Pricer: CRR European at 200 steps.

Output:
  code/figures/lora_arbitrage_audit.csv
  code/figures/lora_arbitrage_audit.pdf  (base vs adapted scatter)

Run:
    julia --project=. examples/lora_arbitrage_audit.jl
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

const LADDER_DIR    = joinpath(@__DIR__, "..", "data", "ladder")
const NN_CACHE      = joinpath(@__DIR__, "..", "figures",
                               "calibrate_ladders_per_ticker_nn_cache.jld2")
const ADAPTER_CACHE = joinpath(@__DIR__, "..", "figures",
                               "lora_sweep_adapters.jld2")
const CELLS_CSV     = joinpath(@__DIR__, "..", "figures", "lora_sweep_cells.csv")
const FIG_DIR       = joinpath(@__DIR__, "..", "figures")

const N_K = 20
const N_T = 20
const K_OVER_S_LO = 0.80
const K_OVER_S_HI = 1.20
const DTE_LO = 7
const DTE_HI = 365
const R_FREE = 0.0425
const N_STEPS_CRR = 200

const RANK   = 2
const ALPHA  = 2.0
const SCALE  = Float32(ALPHA / RANK)

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
# Reload ladder corpus for standardization + anchor spots
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

function _dir_to_date(d)
    m = match(r"options-(\d{2})-(\d{2})-(\d{4})", d)
    Date(parse(Int, m.captures[3]), parse(Int, m.captures[1]), parse(Int, m.captures[2]))
end

function anchor_spot(ticker, date_str)
    target = _dir_to_date(date_str)
    slice = all_data[(all_data.ticker .== ticker) .& (all_data.und_session_date .== target), :]
    if isempty(slice)
        # Some captures roll back a session; try the immediately prior date in the corpus
        prior = sort(unique(all_data.und_session_date[all_data.ticker .== ticker]))
        idx = searchsortedlast(prior, target)
        idx == 0 && error("no $(ticker) rows near $(target)")
        return Float64(all_data[(all_data.ticker .== ticker) .& (all_data.und_session_date .== prior[idx]), :S][1])
    end
    return Float64(slice.S[1])
end

# ============================================================================
# Restore base + adapter for each cell
# ============================================================================

println("Loading NN cache + adapters...")
nn_cache = JLD2.load(NN_CACHE)
adapters_raw = JLD2.load(ADAPTER_CACHE)["adapters"]
cells_df = CSV.read(CELLS_CSV, DataFrame)

# ============================================================================
# Grid + violation tests (same as B1)
# ============================================================================

# Base ψ forward pass (handles 2 → h1 → h2 → 1)
function base_log_psi_pass(base_psi, X)
    h1 = tanh.(base_psi.layers[1].weight * X .+ base_psi.layers[1].bias)
    h2 = tanh.(base_psi.layers[2].weight * h1 .+ base_psi.layers[2].bias)
    return vec(base_psi.layers[3].weight * h2 .+ base_psi.layers[3].bias)
end

# LoRA forward pass with rank-2 deltas
function lora_log_psi_pass(base_psi, ap, X)
    h1 = tanh.(base_psi.layers[1].weight * X .+ base_psi.layers[1].bias .+
               SCALE .* (ap.B1 * (ap.A1 * X)))
    h2 = tanh.(base_psi.layers[2].weight * h1 .+ base_psi.layers[2].bias .+
               SCALE .* (ap.B2 * (ap.A2 * h1)))
    return vec(base_psi.layers[3].weight * h2 .+ base_psi.layers[3].bias .+
               SCALE .* (ap.B3 * (ap.A3 * h2)))
end

function evaluate_surface(iv_fn, S_0)
    k_over_s = exp.(range(log(K_OVER_S_LO), log(K_OVER_S_HI); length=N_K))
    dte_grid = exp.(range(log(DTE_LO), log(DTE_HI); length=N_T))
    K_grid = k_over_s .* S_0
    C = Array{Float64}(undef, N_K, N_T)
    for j in 1:N_T
        T_yrs = dte_grid[j] / 365.0
        for i in 1:N_K
            σ = iv_fn(K_grid[i], S_0, dte_grid[j])
            C[i, j] = crr_european_price(S_0, K_grid[i], σ, R_FREE, T_yrs, N_STEPS_CRR, :call)
        end
    end
    # Butterfly: ∂²C/∂K² via central second difference (non-uniform grid)
    bfly = fill(NaN, N_K, N_T)
    for j in 1:N_T
        for i in 2:(N_K - 1)
            Δup = K_grid[i+1] - K_grid[i]
            Δdn = K_grid[i]   - K_grid[i-1]
            bfly[i, j] = 2 * (Δdn * C[i+1, j] - (Δup + Δdn) * C[i, j] + Δup * C[i-1, j]) /
                        (Δup * Δdn * (Δup + Δdn))
        end
    end
    # Calendar: ∂C/∂T via forward first difference
    cal = fill(NaN, N_K, N_T)
    for i in 1:N_K
        for j in 1:(N_T - 1)
            cal[i, j] = (C[i, j+1] - C[i, j]) / (dte_grid[j+1] - dte_grid[j])
        end
    end
    return bfly, cal
end

function violation_counts(bf, cl)
    bf_valid = .!isnan.(bf); cl_valid = .!isnan.(cl)
    bf_viol = bf_valid .& (bf .< 0); cl_viol = cl_valid .& (cl .< 0)
    return (sum(bf_viol) / sum(bf_valid), sum(cl_viol) / sum(cl_valid),
            sum(bf_viol), sum(bf_valid), sum(cl_viol), sum(cl_valid))
end

# ============================================================================
# Run the audit
# ============================================================================

rows = NamedTuple[]
for (i, row) in enumerate(eachrow(cells_df))
    ticker = String(row.ticker)
    date_dir = String(row.date)
    sector = SECTORS[ticker]

    # Restore base ψ via _restore_nn (use_per_ticker first)
    base_psi, log_theta_base, _ = ScenarioTemplate._restore_nn(nn_cache, ticker;
        use_per_ticker=true, sector=sector)

    # Restore adapter
    key = "$(ticker)__$(date_dir)"
    a = adapters_raw[key]

    S_0 = anchor_spot(ticker, date_dir)

    # Build IV functions
    base_iv = function(K, S, dte)
        z_dte = Float32((log(max(Float64(dte), 1.0)) - MU_DTE) / SIGMA_DTE)
        z_m   = Float32((log(K / S) - MU_M) / SIGMA_M)
        x = reshape(Float32[z_dte, z_m], 2, 1)
        log_psi = first(base_log_psi_pass(base_psi, x))
        return sqrt(max(exp(log_theta_base) * exp(Float64(log_psi)), 1e-10))
    end
    adapted_iv = function(K, S, dte)
        z_dte = Float32((log(max(Float64(dte), 1.0)) - MU_DTE) / SIGMA_DTE)
        z_m   = Float32((log(K / S) - MU_M) / SIGMA_M)
        x = reshape(Float32[z_dte, z_m], 2, 1)
        log_psi = first(lora_log_psi_pass(base_psi, a, x))
        return sqrt(max(exp(Float64(a.log_theta[1])) * exp(Float64(log_psi)), 1e-10))
    end

    bf_b, cl_b = evaluate_surface(base_iv,    S_0)
    bf_a, cl_a = evaluate_surface(adapted_iv, S_0)
    fb_b, fc_b, nb_b, nB_b, nc_b, nC_b = violation_counts(bf_b, cl_b)
    fb_a, fc_a, nb_a, nB_a, nc_a, nC_a = violation_counts(bf_a, cl_a)

    @printf("  %2d/%-2d  %-5s %s  S=%-9.2f  bfly base=%.1f%%  bfly V1=%.1f%%  cal base=%.1f%%  cal V1=%.1f%%\n",
            i, nrow(cells_df), ticker, date_dir, S_0,
            100*fb_b, 100*fb_a, 100*fc_b, 100*fc_a)

    push!(rows, (
        ticker=ticker, date=date_dir, sector=sector,
        cell_kind=row.cell_kind, base_rmse=row.base_rmse, v1_rmse=row.v1_rmse,
        S_0=S_0,
        base_bfly_viol=nb_b, base_bfly_total=nB_b, base_bfly_frac=fb_b,
        adapted_bfly_viol=nb_a, adapted_bfly_total=nB_a, adapted_bfly_frac=fb_a,
        base_cal_viol=nc_b, base_cal_total=nC_b, base_cal_frac=fc_b,
        adapted_cal_viol=nc_a, adapted_cal_total=nC_a, adapted_cal_frac=fc_a,
    ))
end

df = DataFrame(rows)
out_csv = joinpath(FIG_DIR, "lora_arbitrage_audit.csv")
CSV.write(out_csv, df)
@printf("\n[csv] wrote -> %s\n", out_csv)

# Aggregate
println("\n" * "="^78)
println("  Aggregate butterfly violations:")
@printf("    Base    : %4d / %5d  (%.2f%%)\n",
        sum(df.base_bfly_viol),    sum(df.base_bfly_total),
        100*sum(df.base_bfly_viol)    / sum(df.base_bfly_total))
@printf("    Adapted : %4d / %5d  (%.2f%%)\n",
        sum(df.adapted_bfly_viol), sum(df.adapted_bfly_total),
        100*sum(df.adapted_bfly_viol) / sum(df.adapted_bfly_total))
println("  Aggregate calendar violations:")
@printf("    Base    : %4d / %5d  (%.2f%%)\n",
        sum(df.base_cal_viol),    sum(df.base_cal_total),
        100*sum(df.base_cal_viol)    / sum(df.base_cal_total))
@printf("    Adapted : %4d / %5d  (%.2f%%)\n",
        sum(df.adapted_cal_viol), sum(df.adapted_cal_total),
        100*sum(df.adapted_cal_viol) / sum(df.adapted_cal_total))

# Scatter plot
println("\nRendering audit figure...")
COL = Dict("worst" => RGB(0.78, 0.20, 0.20),
           "middle" => RGB(0.55, 0.55, 0.55),
           "best" => RGB(0.20, 0.55, 0.30))
colors = [COL[k] for k in df.cell_kind]

p1 = scatter(100 .* df.base_bfly_frac, 100 .* df.adapted_bfly_frac;
             color=colors, markersize=6, markerstrokewidth=0.4,
             xlabel="Base ψ butterfly violations (% grid pts)",
             ylabel="V1-adapted butterfly violations (% grid pts)",
             title="Butterfly arbitrage: adapted vs base",
             titlefontsize=11, legend=false,
             framestyle=:box, grid=true, gridalpha=0.25)
plot!(p1, x -> x, range(0, max(maximum(df.base_bfly_frac), maximum(df.adapted_bfly_frac)) * 100; length=10);
      color=:black, ls=:dash, alpha=0.5, label="")

p2 = scatter(100 .* df.base_cal_frac, 100 .* df.adapted_cal_frac;
             color=colors, markersize=6, markerstrokewidth=0.4,
             xlabel="Base ψ calendar violations (% grid pts)",
             ylabel="V1-adapted calendar violations (% grid pts)",
             title="Calendar arbitrage: adapted vs base",
             titlefontsize=11, legend=false,
             framestyle=:box, grid=true, gridalpha=0.25)
m_cal = max(maximum(df.base_cal_frac), maximum(df.adapted_cal_frac)) * 100 + 1
plot!(p2, x -> x, range(0, m_cal; length=10);
      color=:black, ls=:dash, alpha=0.5, label="")

p_grid = plot(p1, p2, layout=(1, 2), size=(1300, 580), dpi=200,
              left_margin=8mm, right_margin=4mm,
              top_margin=4mm, bottom_margin=7mm)
fig_pdf = joinpath(FIG_DIR, "lora_arbitrage_audit.pdf")
fig_png = joinpath(FIG_DIR, "lora_arbitrage_audit.png")
savefig(p_grid, fig_pdf); savefig(p_grid, fig_png)
@printf("[fig] wrote -> %s / .png\n", fig_pdf)

println("\nDone.")
