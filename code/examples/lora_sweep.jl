"""
LoRA-adapter sweep across (ticker, date) cells.

Generalizes the LLY 04-28 prototype to a 20-cell sweep: 5 highest-RMSE cells
(problem days), 5 lowest-RMSE cells (controls), and 10 middling cells.
For each cell, fits V0 (level-only) and V1 (LoRA-r2 + level) on the day's
ladder slice and reports RMSE before/after, adapter norms, and the post-adapt
log-θ shift.

Goal: show that V1's improvement scales with base-RMSE (large gains on
problem days, near-zero on controls), and that the rank-2 budget is enough
across ticker-sector pairings beyond LLY.

Output:
  code/figures/lora_sweep_cells.csv     — full per-cell metrics
  code/figures/lora_sweep_summary.pdf   — 4-panel diagnostic plot

Run:
    julia --project=. examples/lora_sweep.jl
"""

using CSV
using DataFrames
using Dates
using Flux
using JLD2
using Plots
using Plots.PlotMeasures
using Printf
using Random
using Statistics

using HestonIV
include(joinpath(@__DIR__, "..", "src", "ScenarioTemplate.jl"))
using .ScenarioTemplate

# ============================================================================
# Configuration
# ============================================================================

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

const RANK     = 2
const ALPHA    = 2.0
const N_EPOCHS = 1500
const PATIENCE = 150
const SEED     = 20260514
const SCALE    = Float32(ALPHA / RANK)

# Cell selection
const N_WORST  = 5
const N_BEST   = 5
const N_MID    = 10

# ============================================================================
# Ladder loading + standardization
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

println("Loading 31-ticker ladder corpus...")
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

function build_Xy(slice::DataFrame)
    z_dte = Float32.((log.(max.(Float64.(slice.actual_dte), 1.0)) .- MU_DTE) ./ SIGMA_DTE)
    z_m   = Float32.((log.(Float64.(slice.moneyness)) .- MU_M) ./ SIGMA_M)
    X = vcat(z_dte', z_m')
    y = Float32.(slice.implied_vol)
    return X, y
end

# ============================================================================
# Base ψ_NN per ticker
# ============================================================================

println("Loading NN cache...")
nn_cache = JLD2.load(NN_CACHE)

# Precompute per-ticker base ψ + log_theta. For tickers without a per_ticker
# payload (PFE, BMY), fall back to the sector network.
struct BasePsi
    layers::Vector{Any}      # 3 Flux Dense layers
    log_theta::Float64
    source::Symbol
end

const BASE_PSI = Dict{String,BasePsi}()
for ticker in sort(collect(keys(SECTORS)))
    sector = SECTORS[ticker]
    psi, log_theta, src = ScenarioTemplate._restore_nn(nn_cache, ticker;
        use_per_ticker=true, sector=sector)
    BASE_PSI[ticker] = BasePsi([psi.layers[1], psi.layers[2], psi.layers[3]],
                               log_theta, src)
end
@printf("Base ψ restored for %d tickers (%d per-ticker, %d sector fallback)\n",
        length(BASE_PSI),
        sum(b -> b.source === :per_ticker, values(BASE_PSI)),
        sum(b -> b.source === :sector,     values(BASE_PSI)))

function base_log_psi(b::BasePsi, X)
    h1 = tanh.(b.layers[1].weight * X .+ b.layers[1].bias)
    h2 = tanh.(b.layers[2].weight * h1 .+ b.layers[2].bias)
    return vec(b.layers[3].weight * h2 .+ b.layers[3].bias)
end

function base_iv(b::BasePsi, X)
    return exp.(Float32(0.5) .* (Float32(b.log_theta) .+ base_log_psi(b, X)))
end

# ============================================================================
# Step 1: scan base RMSE across all 31×15 cells
# ============================================================================

println("\nScanning base ψ RMSE across all (ticker, date) cells...")
rows = NamedTuple[]
for ticker in sort(collect(keys(SECTORS)))
    b = BASE_PSI[ticker]
    for d in ALL_DAYS
        slice = load_slice(ticker, d)
        nrow(slice) < 10 && continue
        X, y = build_Xy(slice)
        σ = Float64.(base_iv(b, X))
        rmse = sqrt(mean((σ .- Float64.(y)).^2))
        push!(rows, (
            ticker=ticker, date=d, sector=SECTORS[ticker],
            psi_source=String(b.source),
            n_obs=nrow(slice), base_rmse=rmse,
        ))
    end
end
scan_df = DataFrame(rows)
@printf("  scanned %d cells; base RMSE quartiles: %.2f%% / %.2f%% / %.2f%%\n",
        nrow(scan_df),
        100*quantile(scan_df.base_rmse, 0.25),
        100*quantile(scan_df.base_rmse, 0.50),
        100*quantile(scan_df.base_rmse, 0.75))

# Pick 5 worst, 5 best, 10 middling (deterministic via sort)
sort!(scan_df, :base_rmse, rev=true)
worst_cells = scan_df[1:N_WORST, :]
best_cells  = scan_df[(end - N_BEST + 1):end, :]
# Middle: stride through the middle-50% band
mid_lo = nrow(scan_df) ÷ 4
mid_hi = 3 * (nrow(scan_df) ÷ 4)
mid_indices = round.(Int, range(mid_lo, mid_hi; length=N_MID))
mid_cells = scan_df[mid_indices, :]

chosen = vcat(worst_cells, mid_cells, best_cells)
chosen[!, :cell_kind] = vcat(fill("worst", N_WORST), fill("middle", N_MID),
                             fill("best",  N_BEST))

println("\nSelected cells:")
println("  rank   ticker date         sector       n_obs   base_rmse(%)  kind")
println("  " * "-"^70)
for (i, row) in enumerate(eachrow(chosen))
    @printf("  %-4d   %-5s  %s   %-11s  %5d   %6.2f       %s\n",
            i, row.ticker, row.date, row.sector, row.n_obs,
            100*row.base_rmse, row.cell_kind)
end

# ============================================================================
# Step 2: train V0 and V1 on each chosen cell
# ============================================================================

mutable struct LoRAAdapter
    A1::Matrix{Float32};  B1::Matrix{Float32}
    A2::Matrix{Float32};  B2::Matrix{Float32}
    A3::Matrix{Float32};  B3::Matrix{Float32}
    log_theta::Vector{Float32}
end
Flux.@layer LoRAAdapter

function make_adapter(b::BasePsi, rng, log_theta_init::Real)
    h1 = size(b.layers[1].weight, 1)   # output dim of Dense 1 (hidden)
    h2 = size(b.layers[2].weight, 1)   # output dim of Dense 2 (hidden)
    init_A(d_in)  = Float32.(randn(rng, RANK, d_in) ./ Float32(sqrt(d_in)))
    init_B(d_out) = zeros(Float32, d_out, RANK)
    return LoRAAdapter(
        init_A(2),  init_B(h1),
        init_A(h1), init_B(h2),
        init_A(h2), init_B(1),
        Float32[log_theta_init],
    )
end

function lora_log_psi(b::BasePsi, a::LoRAAdapter, X)
    h1 = tanh.(b.layers[1].weight * X .+ b.layers[1].bias .+
               SCALE .* (a.B1 * (a.A1 * X)))
    h2 = tanh.(b.layers[2].weight * h1 .+ b.layers[2].bias .+
               SCALE .* (a.B2 * (a.A2 * h1)))
    return vec(b.layers[3].weight * h2 .+ b.layers[3].bias .+
               SCALE .* (a.B3 * (a.A3 * h2)))
end

function model_iv_v1(b::BasePsi, a::LoRAAdapter, X)
    return exp.(Float32(0.5) .* (a.log_theta[1] .+ lora_log_psi(b, a, X)))
end

function model_iv_v0(b::BasePsi, params, X)
    return exp.(Float32(0.5) .* (params.log_theta[1] .+ base_log_psi(b, X)))
end

function train_v0(b::BasePsi, X, y; lr=5e-3, n_epochs=500, patience_max=50)
    Random.seed!(SEED)
    log_θ = Float32[b.log_theta]
    params = (log_theta = log_θ,)
    opt = Flux.setup(Flux.Adam(lr), params)
    best, bs, pat = Inf, params.log_theta[1], 0
    for _ in 1:n_epochs
        l, g = Flux.withgradient(params) do p
            Flux.mse(model_iv_v0(b, p, X), y)
        end
        Flux.update!(opt, params, g[1])
        if l < best; best = l; bs = params.log_theta[1]; pat = 0
        else; pat += 1; end
        pat >= patience_max && break
    end
    params.log_theta[1] = bs
    return params, best
end

function train_v1!(b::BasePsi, adapter::LoRAAdapter, X, y;
                   lr=1e-3, n_epochs=N_EPOCHS, patience_max=PATIENCE)
    opt = Flux.setup(Flux.Adam(lr), adapter)
    best, bs, pat = Inf, deepcopy(Flux.state(adapter)), 0
    for epoch in 1:n_epochs
        l, g = Flux.withgradient(adapter) do a
            Flux.mse(model_iv_v1(b, a, X), y)
        end
        Flux.update!(opt, adapter, g[1])
        if l < best
            best = l
            bs = deepcopy(Flux.state(adapter))
            pat = 0
        else
            pat += 1
        end
        pat >= patience_max && break
        epoch == 500  && Flux.adjust!(opt, 5e-4)
        epoch == 1000 && Flux.adjust!(opt, 2e-4)
    end
    Flux.loadmodel!(adapter, bs)
    return best
end

println("\nTraining V0 and V1 on $(nrow(chosen)) chosen cells...")

# Hold V1 adapters keyed by (ticker, date) so the no-arb sweep (L2) can reuse them.
ADAPTERS = Dict{Tuple{String,String}, LoRAAdapter}()

results = NamedTuple[]
for (i, row) in enumerate(eachrow(chosen))
    ticker, date_dir = row.ticker, row.date
    b = BASE_PSI[ticker]
    slice = load_slice(ticker, date_dir)
    X, y = build_Xy(slice)
    y_f64 = Float64.(y)
    σ_base = Float64.(base_iv(b, X))
    rmse_base = sqrt(mean((σ_base .- y_f64).^2))

    # V0
    p0, _ = train_v0(b, X, y)
    σ_v0 = Float64.(model_iv_v0(b, p0, X))
    rmse_v0 = sqrt(mean((σ_v0 .- y_f64).^2))
    δ_logθ_v0 = Float64(p0.log_theta[1] - Float32(b.log_theta))

    # V1
    adapter = make_adapter(b, MersenneTwister(SEED), b.log_theta)
    train_v1!(b, adapter, X, y)
    σ_v1 = Float64.(model_iv_v1(b, adapter, X))
    rmse_v1 = sqrt(mean((σ_v1 .- y_f64).^2))
    δ_logθ_v1 = Float64(adapter.log_theta[1] - Float32(b.log_theta))
    norm_B1A1 = sqrt(sum((adapter.B1 * adapter.A1).^2))
    norm_B2A2 = sqrt(sum((adapter.B2 * adapter.A2).^2))
    norm_B3A3 = sqrt(sum((adapter.B3 * adapter.A3).^2))
    ADAPTERS[(ticker, date_dir)] = deepcopy(adapter)

    @printf("  %2d/%-2d  %-5s %s  base=%5.2f%%  V0=%5.2f%% (δθ=%+.2f)  V1=%5.2f%% (δθ=%+.2f, ‖BA‖=%.2f/%.2f/%.2f)\n",
            i, nrow(chosen), ticker, date_dir,
            100*rmse_base, 100*rmse_v0, δ_logθ_v0,
            100*rmse_v1, δ_logθ_v1, norm_B1A1, norm_B2A2, norm_B3A3)

    push!(results, (
        ticker=ticker, date=date_dir, sector=row.sector,
        psi_source=String(b.source), n_obs=row.n_obs, cell_kind=row.cell_kind,
        base_rmse=rmse_base, v0_rmse=rmse_v0, v1_rmse=rmse_v1,
        v0_improvement=rmse_base - rmse_v0,
        v1_improvement=rmse_base - rmse_v1,
        v0_delta_log_theta=δ_logθ_v0,
        v1_delta_log_theta=δ_logθ_v1,
        v1_norm_B1A1=norm_B1A1,
        v1_norm_B2A2=norm_B2A2,
        v1_norm_B3A3=norm_B3A3,
    ))
end

results_df = DataFrame(results)

# ============================================================================
# Step 3: Persist + plot
# ============================================================================

mkpath(FIG_DIR)
out_csv = joinpath(FIG_DIR, "lora_sweep_cells.csv")
CSV.write(out_csv, results_df)
@printf("\n[csv] wrote -> %s\n", out_csv)

# Save the adapters too so L2 (no-arb on adapters) can reuse them
adapter_save = Dict{String, Any}()
for ((ticker, date), a) in ADAPTERS
    key = "$(ticker)__$(date)"
    adapter_save[key] = (
        A1=a.A1, B1=a.B1, A2=a.A2, B2=a.B2, A3=a.A3, B3=a.B3,
        log_theta=a.log_theta,
    )
end
JLD2.jldsave(joinpath(FIG_DIR, "lora_sweep_adapters.jld2"); adapters=adapter_save)
@printf("[jld2] wrote -> %s\n", joinpath(FIG_DIR, "lora_sweep_adapters.jld2"))

# Diagnostic figure: 2x2
println("Rendering diagnostic figure...")
COL_WORST  = RGB(0.78, 0.20, 0.20)
COL_MID    = RGB(0.55, 0.55, 0.55)
COL_BEST   = RGB(0.20, 0.55, 0.30)
function cell_color(kind)
    kind == "worst" ? COL_WORST :
    kind == "best"  ? COL_BEST  : COL_MID
end

p1 = scatter(100 .* results_df.base_rmse,
             100 .* results_df.v1_improvement,
             color=cell_color.(results_df.cell_kind),
             markersize=6, markerstrokewidth=0.4,
             xlabel="Base ψ RMSE (% IV)",
             ylabel="V1 improvement: base − V1 (% IV)",
             title="LoRA improvement scales with base RMSE",
             titlefontsize=11, legend=:topleft,
             framestyle=:box, grid=true, gridalpha=0.25,
             label="cell (red=worst, gray=mid, green=best)")
plot!(p1, x -> x, range(0, 25; length=10);
      color=:black, ls=:dash, alpha=0.5,
      label="y = x  (full closure of base RMSE)")
hline!(p1, [0]; color=:black, alpha=0.4, label="")

p2 = scatter(100 .* results_df.base_rmse,
             results_df.v1_norm_B2A2,
             color=cell_color.(results_df.cell_kind),
             markersize=6, markerstrokewidth=0.4,
             xlabel="Base ψ RMSE (% IV)",
             ylabel="V1 middle-layer adapter norm ‖B₂A₂‖_F",
             title="Adapter magnitude tracks base failure",
             titlefontsize=11, legend=false,
             framestyle=:box, grid=true, gridalpha=0.25)

p3 = scatter(100 .* results_df.v0_improvement,
             100 .* results_df.v1_improvement,
             color=cell_color.(results_df.cell_kind),
             markersize=6, markerstrokewidth=0.4,
             xlabel="V0 improvement (% IV)",
             ylabel="V1 improvement (% IV)",
             title="V1 dominates V0 across cells",
             titlefontsize=11, legend=false,
             framestyle=:box, grid=true, gridalpha=0.25)
plot!(p3, x -> x, range(-2, 15; length=10);
      color=:black, ls=:dash, alpha=0.5, label="")

p4 = scatter(results_df.v1_delta_log_theta,
             100 .* (results_df.v1_norm_B2A2 .+ results_df.v1_norm_B1A1 .+ results_df.v1_norm_B3A3) ./ 3,
             color=cell_color.(results_df.cell_kind),
             markersize=6, markerstrokewidth=0.4,
             xlabel="V1 Δlog θ (level shift)",
             ylabel="V1 mean ‖BA‖_F (shape perturbation, ×100)",
             title="Level vs shape adaptation",
             titlefontsize=11, legend=false,
             framestyle=:box, grid=true, gridalpha=0.25)
vline!(p4, [0]; color=:black, alpha=0.4, label="")

p_grid = plot(p1, p2, p3, p4, layout=(2, 2),
              size=(1300, 1000), dpi=200,
              left_margin=8mm, right_margin=4mm,
              top_margin=4mm, bottom_margin=7mm)
fig_pdf = joinpath(FIG_DIR, "lora_sweep_summary.pdf")
fig_png = joinpath(FIG_DIR, "lora_sweep_summary.png")
savefig(p_grid, fig_pdf); savefig(p_grid, fig_png)
@printf("[fig] wrote -> %s / .png\n", fig_pdf)

# ============================================================================
# Console summary
# ============================================================================

println("\n" * "="^78)
println("  LoRA SWEEP SUMMARY")
println("="^78)
for k in ["worst", "middle", "best"]
    sub = results_df[results_df.cell_kind .== k, :]
    nrow(sub) == 0 && continue
    @printf("  %-7s cells (n=%d):  median base = %5.2f%%, V0 = %5.2f%%, V1 = %5.2f%%   median V1 improvement = %5.2f%% IV   median ‖B₂A₂‖ = %.3f\n",
            k, nrow(sub),
            100*median(sub.base_rmse),
            100*median(sub.v0_rmse),
            100*median(sub.v1_rmse),
            100*median(sub.v1_improvement),
            median(sub.v1_norm_B2A2))
end
println()
@printf("  V1 beats V0 on %d / %d cells\n",
        sum(results_df.v1_improvement .> results_df.v0_improvement), nrow(results_df))
@printf("  V1 worsens RMSE (overfit) on %d / %d cells\n",
        sum(results_df.v1_rmse .> results_df.base_rmse), nrow(results_df))

println("\nDone.")
