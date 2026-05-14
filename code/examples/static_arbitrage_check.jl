"""
Static-arbitrage diagnostic on calibrated ψ_NN surfaces (Reviewer 1, 2, 3).

For each of 8 surfaces (6 sector representatives + GS/LLY per-ticker), we
- build IV(K, T) = sqrt(θ_i * ψ_NN((K/S_0)_std, T_std)) on a 20×20 grid
  with K/S ∈ [0.80, 1.20] log-spaced, DTE ∈ [7, 365] log-spaced,
- price European calls via the existing CRR pricer (200 steps),
- test butterfly arbitrage   ∂²C/∂K² ≥ 0   via central second difference,
- test calendar arbitrage    ∂C/∂T  ≥ 0    via forward first difference.

Surfaces with butterfly or calendar violations are unsafe for downstream
pricing, so the JCF revision either reports the violation rates or
acknowledges no-arbitrage as a known limitation.

Outputs:
  code/figures/static_arbitrage_violations.csv
  code/figures/static_arbitrage_heatmap.pdf / .png   (optional visual)

Run:
    julia --project=. examples/static_arbitrage_check.jl
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

# ============================================================================
# Configuration
# ============================================================================

const LADDER_DIR    = joinpath(@__DIR__, "..", "data", "ladder")
const FIG_DIR       = joinpath(@__DIR__, "..", "figures")
const NN_CACHE      = joinpath(FIG_DIR, "calibrate_ladders_per_ticker_nn_cache.jld2")
const ANCHOR_DATE   = Date("2026-04-28")
const R_FREE        = 0.0425
const N_STEPS_CRR   = 200

# 8 surfaces: 6 sector representatives (with sector-NN fallback if available,
# else per-ticker) + GS/LLY per-ticker
const SURFACES = [
    (label="ETF / SPY",          ticker="SPY",  sector="ETF",        per_ticker=false),
    (label="Tech / NVDA",        ticker="NVDA", sector="Tech",       per_ticker=false),
    (label="Healthcare / LLY",   ticker="LLY",  sector="Healthcare", per_ticker=false),
    (label="Financials / GS",    ticker="GS",   sector="Financials", per_ticker=false),
    (label="Retail / WMT",       ticker="WMT",  sector="Retail",     per_ticker=false),
    (label="Energy / XOM",       ticker="XOM",  sector="Energy",     per_ticker=false),
    (label="GS per-ticker",      ticker="GS",   sector="Financials", per_ticker=true),
    (label="LLY per-ticker",     ticker="LLY",  sector="Healthcare", per_ticker=true),
]

const N_K = 20
const N_T = 20
const K_OVER_S_LO = 0.80
const K_OVER_S_HI = 1.20
const DTE_LO = 7
const DTE_HI = 365

# ============================================================================
# Shared infrastructure (mirror what ScenarioTemplate does internally)
# ============================================================================

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
        df = CSV.read(f, DataFrame)
        ticker = string(df.underlying[1])
        S = df.und_close[1]
        df[!, :ticker] .= ticker
        df[!, :S] .= S
        df[!, :moneyness] = df.strike ./ S
        v = df[
            .!ismissing.(df.implied_vol) .&
            .!isnan.(coalesce.(df.implied_vol, NaN)) .&
            (coalesce.(df.implied_vol, 0.0) .> 0.01) .&
            (coalesce.(df.implied_vol, 999.0) .< 2.0) .&
            (df.bid .> 0) .&
            (df.moneyness .>= 0.80) .&
            (df.moneyness .<= 1.20) .&
            (df.actual_dte .> 0), :]
        nrow(v) > 0 && push!(frames, v)
    end
    return vcat(frames...)
end

println("Loading ladder corpus for standardisation constants...")
all_data = _load_all_ladders(LADDER_DIR)
const MU_DTE    = mean(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const SIGMA_DTE = std(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const MU_M      = mean(log.(Float64.(all_data.moneyness)))
const SIGMA_M   = std(log.(Float64.(all_data.moneyness)))

function anchor_spot(ticker)
    slice = all_data[all_data.ticker .== ticker, :]
    rows = slice[slice.und_session_date .== ANCHOR_DATE, :]
    isempty(rows) && error("no $(ticker) rows for $(ANCHOR_DATE)")
    return Float64(rows.S[1])
end

println("Loading NN cache from $(basename(NN_CACHE)) ...")
nn_cache = JLD2.load(NN_CACHE)

function build_iv_fn(ticker::String, sector::String, per_ticker::Bool)
    psi_nn, log_theta, src = ScenarioTemplate._restore_nn(nn_cache, ticker;
        use_per_ticker=per_ticker, sector=sector)
    theta = exp(log_theta)
    iv = function(K::Float64, S::Float64, dte_days::Real)
        log_dte_s = Float32((log(max(Float64(dte_days), 1.0)) - MU_DTE) / SIGMA_DTE)
        log_m_s   = Float32((log(K / S) - MU_M) / SIGMA_M)
        x = reshape(Float32[log_dte_s, log_m_s], 2, 1)
        log_psi = first(psi_nn(x))
        return sqrt(max(theta * exp(Float64(log_psi)), 1e-10))
    end
    return iv, src
end

# ============================================================================
# Grid + violation tests
# ============================================================================

function evaluate_surface(iv_fn, S_0)
    # Log-spaced grids
    k_over_s = exp.(range(log(K_OVER_S_LO), log(K_OVER_S_HI); length=N_K))
    dte_grid = exp.(range(log(DTE_LO),     log(DTE_HI);     length=N_T))
    K_grid = k_over_s .* S_0
    # C[i, j] = call price at K_grid[i], dte_grid[j]
    C = Array{Float64}(undef, N_K, N_T)
    IV = Array{Float64}(undef, N_K, N_T)
    for j in 1:N_T
        T_yrs = dte_grid[j] / 365.0
        for i in 1:N_K
            σ = iv_fn(K_grid[i], S_0, dte_grid[j])
            IV[i, j] = σ
            C[i, j] = crr_european_price(S_0, K_grid[i], σ, R_FREE, T_yrs, N_STEPS_CRR, :call)
        end
    end
    # Butterfly test: ∂²C/∂K² ≥ 0 — central second difference along K (inner 18 strikes per row)
    bfly = fill(NaN, N_K, N_T)
    for j in 1:N_T
        for i in 2:(N_K - 1)
            Δ_up = K_grid[i+1] - K_grid[i]
            Δ_dn = K_grid[i]   - K_grid[i-1]
            # Use the standard non-uniform second difference
            bfly[i, j] = 2 * (Δ_dn * C[i+1, j] - (Δ_up + Δ_dn) * C[i, j] + Δ_up * C[i-1, j]) /
                        (Δ_up * Δ_dn * (Δ_up + Δ_dn))
        end
    end
    # Calendar test: ∂C/∂T ≥ 0 — forward first difference along T (inner N_T-1 maturities)
    cal = fill(NaN, N_K, N_T)
    for i in 1:N_K
        for j in 1:(N_T - 1)
            cal[i, j] = (C[i, j+1] - C[i, j]) / (dte_grid[j+1] - dte_grid[j])
        end
    end
    return (K_grid=K_grid, dte_grid=dte_grid, C=C, IV=IV, butterfly=bfly, calendar=cal)
end

function violation_counts(surface)
    bf = surface.butterfly
    cl = surface.calendar
    bf_valid = .!isnan.(bf)
    cl_valid = .!isnan.(cl)
    bf_viol  = bf_valid .& (bf .< 0.0)
    cl_viol  = cl_valid .& (cl .< 0.0)
    n_bf     = sum(bf_valid)
    n_cl     = sum(cl_valid)
    n_bf_v   = sum(bf_viol)
    n_cl_v   = sum(cl_viol)
    max_bf_v = n_bf_v > 0 ? minimum(bf[bf_viol]) : 0.0   # most-negative
    max_cl_v = n_cl_v > 0 ? minimum(cl[cl_viol]) : 0.0
    return (n_butterfly=n_bf, n_butterfly_viol=n_bf_v,
            frac_butterfly_viol=n_bf_v / max(n_bf, 1),
            n_calendar=n_cl, n_calendar_viol=n_cl_v,
            frac_calendar_viol=n_cl_v / max(n_cl, 1),
            max_butterfly_violation=max_bf_v,
            max_calendar_violation=max_cl_v)
end

# ============================================================================
# Run all 8 surfaces
# ============================================================================

rows = NamedTuple[]
surfaces_for_plot = []
for s in SURFACES
    @printf("\n[%-24s] ticker=%s sector=%s per_ticker=%s\n", s.label, s.ticker, s.sector, s.per_ticker)
    iv_fn, src = build_iv_fn(s.ticker, s.sector, s.per_ticker)
    S_0 = anchor_spot(s.ticker)
    surf = evaluate_surface(iv_fn, S_0)
    vc = violation_counts(surf)
    @printf("  S_0 = \$%.2f   ψ source = %s\n", S_0, src)
    @printf("  butterfly  : %4d / %4d violations  (%.2f%%)  max neg %+0.4e\n",
            vc.n_butterfly_viol, vc.n_butterfly, 100*vc.frac_butterfly_viol, vc.max_butterfly_violation)
    @printf("  calendar   : %4d / %4d violations  (%.2f%%)  max neg %+0.4e\n",
            vc.n_calendar_viol, vc.n_calendar, 100*vc.frac_calendar_viol, vc.max_calendar_violation)
    push!(rows, (
        surface_label = s.label,
        ticker = s.ticker, sector = s.sector,
        psi_source = String(src),
        S_0 = S_0,
        n_grid_pts = N_K * N_T,
        n_butterfly = vc.n_butterfly, n_butterfly_viol = vc.n_butterfly_viol,
        frac_butterfly_viol = vc.frac_butterfly_viol,
        max_butterfly_violation = vc.max_butterfly_violation,
        n_calendar = vc.n_calendar, n_calendar_viol = vc.n_calendar_viol,
        frac_calendar_viol = vc.frac_calendar_viol,
        max_calendar_violation = vc.max_calendar_violation,
    ))
    push!(surfaces_for_plot, (label=s.label, surf=surf))
end

# ============================================================================
# CSV
# ============================================================================

mkpath(FIG_DIR)
df = DataFrame(rows)
out_csv = joinpath(FIG_DIR, "static_arbitrage_violations.csv")
CSV.write(out_csv, df)
@printf("\n[csv] wrote -> %s\n", out_csv)

println("\nAggregate across all 8 surfaces:")
@printf("  Total butterfly violations:  %d / %d  (%.2f%%)\n",
        sum(df.n_butterfly_viol), sum(df.n_butterfly),
        100 * sum(df.n_butterfly_viol) / sum(df.n_butterfly))
@printf("  Total calendar  violations:  %d / %d  (%.2f%%)\n",
        sum(df.n_calendar_viol), sum(df.n_calendar),
        100 * sum(df.n_calendar_viol) / sum(df.n_calendar))

# ============================================================================
# Heatmap: butterfly violations across all 8 surfaces (2x4)
# ============================================================================

println("\nRendering 2x4 violation heatmap...")
panels = []
for entry in surfaces_for_plot
    s = entry.surf
    # Show butterfly term: red where violating, blue/green where safe
    bfly = s.butterfly
    # Replace NaN border with 0 for plotting
    bfly_plot = copy(bfly); bfly_plot[isnan.(bfly_plot)] .= 0.0
    p = heatmap(s.dte_grid, s.K_grid ./ s.K_grid[Int(round(N_K/2))], bfly_plot;
                color=:RdYlGn,
                clims=(-maximum(abs, filter(!isnan, bfly)),
                        maximum(abs, filter(!isnan, bfly))),
                title=entry.label, titlefontsize=10,
                xlabel="DTE (days)", ylabel="K / K_ATM",
                guidefontsize=9, tickfontsize=8,
                colorbar_title="∂²C/∂K²",
                xscale=:log10, yscale=:log10,
                framestyle=:box)
    push!(panels, p)
end
p_grid = plot(panels..., layout=(2, 4), size=(1800, 800), dpi=180,
              left_margin=8mm, right_margin=4mm, top_margin=4mm, bottom_margin=6mm)
out_pdf = joinpath(FIG_DIR, "static_arbitrage_heatmap.pdf")
out_png = joinpath(FIG_DIR, "static_arbitrage_heatmap.png")
savefig(p_grid, out_pdf); savefig(p_grid, out_png)
@printf("[fig] wrote -> %s / .png\n", out_pdf)

println("\nDone.")
