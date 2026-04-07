"""
IV Residuals — Cross-Ticker
Publication-quality standalone plot for NeurIPS-style paper.

Loads option chains for NVDA, AMD, MU, INTC; calibrates β on NVDA
(β₁=0.20 fixed, fit θ_base, β₂, β₃, β₄); re-fits only θ_base per ticker;
plots residuals (model IV - market IV) vs moneyness colored by ticker.
"""

using CSV
using TOML
using DataFrames
using Statistics
using Optim
using Plots
using Plots.PlotMeasures

# ============================================================================
# Paths
# ============================================================================

const DATA_DIR = joinpath(@__DIR__, "..", "data", "options")
const PLOT_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(PLOT_DIR)

const TICKERS = ["nvda", "amd", "mu", "intc"]

# Publication colors matching the all-tickers smile plot
# navy, crimson, forest green, dark orange
const TICKER_COLORS = Dict(
    "nvda" => RGB(0.122, 0.247, 0.490),   # navy
    "amd"  => RGB(0.698, 0.094, 0.125),   # crimson
    "mu"   => RGB(0.133, 0.420, 0.184),   # forest green
    "intc" => RGB(0.816, 0.400, 0.000),   # dark orange
)

const TICKER_LABELS = Dict(
    "nvda" => "NVDA",
    "amd"  => "AMD",
    "mu"   => "MU",
    "intc" => "INTC",
)

# ============================================================================
# Data loading
# ============================================================================

function load_chain(ticker::String)
    csv_path  = joinpath(DATA_DIR, "$(ticker).csv")
    toml_path = joinpath(DATA_DIR, "$(ticker).toml")

    df   = CSV.read(csv_path, DataFrame)
    meta = TOML.parsefile(toml_path)["metadata"]

    S     = parse(Float64, meta["underlying_share_price_mid"])
    DTE   = parse(Int,     meta["DTE"])
    atm_iv = parse(Float64, meta["atm_IV"])

    df[!, :KoverS] = df.Strike ./ S

    # Filter: IV in (0.01, 2.0), moneyness in [0.70, 1.30]
    valid = df[(df.IV .> 0.01) .& (df.IV .< 2.0) .&
               (df.KoverS .>= 0.70) .& (df.KoverS .<= 1.30), :]

    return valid, S, DTE, atm_iv
end

println("Loading option chains...")
chains = Dict{String,Any}()
for t in TICKERS
    df, S, DTE, atm_iv = load_chain(t)
    chains[t] = (data=df, S=S, DTE=DTE, atm_iv=atm_iv)
    println("  $(uppercase(t)): $(nrow(df)) contracts, S=\$$(round(S, digits=2)), DTE=$DTE, ATM IV=$(round(atm_iv*100, digits=1))%")
end

# ============================================================================
# ψ helper
# ============================================================================

const β₁_FIXED = 0.20

function psi_val(β::Vector{Float64}, DTE::Float64, moneyness::Float64)::Float64
    log_DTE = log(max(DTE, 1.0))
    log_m   = log(moneyness)
    return exp(β[1] * log_DTE + β[2] * log_m + β[3] * log_DTE * log_m + β[4] * log_m^2)
end

# ============================================================================
# Step 1: Calibrate on NVDA — fit θ_base, β₂, β₃, β₄; β₁=0.20 fixed
# ============================================================================

function calibrate_nvda(df, S, DTE; β₁=β₁_FIXED)
    n       = nrow(df)
    ivs     = df.IV
    m       = df.Strike ./ S
    log_DTE = log(max(Float64(DTE), 1.0))

    x0 = [log(mean(ivs)^2), 0.0, 0.0, 0.0]  # [log(θ_base), β₂, β₃, β₄]

    function obj(x)
        θ_base        = exp(x[1])
        β₂, β₃, β₄   = x[2], x[3], x[4]
        β = [β₁, β₂, β₃, β₄]
        err = 0.0
        for i in 1:n
            ψ       = psi_val(β, Float64(DTE), m[i])
            σ_model = sqrt(max(θ_base * ψ, 1e-10))
            err    += (σ_model - ivs[i])^2
        end
        return err / n
    end

    res = optimize(obj, x0, NelderMead(), Optim.Options(iterations=20000, g_tol=1e-14))
    res = optimize(obj, Optim.minimizer(res), NelderMead(),
                   Optim.Options(iterations=20000, g_tol=1e-14))

    x_opt  = Optim.minimizer(res)
    θ_base = exp(x_opt[1])
    β      = [β₁, x_opt[2], x_opt[3], x_opt[4]]
    rmse   = sqrt(Optim.minimum(res))

    model_ivs = [sqrt(max(θ_base * psi_val(β, Float64(DTE), m[i]), 1e-10)) for i in 1:n]

    return θ_base, β, rmse, model_ivs
end

println("\nCalibrating on NVDA...")
nvda        = chains["nvda"]
θ_base_nvda, β_nvda, rmse_nvda, model_ivs_nvda =
    calibrate_nvda(nvda.data, nvda.S, nvda.DTE)

println("  θ_base = $(round(θ_base_nvda, digits=6))  (√θ = $(round(sqrt(θ_base_nvda)*100, digits=1))%)")
println("  β = [$(join(round.(β_nvda, digits=4), ", "))]")
println("  RMSE = $(round(rmse_nvda*100, digits=2))% IV")

# ============================================================================
# Step 2: Re-fit only θ_base per ticker (β shared from NVDA)
# ============================================================================

function fit_theta_base(df, S, DTE, β::Vector{Float64})
    n   = nrow(df)
    ivs = df.IV
    m   = df.Strike ./ S

    function obj(x)
        θ_base = exp(x[1])
        err = 0.0
        for i in 1:n
            ψ       = psi_val(β, Float64(DTE), m[i])
            σ_model = sqrt(max(θ_base * ψ, 1e-10))
            err    += (σ_model - ivs[i])^2
        end
        return err / n
    end

    res    = optimize(obj, [log(mean(ivs)^2)], NelderMead(),
                      Optim.Options(iterations=10000, g_tol=1e-14))
    θ_base = exp(Optim.minimizer(res)[1])
    rmse   = sqrt(Optim.minimum(res))

    model_ivs = [sqrt(max(θ_base * psi_val(β, Float64(DTE), m[i]), 1e-10)) for i in 1:n]

    return θ_base, rmse, model_ivs
end

println("\nFitting θ_base per ticker (shared β from NVDA)...")
results = Dict{String,Any}()

# NVDA uses its own calibration
results["nvda"] = (θ_base  = θ_base_nvda,
                   rmse    = rmse_nvda,
                   model_ivs = model_ivs_nvda)

for t in ["amd", "mu", "intc"]
    ch                  = chains[t]
    θ_b, rmse_t, mivs  = fit_theta_base(ch.data, ch.S, ch.DTE, β_nvda)
    results[t]          = (θ_base=θ_b, rmse=rmse_t, model_ivs=mivs)
    println("  $(uppercase(t)): θ_base=$(round(θ_b, digits=4))  (√θ=$(round(sqrt(θ_b)*100, digits=1))%),  RMSE=$(round(rmse_t*100, digits=2))%")
end

# ============================================================================
# Step 3: Compute residuals
# ============================================================================

struct TickerResiduals
    moneyness ::Vector{Float64}
    residuals ::Vector{Float64}   # (model - market) × 100, in pct
end

ticker_res = Dict{String,TickerResiduals}()
for t in TICKERS
    ch  = chains[t]
    m   = ch.data.Strike ./ ch.S
    res = (results[t].model_ivs .- ch.data.IV) .* 100.0
    ticker_res[t] = TickerResiduals(m, res)
end

# ============================================================================
# Step 4: Plot
# ============================================================================

gr()

all_res_vals = vcat([ticker_res[t].residuals for t in TICKERS]...)

# Symmetric y-axis: 93rd-percentile of |residuals|, capped at 5.5%, rounded to 0.5
ylim_abs = ceil(quantile(abs.(all_res_vals), 0.93) * 2.0) / 2.0
ylim_abs = clamp(ylim_abs, 2.5, 5.5)

# Clean tick strategy: use round multiples of 1 or 2 that span the range cleanly
if ylim_abs <= 3.0
    ytick_step = 1.0
    yticks_vals = collect(-floor(ylim_abs) : ytick_step : floor(ylim_abs))
elseif ylim_abs <= 4.5
    ytick_step = 1.0
    yticks_vals = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    yticks_vals = filter(v -> abs(v) <= ylim_abs + 0.1, yticks_vals)
else
    # ±5 or ±5.5: use ±2, ±4 ticks plus zero
    yticks_vals = [-4.0, -2.0, 0.0, 2.0, 4.0]
end
0.0 in yticks_vals || (push!(yticks_vals, 0.0); sort!(yticks_vals))

p = plot(
    size                    = (700, 350),
    dpi                     = 300,
    framestyle              = :box,
    background_color        = :white,
    grid                    = false,
    xlims                   = (0.66, 1.34),
    ylims                   = (-ylim_abs - 0.4, ylim_abs + 0.4),
    xticks                  = 0.70:0.10:1.30,
    yticks                  = yticks_vals,
    xlabel                  = "Moneyness (K/S)",
    ylabel                  = "IV Residual (%)",
    tickfontsize            = 10,
    guidefontsize           = 11,
    legendfontsize          = 9,
    legend                  = :topright,
    foreground_color_legend = nothing,
    background_color_legend = :white,
    left_margin             = 6mm,
    bottom_margin           = 6mm,
    right_margin            = 4mm,
    top_margin              = 3mm,
    fontfamily              = "Helvetica",
)

# Horizontal grid — light gray for non-zero; zero line is solid medium-gray, thicker
for yv in yticks_vals
    hline!(p, [yv],
           color = yv == 0.0 ? RGB(0.40, 0.40, 0.40) : RGB(0.91, 0.91, 0.91),
           lw    = yv == 0.0 ? 1.2 : 0.55,
           label = nothing)
end

# Scatter per ticker
for t in TICKERS
    tr = ticker_res[t]
    c  = TICKER_COLORS[t]
    scatter!(p, tr.moneyness, tr.residuals,
             label             = TICKER_LABELS[t],
             marker            = :circle,
             ms                = 3.5,
             color             = c,
             markerstrokecolor = :black,
             markerstrokewidth = 0.5,
             alpha             = 0.75)
end

out_path = joinpath(PLOT_DIR, "iv_residuals_cross_ticker.pdf")
savefig(p, out_path)
println("\nSaved → $out_path")
