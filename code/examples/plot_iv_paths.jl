"""
Forward IV Paths — standalone plotting script for publication figure.

Loads NVDA options data, calibrates θ_base and β, simulates 500 Heston variance
paths, then plots the IV path distribution.
"""

using CSV
using TOML
using DataFrames
using Statistics
using Optim
using Plots
using Plots.PlotMeasures
using Random

# ============================================================================
# Step 1: Load NVDA option chain
# ============================================================================

const DATA_DIR = joinpath(@__DIR__, "..", "data", "options")

csv_path  = joinpath(DATA_DIR, "nvda.csv")
toml_path = joinpath(DATA_DIR, "nvda.toml")

df   = CSV.read(csv_path, DataFrame)
meta = TOML.parsefile(toml_path)["metadata"]

S      = parse(Float64, meta["underlying_share_price_mid"])
DTE    = parse(Int,     meta["DTE"])
atm_iv = parse(Float64, meta["atm_IV"])

df[!, :KoverS] = df.Strike ./ S
valid = df[(df.IV .> 0.01) .& (df.IV .< 2.0) .&
           (df.KoverS .> 0.70) .& (df.KoverS .< 1.30), :]

println("NVDA: $(nrow(valid)) contracts, S=\$$(round(S, digits=2)), DTE=$DTE, ATM IV=$(round(atm_iv*100, digits=1))%")

# ============================================================================
# Step 2: Calibrate θ_base and β on NVDA (β₁ = 0.20 fixed)
# ============================================================================

const β₁_FIXED = 0.20

n_obs     = nrow(valid)
strikes   = valid.Strike
ivs       = valid.IV
moneyness = strikes ./ S
log_DTE   = log(max(Float64(DTE), 1.0))

x0 = [log(mean(ivs)^2), 0.0, 0.0, 0.0]

function objective(x)
    θ_b = exp(x[1])
    β₂, β₃, β₄ = x[2], x[3], x[4]
    err = 0.0
    for i in 1:n_obs
        log_m = log(moneyness[i])
        ψ_val = exp(β₁_FIXED * log_DTE + β₂ * log_m + β₃ * log_DTE * log_m + β₄ * log_m^2)
        σ_model = sqrt(max(θ_b * ψ_val, 1e-10))
        err += (σ_model - ivs[i])^2
    end
    return err / n_obs
end

result  = optimize(objective, x0, NelderMead(), Optim.Options(iterations=10000, g_tol=1e-12))
x_opt   = Optim.minimizer(result)
θ_base  = exp(x_opt[1])
β       = [β₁_FIXED, x_opt[2], x_opt[3], x_opt[4]]
rmse    = sqrt(Optim.minimum(result))

println("θ_base = $(round(θ_base, digits=6))  (√θ = $(round(sqrt(θ_base)*100, digits=1))% IV)")
println("β = $β")
println("RMSE = $(round(rmse*100, digits=2))% IV")

# ============================================================================
# Step 3: Simulate Heston variance paths — ATM, pure θ_base dynamics
# ============================================================================

const N_PATHS = 500
const κ       = 5.0
const σ_v     = 0.3
const Δt      = 1.0 / 252.0

# ψ for ATM (moneyness=1) as a function of remaining DTE
function ψ_atm(dte::Float64)
    return exp(β[1] * log(max(dte, 1.0)))
end

# For ATM paths, the time-varying mean reversion target is
#   θ(t) = θ_base · ψ_atm(DTE - t)
# i.e., the model expects IV to follow the term structure as DTE shrinks.

Random.seed!(42)
rng = MersenneTwister(42)

# iv_paths[path, day] — IV in fraction (not %)
iv_paths = Matrix{Float64}(undef, N_PATHS, DTE)

for p in 1:N_PATHS
    # Initialise at equilibrium for current DTE
    v = θ_base * ψ_atm(Float64(DTE))
    for t in 1:DTE
        iv_paths[p, t] = sqrt(max(v, 0.0))
        # Remaining DTE at start of day t+1
        rem_dte = max(Float64(DTE - t), 1.0)
        θ_t  = θ_base * ψ_atm(rem_dte)
        v_c  = max(v, 0.0)
        dv   = κ * (θ_t - v_c) * Δt + σ_v * sqrt(v_c) * sqrt(Δt) * randn(rng)
        v    = abs(v_c + dv)
    end
end

# Summary statistics
iv_mean = vec(mean(iv_paths, dims=1))
iv_q05  = [quantile(iv_paths[:, t], 0.05) for t in 1:DTE]
iv_q95  = [quantile(iv_paths[:, t], 0.95) for t in 1:DTE]

days = 1:DTE

# ============================================================================
# Step 4: Publication-quality plot
# ============================================================================

const SAMPLE_N    = 200
const MUTED_BLUE  = RGB(0.42, 0.60, 0.80)   # steel blue — visible but not overwhelming
const DARK_RED    = RGB(0.60, 0.10, 0.10)
const DARK_GREEN  = RGB(0.10, 0.45, 0.20)
const LINE_BLACK  = RGB(0.08, 0.08, 0.08)   # near-black for mean
const GOLD        = RGB(0.85, 0.60, 0.10)

# Helvetica font alias (falls back gracefully on systems without it)
const FONT = "Helvetica"

gr()

p = plot(
    size        = (700, 450),
    dpi         = 300,
    background_color = :white,
    framestyle  = :box,
    grid        = true,
    gridcolor   = RGB(0.88, 0.88, 0.88),
    gridlinewidth = 0.5,
    minorgrid   = false,
    left_margin  = 8mm,
    bottom_margin = 6mm,
    right_margin  = 4mm,
    top_margin    = 4mm,
    legend      = :topright,
    legendfontsize  = 10,
    legendfontfamily = FONT,
    legend_background_color = :transparent,
    legend_foreground_color = :transparent,
    foreground_color_legend = :transparent,
    background_color_legend = :transparent,
    tickfontsize  = 10,
    tickfontfamily = FONT,
    guidefontsize  = 11,
    guidefontfamily = FONT,
    xlabel = "Trading Days",
    ylabel = "Implied Volatility (%)",
    ylims  = (26.0, 59.0),
    xlims  = (-1, DTE + 2),
    xticks = 0:20:DTE,
    yticks = 30:5:55,
)

# Sample paths (thin, semi-transparent) — draw first so stat lines render on top
for i in 1:SAMPLE_N
    plot!(p, days, iv_paths[i, :] .* 100,
          color     = MUTED_BLUE,
          alpha     = 0.07,
          linewidth = 0.3,
          label     = nothing)
end

# 5th percentile
plot!(p, days, iv_q05 .* 100,
      color     = DARK_RED,
      linewidth = 1.5,
      linestyle = :dash,
      label     = "5th percentile")

# 95th percentile
plot!(p, days, iv_q95 .* 100,
      color     = DARK_GREEN,
      linewidth = 1.5,
      linestyle = :dash,
      label     = "95th percentile")

# Mean IV
plot!(p, days, iv_mean .* 100,
      color     = LINE_BLACK,
      linewidth = 2.8,
      label     = "Mean IV")

# Market ATM IV reference
hline!(p, [atm_iv * 100],
       color     = GOLD,
       linewidth = 1.5,
       linestyle = :dot,
       label     = "Market ATM IV")

const PLOT_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(PLOT_DIR)
out_path = joinpath(PLOT_DIR, "forward_iv_paths.pdf")
savefig(p, out_path)
println("Saved → $out_path")

include(joinpath(@__DIR__, "..", "scripts", "promote_figures.jl"))
promote_figures()
