"""
NVDA IV Smile — Model vs Market

Standalone plotting script. Loads the NVDA option chain, fits the single-DTE
smile (β₁ fixed at 0.20, optimizing θ_base, β₂, β₃, β₄), and saves a
publication-quality figure of market data vs model curve.
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

const DATA_DIR   = joinpath(@__DIR__, "..", "data", "options")
const FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

# ============================================================================
# Load NVDA option chain
# ============================================================================

csv_path  = joinpath(DATA_DIR, "nvda.csv")
toml_path = joinpath(DATA_DIR, "nvda.toml")

df   = CSV.read(csv_path, DataFrame)
meta = TOML.parsefile(toml_path)["metadata"]

S   = parse(Float64, meta["underlying_share_price_mid"])
DTE = parse(Int,     meta["DTE"])

df[!, :KoverS] = df.Strike ./ S

# Filter to tradeable range
valid = df[(df.IV .> 0.01) .& (df.IV .< 2.0) .&
           (df.KoverS .> 0.70) .& (df.KoverS .< 1.30), :]

println("NVDA: $(nrow(valid)) contracts after filter  |  S=\$$(round(S,digits=2))  DTE=$DTE")

# ============================================================================
# Calibrate: optimize θ_base, β₂, β₃, β₄  (β₁ = 0.20 fixed)
# ============================================================================

const β₁_FIXED = 0.20

strikes   = valid.Strike
ivs       = valid.IV
moneyness = strikes ./ S
log_DTE   = log(max(DTE, 1.0))

x0 = [log(mean(ivs)^2), 0.0, 0.0, 0.0]

function objective(x)
    θ_base       = exp(x[1])
    β₂, β₃, β₄  = x[2], x[3], x[4]
    err = 0.0
    for i in eachindex(ivs)
        log_m  = log(moneyness[i])
        ψ_val  = exp(β₁_FIXED * log_DTE + β₂ * log_m + β₃ * log_DTE * log_m + β₄ * log_m^2)
        σ_mod  = sqrt(max(θ_base * ψ_val, 1e-10))
        err   += (σ_mod - ivs[i])^2
    end
    return err / length(ivs)
end

result  = optimize(objective, x0, NelderMead(), Optim.Options(iterations=10_000, g_tol=1e-12))
x_opt   = Optim.minimizer(result)
θ_base  = exp(x_opt[1])
β       = [β₁_FIXED, x_opt[2], x_opt[3], x_opt[4]]
rmse    = sqrt(Optim.minimum(result))

println("θ_base=$(round(θ_base,digits=5))  β=$(round.(β,digits=4))  RMSE=$(round(rmse*100,digits=2))% IV")

# ============================================================================
# Build smooth model curve across the strike range
# ============================================================================

K_min   = minimum(valid.Strike)
K_max   = maximum(valid.Strike)
K_curve = range(K_min, K_max, length=400)

function model_iv(K)
    log_m = log(K / S)
    ψ     = exp(β[1] * log_DTE + β[2] * log_m + β[3] * log_DTE * log_m + β[4] * log_m^2)
    return sqrt(max(θ_base * ψ, 1e-10))
end

iv_curve = model_iv.(K_curve) .* 100   # percent

# ============================================================================
# Split market data into calls and puts
# ============================================================================

calls = valid[valid.Type .== "Call", :]
puts  = valid[valid.Type .== "Put",  :]

# ============================================================================
# Publication-quality figure
# ============================================================================

# --- Aesthetics ---
col_calls  = RGB(0.27, 0.51, 0.71)   # steel blue
col_puts   = RGB(0.84, 0.37, 0.30)   # coral / tomato
col_model  = RGB(0.13, 0.13, 0.13)   # near-black
col_atm    = RGB(0.55, 0.55, 0.55)   # medium gray
col_grid   = RGB(0.90, 0.90, 0.90)   # light gray

gr()   # GR backend — always available, produces clean raster output

col_spot = RGB(0.38, 0.38, 0.38)

fig = plot(
    # canvas
    size          = (600, 400),
    dpi           = 300,
    background_color = :white,
    foreground_color = :black,

    # axes labels
    xlabel        = "Strike Price (\$)",
    ylabel        = "Implied Volatility (%)",
    xlabelfontsize = 11,
    ylabelfontsize = 11,

    # tick styling
    tickfontsize  = 10,
    tick_direction = :out,
    minorticks    = false,

    # axis limits — show full data range with clean tick positions
    xlims         = (142.0, 295.0),
    xticks        = collect(150.0:25.0:275.0),
    ylims         = (38.0, 62.0),
    yticks        = collect(40.0:5.0:60.0),

    # frame
    framestyle    = :box,
    grid          = true,
    gridlinewidth = 0.6,
    gridcolor     = col_grid,
    gridalpha     = 1.0,

    # legend — top-right is open space at lower IV end; fully transparent, no border
    legend                  = :topright,
    legendfontsize          = 9,
    legendframealpha        = 0.0,
    legendbackground_color  = :transparent,
    legendforegroundcolor   = :transparent,

    # margins
    left_margin   = 4mm,
    bottom_margin = 3mm,
    right_margin  = 3mm,
    top_margin    = 2mm,
)

# Layer 1: ATM vertical dashed line — drawn first so it sits behind all data
# Label omitted from legend; instead annotated directly on the figure below.
vline!(fig, [S],
    color     = col_spot,
    linestyle = :dash,
    linewidth = 1.0,
    label     = nothing,
)

# Layer 2: market calls
scatter!(fig, calls.Strike, calls.IV .* 100,
    marker            = :circle,
    markersize        = 4,
    color             = col_calls,
    markerstrokewidth = 0.5,
    markerstrokecolor = :black,
    label             = "Calls (market)",
    alpha             = 0.65,
)

# Layer 3: market puts
scatter!(fig, puts.Strike, puts.IV .* 100,
    marker            = :diamond,
    markersize        = 4,
    color             = col_puts,
    markerstrokewidth = 0.5,
    markerstrokecolor = :black,
    label             = "Puts (market)",
    alpha             = 0.65,
)

# Layer 4: model curve — on top so always visible through scatter cloud
plot!(fig, collect(K_curve), iv_curve,
    color     = col_model,
    linewidth = 2.2,
    linestyle = :solid,
    label     = "Model",
)

# Annotate the spot line directly instead of cluttering the legend
annotate!(fig, S + 3.5, 61.0,
    text("\$$(round(Int, S))", :left, 8, col_spot))

# ============================================================================
# Save
# ============================================================================

outpath = joinpath(FIGURE_DIR, "nvda_iv_smile.pdf")
savefig(fig, outpath)
println("Saved → $outpath")
