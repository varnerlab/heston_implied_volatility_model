"""
IV Surface Heatmap — Publication Quality (NeurIPS single-column)

Loads NVDA option chain data across 7 DTEs, fits the ψ model (excluding 0DTE),
and generates a single heatmap of model-predicted IV as a function of moneyness
(x-axis) and DTE (y-axis), with market observation points overlaid.
"""

using CSV
using TOML
using DataFrames
using Statistics
using Optim
using Plots
using Printf

# ============================================================================
# Paths
# ============================================================================

const DATA_DIR = joinpath(@__DIR__, "..", "..", "data", "time")
const PLOT_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(PLOT_DIR)

const DTE_FILES = [
    "nvda_0dte",
    "nvda_2dte",
    "nvda_4dte",
    "nvda_7dte",
    "nvda_14dte",
    "nvda_25dte",
    "nvda_46dte",
]

# ============================================================================
# Data loading
# ============================================================================

function load_all_slices()
    frames = DataFrame[]
    for f in DTE_FILES
        csv_path = joinpath(DATA_DIR, "$(f).csv")
        toml_path = joinpath(DATA_DIR, "$(f).toml")
        df = CSV.read(csv_path, DataFrame)
        meta = TOML.parsefile(toml_path)["metadata"]
        S = parse(Float64, meta["underlying_share_price"])
        dte = parse(Int, meta["DTE"])
        df[!, :DTE] .= dte
        df[!, :S] .= S
        df[!, :Moneyness] = df.Strike ./ S
        push!(frames, df)
    end
    return vcat(frames...)
end

println("Loading data...")
all_data = load_all_slices()

# Filter: calls + puts, DTE > 0, moneyness in [0.90, 1.10], IV > 0, Volume > 0
calls = all_data[(all_data.Type .== "Call") .&
                 (all_data.DTE .> 0) .&
                 (all_data.Moneyness .>= 0.90) .&
                 (all_data.Moneyness .<= 1.10) .&
                 (all_data.IV .> 0.0) .&
                 (all_data.Volume .> 0), :]

puts = all_data[(all_data.Type .== "Put") .&
                (all_data.DTE .> 0) .&
                (all_data.Moneyness .>= 0.90) .&
                (all_data.Moneyness .<= 1.10) .&
                (all_data.IV .> 0.0) .&
                (all_data.Volume .> 0), :]

fit_data = vcat(calls, puts)
println("  Fitting observations: $(nrow(fit_data))")

# ============================================================================
# ψ model
# ============================================================================

function eval_psi(β::Vector{Float64}, log_dte::Float64, log_m::Float64)::Float64
    return exp(β[1] * log_dte + β[2] * log_m + β[3] * log_dte * log_m +
               β[4] * log_m^2 + β[5] * log_dte^2)
end

function fit_psi(df::DataFrame)
    n = nrow(df)
    ivs = df.IV
    dtes = Float64.(df.DTE)
    moneyness = df.Moneyness

    mid_dte_mask = (dtes .>= 4) .& (dtes .<= 14) .& (abs.(moneyness .- 1.0) .< 0.05)
    σ_mid = any(mid_dte_mask) ? mean(ivs[mid_dte_mask]) : mean(ivs)
    x0 = [log(σ_mid^2), 0.3, -1.0, 0.1, 2.0, -0.15]

    function objective(x)
        θ_base = exp(x[1])
        β = x[2:6]
        err = 0.0
        for i in 1:n
            log_dte = log(max(dtes[i], 1.0))
            log_m = log(moneyness[i])
            ψ = eval_psi(β, log_dte, log_m)
            σ_model = sqrt(max(θ_base * ψ, 1e-10))
            err += (σ_model - ivs[i])^2
        end
        return err / n
    end

    result = optimize(objective, x0, NelderMead(),
                      Optim.Options(iterations=100000, g_tol=1e-14))
    result = optimize(objective, Optim.minimizer(result), NelderMead(),
                      Optim.Options(iterations=100000, g_tol=1e-14))

    x_opt = Optim.minimizer(result)
    θ_base = exp(x_opt[1])
    σ_base = sqrt(θ_base)
    β = x_opt[2:6]
    return σ_base, β
end

println("Fitting ψ model...")
σ_base, β = fit_psi(fit_data)
println("  σ_base = $(round(σ_base * 100, digits=2))%")

# ============================================================================
# Build IV surface grid (100×100 for smooth appearance)
# Use log-spaced DTE grid so market observations spread evenly across the plot
# ============================================================================

m_grid   = range(0.90, 1.10, length=100)
# Log-spaced DTE grid — pad slightly beyond [1, 46] so boundary dots aren't clipped
log_dte_min = log(0.75)   # a little below DTE=1 for bottom breathing room
log_dte_max = log(52.0)   # a little above DTE=46 for top breathing room
log_dte_grid = range(log_dte_min, log_dte_max, length=100)
dte_grid_log = exp.(log_dte_grid)   # actual DTE values for evaluation

iv_surface = [σ_base * sqrt(eval_psi(β, log(d), log(m))) * 100
              for d in dte_grid_log, m in m_grid]

println("  IV surface range: $(round(minimum(iv_surface), digits=1))% – $(round(maximum(iv_surface), digits=1))%")

# ============================================================================
# Contour levels — 4 clean round numbers spanning the range
# ============================================================================

iv_min = minimum(iv_surface)
iv_max = maximum(iv_surface)
println("  IV min=$(round(iv_min,digits=1))%, max=$(round(iv_max,digits=1))%")

# Pick 3 clean interior levels — avoid 65% which hugs the upper boundary at DTE=46
all_levels = [35.0, 45.0, 55.0]
contour_levels = filter(l -> iv_min + 3 < l < iv_max - 3, all_levels)
println("  Contour levels: $contour_levels")

# ============================================================================
# Plot
# ============================================================================

# NeurIPS single-column: ~3.5 inches wide → 600 px @300 dpi (actually use 525px)
# Use GR backend (default) — fast and reliable

gr()

# Set clean colorbar tick values
cb_min = 25.0
cb_max = 65.0
cb_ticks = [25, 35, 45, 55, 65]

# Moneyness x-axis ticks
xticks_vals = [0.90, 0.95, 1.00, 1.05, 1.10]
xticks_lbls = [@sprintf("%.2f", v) for v in xticks_vals]

# DTE y-axis: actual DTE values mapped into log space
yticks_dtes = [1, 2, 4, 7, 14, 25, 46]
yticks_log  = log.(Float64.(yticks_dtes))
yticks_lbls = string.(yticks_dtes)

# Build the heatmap as the base layer — colorbar attaches here
fig = heatmap(
    collect(m_grid), log_dte_grid, iv_surface,
    color            = :viridis,
    colorbar_title   = "Implied Volatility (%)",
    colorbar_titlefontsize = 10,
    colorbar_tickfontsize  = 11,
    clims            = (cb_min, cb_max),
    colorbar_ticks   = cb_ticks,
    # Canvas / style
    size             = (620, 450),
    dpi              = 300,
    background_color        = :white,
    background_color_inside = :white,
    framestyle = :box,
    grid       = false,
    # Axis labels
    xlabel = "Moneyness (K/S)",
    ylabel = "Days to Expiration",
    # Ticks
    xticks         = (xticks_vals, xticks_lbls),
    yticks         = (yticks_log,  yticks_lbls),
    xtickfontsize  = 11,
    ytickfontsize  = 11,
    xguidefontsize = 11,
    yguidefontsize = 11,
    # Margins
    left_margin   = 5Plots.mm,
    bottom_margin = 5Plots.mm,
    right_margin  = 2Plots.mm,
    top_margin    = 4Plots.mm,
    legend   = false,
    colorbar = true,
)

# Contour lines over the heatmap
if !isempty(contour_levels)
    contour!(fig,
        collect(m_grid), log_dte_grid, iv_surface,
        levels     = contour_levels,
        linecolor  = :white,
        linewidth  = 1.0,
        linestyle  = :solid,
        # NOTE: do NOT pass colorbar=false here — it kills the heatmap colorbar in GR
        label      = false,
    )
end

# Market observation scatter — y in log(DTE) space
scatter!(fig,
    fit_data.Moneyness, log.(Float64.(fit_data.DTE)),
    marker        = :circle,
    markersize    = 4,
    markercolor   = :white,
    markerstrokecolor = :black,
    markerstrokewidth = 0.8,
    alpha         = 0.85,
    label         = false,
)

out_path = joinpath(PLOT_DIR, "iv_surface_heatmap.pdf")
savefig(fig, out_path)
println("Saved → $out_path")
