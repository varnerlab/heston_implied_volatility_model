"""
ATM Term Structure — Publication-Quality Figure

Loads NVDA option chains across 7 DTEs (0, 2, 4, 7, 14, 25, 46 days),
fits the ψ surface (excluding 0DTE), and produces a single ATM term structure
plot suitable for a NeurIPS single-column figure.
"""

using CSV
using TOML
using DataFrames
using Statistics
using Optim
using Plots
using Plots.PlotMeasures
using Printf

# ============================================================================
# Configuration
# ============================================================================

const DATA_DIR  = joinpath(@__DIR__, "..", "..", "data", "time")
const PLOT_DIR  = joinpath(@__DIR__, "..", "figures")
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
        csv_path  = joinpath(DATA_DIR, "$(f).csv")
        toml_path = joinpath(DATA_DIR, "$(f).toml")

        df   = CSV.read(csv_path, DataFrame)
        meta = TOML.parsefile(toml_path)["metadata"]

        S   = parse(Float64, meta["underlying_share_price"])
        dte = parse(Int,     meta["DTE"])

        df[!, :DTE]      .= dte
        df[!, :S]        .= S
        df[!, :Moneyness] = df.Strike ./ S

        push!(frames, df)
    end
    return vcat(frames...)
end

all_data = load_all_slices()

# Filter: exclude 0DTE, moneyness [0.90, 1.10], IV > 0, Volume > 0
calls = all_data[(all_data.Type     .== "Call") .&
                 (all_data.DTE      .>  0)      .&
                 (all_data.Moneyness .>= 0.90)  .&
                 (all_data.Moneyness .<= 1.10)  .&
                 (all_data.IV       .>  0.0)    .&
                 (all_data.Volume   .>  0), :]

puts  = all_data[(all_data.Type     .== "Put")  .&
                 (all_data.DTE      .>  0)      .&
                 (all_data.Moneyness .>= 0.90)  .&
                 (all_data.Moneyness .<= 1.10)  .&
                 (all_data.IV       .>  0.0)    .&
                 (all_data.Volume   .>  0), :]

fit_data = vcat(calls, puts)

# ============================================================================
# ψ surface fit
# ============================================================================

function eval_psi(β::Vector{Float64}, log_dte::Float64, log_m::Float64)::Float64
    return exp(β[1]*log_dte + β[2]*log_m + β[3]*log_dte*log_m +
               β[4]*log_m^2 + β[5]*log_dte^2)
end

function fit_psi(df::DataFrame)
    n        = nrow(df)
    ivs      = df.IV
    dtes     = Float64.(df.DTE)
    moneyness = df.Moneyness

    mid_dte_mask = (dtes .>= 4) .& (dtes .<= 14) .& (abs.(moneyness .- 1.0) .< 0.05)
    σ_mid = any(mid_dte_mask) ? mean(ivs[mid_dte_mask]) : mean(ivs)
    x0 = [log(σ_mid^2), 0.3, -1.0, 0.1, 2.0, -0.15]

    function objective(x)
        θ_base = exp(x[1])
        β      = x[2:6]
        err    = 0.0
        for i in 1:n
            log_dte = log(max(dtes[i], 1.0))
            log_m   = log(moneyness[i])
            ψ       = eval_psi(β, log_dte, log_m)
            σ_model = sqrt(max(θ_base * ψ, 1e-10))
            err    += (σ_model - ivs[i])^2
        end
        return err / n
    end

    result = optimize(objective, x0, NelderMead(),
                      Optim.Options(iterations=100_000, g_tol=1e-14))
    result = optimize(objective, Optim.minimizer(result), NelderMead(),
                      Optim.Options(iterations=100_000, g_tol=1e-14))

    x_opt  = Optim.minimizer(result)
    θ_base = exp(x_opt[1])
    σ_base = sqrt(θ_base)
    β      = x_opt[2:6]

    model_ivs = Vector{Float64}(undef, n)
    for i in 1:n
        log_dte       = log(max(dtes[i], 1.0))
        log_m         = log(moneyness[i])
        model_ivs[i]  = sqrt(max(θ_base * eval_psi(β, log_dte, log_m), 1e-10))
    end

    rmse = sqrt(mean((model_ivs .- ivs).^2))
    return σ_base, β, rmse, model_ivs
end

println("Fitting ψ surface...")
σ_base, β, rmse, _ = fit_psi(fit_data)
@printf("  RMSE = %.2f%% IV\n", rmse * 100)

# ============================================================================
# ATM data for plot
# ============================================================================

# Market ATM points (from fit_data; excludes 0DTE)
atm_dtes       = Int[]
atm_ivs_market = Float64[]
for dte in sort(unique(fit_data.DTE))
    slice = fit_data[(fit_data.DTE .== dte) .& (abs.(fit_data.Moneyness .- 1.0) .< 0.03), :]
    if nrow(slice) > 0
        push!(atm_dtes, dte)
        push!(atm_ivs_market, mean(slice.IV) * 100)
    end
end

# Continuous model curve
dte_range    = range(1, 50, length=200)
iv_atm_curve = [σ_base * sqrt(exp(β[1]*log(d) + β[5]*log(d)^2)) * 100 for d in dte_range]

# ============================================================================
# Publication-quality plot — iteration 6
# ============================================================================

# Professional color palette (avoid Plots.jl defaults)
col_market = RGB(0.13, 0.35, 0.63)   # deep navy blue
col_model  = RGB(0.80, 0.12, 0.12)   # crimson red
col_grid   = RGB(0.82, 0.82, 0.82)   # light gray for grid

# Model curve from DTE=2 (first fitted DTE) to 47
dte_range_plot = range(2.0, 47.0, length=300)
iv_curve_plot  = [σ_base * sqrt(exp(β[1]*log(d) + β[5]*log(d)^2))*100 for d in dte_range_plot]

# Find the trough DTE for annotation
trough_idx = argmin(iv_curve_plot)
trough_dte = dte_range_plot[trough_idx]
trough_iv  = iv_curve_plot[trough_idx]

# Axis limits
x_min = 0.0
x_max = 49.0
y_min = 29.0
y_max = 43.0

# Grid reference positions (skip x=0 to avoid left-spine doubling)
ytick_vals = 30:2:42
xtick_vals = [0, 10, 20, 30, 40, 46]
xgrid_vals = [10, 20, 30, 40, 46]   # omit 0 to avoid doubling the left spine

fig = plot(
    # No title — caption goes in LaTeX
    size        = (600, 400),
    dpi         = 300,
    fontfamily  = "helvetica",
    # Axis labels
    xlabel      = "Days to Expiration",
    ylabel      = "ATM Implied Volatility (%)",
    xlabelfontsize = 11,
    ylabelfontsize = 11,
    tickfontsize   = 10,
    # Explicit tick positions
    xticks = (xtick_vals, string.(xtick_vals)),
    yticks = (collect(ytick_vals), string.(collect(ytick_vals))),
    # Axis limits
    xlims  = (x_min, x_max),
    ylims  = (y_min, y_max),
    # Grid off — drawn manually to control appearance precisely
    grid   = false,
    # Background
    background_color        = :white,
    background_color_inside = :white,
    # Open-box frame (no top/right spines)
    framestyle = :axes,
    # Legend: bottom-right is empty space below the rising curve tail
    legend              = :bottomright,
    legendfontsize      = 9,
    legendframestyle    = :none,
    legend_background_color = RGBA(1, 1, 1, 0),
    # Margins — extra right margin for readability
    left_margin   = 5mm,
    bottom_margin = 4mm,
    right_margin  = 5mm,
    top_margin    = 4mm,
)

# Horizontal grid lines (behind all data)
for y in ytick_vals
    hline!(fig, [Float64(y)]; color=col_grid, linewidth=0.6, label=nothing)
end

# Vertical grid lines (skip x=0 — would double the left spine)
for x in xgrid_vals
    vline!(fig, [Float64(x)]; color=col_grid, linewidth=0.6, label=nothing)
end

# Model curve
plot!(fig, collect(dte_range_plot), iv_curve_plot;
      label     = "Model",
      linewidth = 2.3,
      color     = col_model,
      linestyle = :solid)

# Market ATM scatter
scatter!(fig, atm_dtes, atm_ivs_market;
         label             = "Market ATM",
         marker            = :circle,
         markersize        = 7,
         color             = col_market,
         markerstrokecolor = :black,
         markerstrokewidth = 0.8)

out_path = joinpath(PLOT_DIR, "atm_term_structure.pdf")
savefig(fig, out_path)
println("Saved → $out_path")
@printf("  Trough: DTE=%.0f, ATM IV=%.1f%%\n", trough_dte, trough_iv)
