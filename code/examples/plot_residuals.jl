"""
IV Residuals by DTE — Publication-quality standalone plot

Loads NVDA cross-sectional option data (7 DTEs: 0, 2, 4, 7, 14, 25, 46),
fits the ψ surface (excluding 0DTE), and plots model–market IV residuals
vs moneyness colored by DTE.
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
# Load data
# ============================================================================

function load_all_slices()
    frames = DataFrame[]
    for f in DTE_FILES
        csv_path  = joinpath(DATA_DIR, "$(f).csv")
        toml_path = joinpath(DATA_DIR, "$(f).toml")
        df   = CSV.read(csv_path, DataFrame)
        meta = TOML.parsefile(toml_path)["metadata"]
        S    = parse(Float64, meta["underlying_share_price"])
        dte  = parse(Int,     meta["DTE"])
        df[!, :DTE]      .= dte
        df[!, :S]        .= S
        df[!, :Moneyness] = df.Strike ./ S
        push!(frames, df)
    end
    return vcat(frames...)
end

all_data = load_all_slices()

# Filter: calls + puts, DTE > 0, moneyness [0.90, 1.10], IV > 0, Volume > 0
calls = all_data[(all_data.Type .== "Call") .& (all_data.DTE .> 0) .&
                 (all_data.Moneyness .>= 0.90) .& (all_data.Moneyness .<= 1.10) .&
                 (all_data.IV .> 0.0) .& (all_data.Volume .> 0), :]
puts  = all_data[(all_data.Type .== "Put")  .& (all_data.DTE .> 0) .&
                 (all_data.Moneyness .>= 0.90) .& (all_data.Moneyness .<= 1.10) .&
                 (all_data.IV .> 0.0) .& (all_data.Volume .> 0), :]
fit_data = vcat(calls, puts)

# ============================================================================
# Fit ψ surface
# ============================================================================

function eval_psi(β::Vector{Float64}, log_dte::Float64, log_m::Float64)::Float64
    return exp(β[1]*log_dte + β[2]*log_m + β[3]*log_dte*log_m +
               β[4]*log_m^2 + β[5]*log_dte^2)
end

function fit_psi(df::DataFrame)
    n         = nrow(df)
    ivs       = df.IV
    dtes      = Float64.(df.DTE)
    moneyness = df.Moneyness

    mid_mask = (dtes .>= 4) .& (dtes .<= 14) .& (abs.(moneyness .- 1.0) .< 0.05)
    σ_mid = any(mid_mask) ? mean(ivs[mid_mask]) : mean(ivs)
    x0 = [log(σ_mid^2), 0.3, -1.0, 0.1, 2.0, -0.15]

    function objective(x)
        θ_base = exp(x[1])
        β = x[2:6]
        err = 0.0
        for i in 1:n
            log_dte = log(max(dtes[i], 1.0))
            log_m   = log(moneyness[i])
            ψ       = eval_psi(β, log_dte, log_m)
            σ_model = sqrt(max(θ_base * ψ, 1e-10))
            err    += (σ_model - ivs[i])^2
        end
        return err / n
    end

    result = optimize(objective, x0, NelderMead(), Optim.Options(iterations=100000, g_tol=1e-14))
    result = optimize(objective, Optim.minimizer(result), NelderMead(), Optim.Options(iterations=100000, g_tol=1e-14))

    x_opt  = Optim.minimizer(result)
    θ_base = exp(x_opt[1])
    σ_base = sqrt(θ_base)
    β      = x_opt[2:6]

    model_ivs = Vector{Float64}(undef, n)
    for i in 1:n
        log_dte      = log(max(dtes[i], 1.0))
        log_m        = log(moneyness[i])
        model_ivs[i] = sqrt(max(θ_base * eval_psi(β, log_dte, log_m), 1e-10))
    end

    rmse = sqrt(mean((model_ivs .- ivs).^2))
    return σ_base, β, rmse, model_ivs
end

println("Fitting ψ surface...")
σ_base, β, rmse, model_ivs = fit_psi(fit_data)
println("  RMSE = $(round(rmse*100, digits=2))% IV")

fit_data[!, :ModelIV]  = model_ivs
fit_data[!, :Residual] = model_ivs .- fit_data.IV

# ============================================================================
# Color palette — Okabe-Ito inspired, accessible, ordered warm→cool by DTE
# ============================================================================

const DTE_COLORS = Dict(
     2 => RGB(0.843, 0.188, 0.122),   # vivid red
     4 => RGB(0.929, 0.694, 0.125),   # bright amber-yellow
     7 => RGB(0.275, 0.733, 0.196),   # lime-green (distinct from teal)
    14 => RGB(0.059, 0.533, 0.451),   # dark teal
    25 => RGB(0.122, 0.471, 0.706),   # steel blue
    46 => RGB(0.576, 0.169, 0.514),   # plum-purple
)

# ============================================================================
# Build residuals plot
# ============================================================================

gr()

# Symmetric y-axis limits — clip extreme outliers, round to nearest 0.5
all_res  = fit_data.Residual .* 100
# Use 98th percentile to avoid extreme outliers dominating axis
ylim_abs = ceil(quantile(abs.(all_res), 0.97) * 2) / 2   # round to nearest 0.5
ylim_abs = max(ylim_abs, 2.5)

# Y-tick positions at round values
ytick_step  = ylim_abs <= 3.0 ? 1.0 : 2.0
yticks_vals = collect(-ylim_abs:ytick_step:ylim_abs)
# Ensure 0 is in ticks
if !(0.0 in yticks_vals)
    push!(yticks_vals, 0.0)
    sort!(yticks_vals)
end

p = plot(
    size                    = (700, 360),
    dpi                     = 300,
    framestyle              = :box,
    background_color        = :white,
    grid                    = false,
    xlims                   = (0.885, 1.115),
    ylims                   = (-ylim_abs - 0.2, ylim_abs + 0.2),
    xticks                  = 0.90:0.05:1.10,
    yticks                  = yticks_vals,
    xlabel                  = "Moneyness (K/S)",
    ylabel                  = "IV Residual (%)",
    tickfontsize            = 10,
    guidefontsize           = 11,
    legendfontsize          = 9,
    legend                  = :bottomleft,
    foreground_color_legend = :white,
    background_color_legend = RGB(1.0, 1.0, 1.0),
    left_margin             = 6mm,
    bottom_margin           = 5mm,
    right_margin            = 6mm,
    top_margin              = 3mm,
    fontfamily              = "Helvetica",
)

# Horizontal grid lines — light for non-zero, prominent for zero
for yv in yticks_vals
    hline!(p, [yv],
           color = yv == 0.0 ? RGB(0.45, 0.45, 0.45) : RGB(0.88, 0.88, 0.88),
           lw    = yv == 0.0 ? 1.0 : 0.6,
           label = nothing)
end

# Scatter residuals by DTE — slightly larger markers
for dte in sort(unique(fit_data.DTE))
    slice = fit_data[fit_data.DTE .== dte, :]
    c     = get(DTE_COLORS, dte, RGB(0.3, 0.3, 0.3))
    scatter!(p, slice.Moneyness, slice.Residual .* 100,
             label             = "$(dte)d",
             marker            = :circle,
             ms                = 5.5,
             color             = c,
             markerstrokecolor = :black,
             markerstrokewidth = 0.6,
             alpha             = 0.72)
end

out = joinpath(PLOT_DIR, "iv_residuals_by_dte.pdf")
savefig(p, out)
println("Saved → $out")

include(joinpath(@__DIR__, "..", "scripts", "promote_figures.jl"))
promote_figures()
