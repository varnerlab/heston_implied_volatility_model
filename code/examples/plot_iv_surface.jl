"""
IV Surface — Market vs Model (publication figure)

Loads NVDA option chains across 7 DTEs, fits the ψ surface (excluding 0DTE),
and produces a single publication-quality IV surface plot for the paper.
"""

using CSV
using TOML
using DataFrames
using Statistics
using Optim
using Plots
using Printf

gr()   # explicit GR backend for consistent rendering

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

all_data = load_all_slices()

# Filter for fitting (exclude 0DTE, use both calls and puts in moneyness band)
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

# All data (including 0DTE) for scatter overlay
scatter_data = all_data[(all_data.Moneyness .>= 0.90) .&
                         (all_data.Moneyness .<= 1.10) .&
                         (all_data.IV .> 0.0) .&
                         (all_data.Volume .> 0), :]

# ============================================================================
# ψ surface fitting
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

    model_ivs = Vector{Float64}(undef, n)
    for i in 1:n
        log_dte = log(max(dtes[i], 1.0))
        log_m = log(moneyness[i])
        model_ivs[i] = sqrt(max(θ_base * eval_psi(β, log_dte, log_m), 1e-10))
    end

    rmse = sqrt(mean((model_ivs .- ivs).^2))
    return σ_base, β, rmse, model_ivs
end

println("Fitting ψ surface...")
σ_base, β, rmse, _ = fit_psi(fit_data)
@printf("  σ_base = %.2f%%, RMSE = %.2f%% IV\n", σ_base * 100, rmse * 100)

# ============================================================================
# Publication-quality plot
# ============================================================================

# Perceptually uniform color palette — sequential from blue-green to red-purple
# Maps DTEs: 0, 2, 4, 7, 14, 25, 46 → 7 colors evenly spaced in viridis
# Use ColorSchemes-style manual palette that prints in grayscale
const ALL_DTES = [0, 2, 4, 7, 14, 25, 46]

# Cubehelix-inspired palette: distinct in both color and brightness
# From light (short DTE) to dark (long DTE) — graceful grayscale degradation
const DTE_PALETTE = Dict(
    0  => RGB(0.835, 0.243, 0.310),   # warm red     — 0DTE (distinct, excluded from fit)
    2  => RGB(0.918, 0.502, 0.176),   # orange
    4  => RGB(0.769, 0.694, 0.157),   # yellow-green
    7  => RGB(0.259, 0.651, 0.365),   # green
    14 => RGB(0.157, 0.596, 0.710),   # teal-blue
    25 => RGB(0.231, 0.376, 0.753),   # blue
    46 => RGB(0.459, 0.188, 0.635),   # purple
)

# NeurIPS single-column: ~252pt wide → ~3.5in. At 300dpi → 1050px.
# Use 700×440 with dpi=300 for a compact, professional figure.
fig = plot(
    xlabel="Moneyness (K/S)",
    ylabel="Implied Volatility (%)",
    legend=:outerright,
    legendtitle="DTE",
    legendtitlefontcolor=RGB(0.15, 0.15, 0.15),
    legendfontcolor=RGB(0.15, 0.15, 0.15),
    legendfontsize=8,
    tickfontsize=9,
    guidefontsize=11,
    titlefontsize=10,
    legendtitlefontsize=8,
    size=(820, 440),
    dpi=300,
    fontfamily="Helvetica",
    background_color=:white,
    grid=true,
    gridcolor=RGB(0.88, 0.88, 0.88),
    gridlinewidth=0.6,
    gridalpha=1.0,
    framestyle=:box,
    legend_background_color=:transparent,
    legend_foreground_color=:transparent,
    minorticks=false,
    foreground_color_border=RGB(0.25, 0.25, 0.25),
    foreground_color_axis=RGB(0.25, 0.25, 0.25),
    foreground_color_text=RGB(0.1, 0.1, 0.1),
    margin=4Plots.mm,
    right_margin=12Plots.mm,
    ylims=(18, 63),
    xticks=([0.90, 0.95, 1.00, 1.05, 1.10],
            ["0.90", "0.95", "1.00", "1.05", "1.10"]),
    yticks=20:10:60,
)

# ATM reference line (behind data)
vline!(fig, [1.0],
       label=nothing,
       ls=:dash,
       color=RGB(0.5, 0.5, 0.5),
       lw=1.0,
       alpha=0.9)

# Plot each DTE: scatter market data + model curve
# One legend entry per DTE (color shared between marker + line)
m_curve = range(0.90, 1.10, length=200)

for dte in sort(unique(scatter_data.DTE))
    slice = scatter_data[scatter_data.DTE .== dte, :]
    c = get(DTE_PALETTE, dte, RGB(0.3, 0.3, 0.3))
    label_str = "$(dte)d"

    if dte == 0
        # 0DTE: diamond marker, excluded from fit (noted in caption)
        scatter!(fig,
                 slice.Moneyness, slice.IV .* 100,
                 label=label_str,
                 marker=:diamond,
                 ms=5,
                 color=c,
                 markerstrokecolor=RGB(0.1, 0.1, 0.1),
                 markerstrokewidth=0.5,
                 alpha=0.65)
    else
        # Market scatter — filled circles with thin dark stroke
        scatter!(fig,
                 slice.Moneyness, slice.IV .* 100,
                 label=label_str,
                 marker=:circle,
                 ms=5.5,
                 color=c,
                 markerstrokecolor=RGB(0.08, 0.08, 0.08),
                 markerstrokewidth=0.7,
                 alpha=0.82)

        # Model curve
        log_dte = log(max(Float64(dte), 1.0))
        iv_curve = [σ_base * sqrt(eval_psi(β, log_dte, log(m))) * 100
                    for m in m_curve]
        plot!(fig, collect(m_curve), iv_curve,
              label=nothing,
              lw=2.2,
              color=c,
              alpha=0.92)
    end
end

savefig(fig, joinpath(PLOT_DIR, "iv_surface_market_vs_model.pdf"))
println("Saved → code/figures/iv_surface_market_vs_model.pdf")

include(joinpath(@__DIR__, "..", "scripts", "promote_figures.jl"))
promote_figures()
