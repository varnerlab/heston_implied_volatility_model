"""
IV Smile Fit — All Tickers (publication figure)

Loads option chains for NVDA, AMD, MU, INTC. Calibrates the ψ smile function
(β₁ fixed, fit θ_base, β₂, β₃, β₄) on NVDA, then re-fits only θ_base per
ticker with shared β. Plots market scatter + model line by moneyness.
"""

using CSV
using TOML
using DataFrames
using Statistics
using Optim
using Plots
using Printf

gr()

# ============================================================================
# Paths
# ============================================================================

const DATA_DIR  = joinpath(@__DIR__, "..", "data", "options")
const PLOT_DIR  = joinpath(@__DIR__, "..", "figures")
mkpath(PLOT_DIR)

const TICKERS   = ["nvda", "amd", "mu", "intc"]
const LABELS    = ["NVDA", "AMD", "MU", "INTC"]

# ============================================================================
# Data loading
# ============================================================================

struct TickerData
    df::DataFrame
    S::Float64
    DTE::Int
    atm_iv::Float64
end

function load_ticker(ticker::String)::TickerData
    csv_path  = joinpath(DATA_DIR, "$(ticker).csv")
    toml_path = joinpath(DATA_DIR, "$(ticker).toml")

    df   = CSV.read(csv_path, DataFrame)
    meta = TOML.parsefile(toml_path)["metadata"]

    S      = parse(Float64, meta["underlying_share_price_mid"])
    dte    = parse(Int,     meta["DTE"])
    atm_iv = parse(Float64, meta["atm_IV"])

    # Compute moneyness from Strike and S (override CSV Moneyness column)
    df[!, :KoverS] = df.Strike ./ S

    # Filter: valid IV, moneyness band
    mask = (df.IV .> 0.01) .& (df.IV .< 2.0) .&
           (df.KoverS .>= 0.75) .& (df.KoverS .<= 1.25)
    df = df[mask, :]

    return TickerData(df, S, dte, atm_iv)
end

# ============================================================================
# ψ smile model:  σ_model(m) = √(θ_base · ψ(m))
#   ψ(m) = exp(β₁·ln(DTE) + β₂·ln(m) + β₃·ln(DTE)·ln(m) + β₄·(ln(m))²)
# β₁ = 0.20 fixed (term structure)
# ============================================================================

const β₁_FIXED = 0.20

function eval_model_iv(θ_base::Float64, β::Vector{Float64},
                        log_dte::Float64, log_m::Float64)::Float64
    ψ = exp(β₁_FIXED * log_dte + β[1] * log_m + β[2] * log_dte * log_m + β[3] * log_m^2)
    return sqrt(max(θ_base * ψ, 1e-10))
end

# --- Stage 1: calibrate on NVDA (fit θ_base, β₂, β₃, β₄) ---
function calibrate_nvda(td::TickerData)
    df      = td.df
    log_dte = log(max(Float64(td.DTE), 1.0))
    ivs     = df.IV
    log_ms  = log.(df.KoverS)
    n       = length(ivs)

    # x = [log(θ_base), β₂, β₃, β₄]
    x0 = [log(td.atm_iv^2), -0.5, -0.05, 1.5]

    function objective(x)
        θ_base = exp(x[1])
        β = x[2:4]
        err = 0.0
        for i in 1:n
            σ_m = eval_model_iv(θ_base, β, log_dte, log_ms[i])
            err += (σ_m - ivs[i])^2
        end
        return err / n
    end

    res = optimize(objective, x0, NelderMead(),
                   Optim.Options(iterations=50000, g_tol=1e-14))
    res = optimize(objective, Optim.minimizer(res), NelderMead(),
                   Optim.Options(iterations=50000, g_tol=1e-14))
    xopt = Optim.minimizer(res)
    θ_base = exp(xopt[1])
    β = xopt[2:4]
    rmse = sqrt(Optim.minimum(res))
    return θ_base, β, rmse
end

# --- Stage 2: re-fit only θ_base for a ticker given fixed β ---
function fit_theta_base(td::TickerData, β::Vector{Float64})
    df      = td.df
    log_dte = log(max(Float64(td.DTE), 1.0))
    ivs     = df.IV
    log_ms  = log.(df.KoverS)
    n       = length(ivs)

    function objective(x)
        θ_base = exp(x[1])
        err = 0.0
        for i in 1:n
            σ_m = eval_model_iv(θ_base, β, log_dte, log_ms[i])
            err += (σ_m - ivs[i])^2
        end
        return err / n
    end

    x0  = [log(td.atm_iv^2)]
    res = optimize(objective, x0, NelderMead(),
                   Optim.Options(iterations=20000, g_tol=1e-14))
    θ_base = exp(Optim.minimizer(res)[1])
    rmse   = sqrt(Optim.minimum(res))
    return θ_base, rmse
end

# ============================================================================
# Load data + calibrate
# ============================================================================

println("Loading option chains...")
ticker_data = [load_ticker(t) for t in TICKERS]

for (t, td) in zip(TICKERS, ticker_data)
    @printf("  %-4s  S=%.2f  DTE=%d  n=%d  ATM_IV=%.1f%%\n",
            uppercase(t), td.S, td.DTE, nrow(td.df), td.atm_iv*100)
end

println("\nStage 1: calibrating β on NVDA...")
nvda_td       = ticker_data[1]
θ_nvda, β_shared, rmse_nvda = calibrate_nvda(nvda_td)
@printf("  NVDA  θ_base=%.4f  β=[%.3f, %.3f, %.3f]  RMSE=%.2f%%\n",
        θ_nvda, β_shared[1], β_shared[2], β_shared[3], rmse_nvda*100)

println("\nStage 2: fitting θ_base per ticker (shared β)...")
θ_bases = Vector{Float64}(undef, length(TICKERS))
θ_bases[1] = θ_nvda

for i in 2:length(TICKERS)
    θ_i, rmse_i = fit_theta_base(ticker_data[i], β_shared)
    θ_bases[i] = θ_i
    @printf("  %-4s  θ_base=%.4f  RMSE=%.2f%%\n",
            uppercase(TICKERS[i]), θ_i, rmse_i*100)
end

# ============================================================================
# Colors  (navy, crimson, forest-green, dark-orange)
# ============================================================================

const COLORS = [
    RGB(0.122, 0.235, 0.498),   # navy
    RGB(0.698, 0.094, 0.122),   # crimson
    RGB(0.133, 0.420, 0.196),   # forest green
    RGB(0.800, 0.400, 0.000),   # dark orange
]

# ============================================================================
# Build figure
# ============================================================================

fig = plot(
    xlabel          = "Moneyness (K/S)",
    ylabel          = "Implied Volatility (%)",
    # Place legend in upper-left with slight offset to clear data
    legend          = (0.015, 0.975),
    legendfontsize  = 9,
    legendtitlefontsize = 9,
    tickfontsize    = 10,
    guidefontsize   = 11,
    size            = (700, 450),
    dpi             = 300,
    fontfamily      = "Helvetica",
    background_color = :white,
    grid            = true,
    gridcolor       = RGB(0.88, 0.88, 0.88),
    gridlinewidth   = 0.6,
    gridalpha       = 1.0,
    framestyle      = :box,
    legend_background_color = RGBA(1.0, 1.0, 1.0, 0.75),
    legend_foreground_color = RGB(0.75, 0.75, 0.75),
    foreground_color_border = RGB(0.25, 0.25, 0.25),
    foreground_color_axis   = RGB(0.25, 0.25, 0.25),
    foreground_color_text   = RGB(0.10, 0.10, 0.10),
    margin          = 4Plots.mm,
    ylims           = (33, 88),
    yticks          = ([40, 50, 60, 70, 80], ["40", "50", "60", "70", "80"]),
    xticks          = ([0.80, 0.90, 1.00, 1.10, 1.20],
                       ["0.80", "0.90", "1.00", "1.10", "1.20"]),
)

# ATM reference line
vline!(fig, [1.0],
       label   = nothing,
       ls      = :dash,
       color   = RGB(0.40, 0.40, 0.40),
       lw      = 1.2,
       alpha   = 0.85)

for i in eachindex(TICKERS)
    td      = ticker_data[i]
    c       = COLORS[i]
    label   = LABELS[i]
    log_dte = log(max(Float64(td.DTE), 1.0))
    θ_base  = θ_bases[i]

    # Market scatter
    scatter!(fig,
             td.df.KoverS, td.df.IV .* 100,
             label              = label,
             marker             = :circle,
             ms                 = 3.5,
             color              = c,
             markerstrokecolor  = RGB(0.05, 0.05, 0.05),
             markerstrokewidth  = 0.5,
             alpha              = 0.70)

    # Model curve — clipped to ticker's data moneyness range (±2% padding)
    m_lo = max(minimum(td.df.KoverS) - 0.02, 0.75)
    m_hi = min(maximum(td.df.KoverS) + 0.02, 1.25)
    m_grid     = range(m_lo, m_hi, length=300)
    log_m_grid = log.(collect(m_grid))
    iv_curve   = [eval_model_iv(θ_base, β_shared, log_dte, log_m_grid[j]) * 100
                  for j in eachindex(m_grid)]
    plot!(fig, collect(m_grid), iv_curve,
          label  = nothing,
          lw     = 2.0,
          color  = c,
          alpha  = 0.95)
end

savefig(fig, joinpath(PLOT_DIR, "iv_smile_all_tickers.pdf"))
println("\nSaved → code/figures/iv_smile_all_tickers.pdf")

include(joinpath(@__DIR__, "..", "scripts", "promote_figures.jl"))
promote_figures()
