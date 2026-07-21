"""
Render the per-ticker smile panel figure with a black background for slides.

Loads the cached models from `calibrate_ladders_per_ticker_nn.jl` (no retraining)
and writes a presentation-ready PDF to
`paper/sections/figures/presentation/ladder_per_ticker_nn_smile_panels_black_background.pdf`.
"""

using CSV
using DataFrames
using JLD2
using Statistics
using Flux
using Plots
using Plots.PlotMeasures

const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladder")
const PLOT_DIR = joinpath(@__DIR__, "..", "figures")
const CACHE_PATH = joinpath(PLOT_DIR, "calibrate_ladders_per_ticker_nn_cache.jld2")
const PRESENTATION_DIR = abspath(joinpath(@__DIR__, "..", "..", "paper-jcf", "sections",
                                          "figures", "presentation"))
mkpath(PRESENTATION_DIR)

isfile(CACHE_PATH) || error("Cache not found at $CACHE_PATH — run calibrate_ladders_per_ticker_nn.jl first.")

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

# ----------------------------------------------------------------------------
# Reload ladder data (same filters as the canonical script)
# ----------------------------------------------------------------------------
function load_ladder(filepath::String)
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
        (df.actual_dte .> 0),
    :]
    return valid
end

function load_all_ladders(dir::String)
    files = String[]
    for (root, _, fs) in walkdir(dir)
        occursin("VIX-data", root) && continue
        for f in fs
            endswith(f, ".csv") && push!(files, joinpath(root, f))
        end
    end
    frames = DataFrame[]
    for f in files
        df = load_ladder(f)
        nrow(df) > 0 && push!(frames, df)
    end
    return vcat(frames...)
end

println("Loading ladder data...")
all_data = load_all_ladders(LADDER_DIR)
all_data[!, :sector] = [get(SECTORS, t, "Other") for t in all_data.ticker]
tickers = sort(unique(all_data.ticker))

const MU_DTE = mean(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const SIGMA_DTE = std(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const MU_M = mean(log.(Float64.(all_data.moneyness)))
const SIGMA_M = std(log.(Float64.(all_data.moneyness)))

# ----------------------------------------------------------------------------
# Reload trained models from cache
# ----------------------------------------------------------------------------
println("Loading cached models from $(basename(CACHE_PATH))...")
build_psi_nn(n_obs::Integer, threshold::Integer) = n_obs >= threshold ?
    Chain(Dense(2 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 1)) :
    Chain(Dense(2 => 8,  tanh), Dense(8  => 8,  tanh), Dense(8  => 1))

cache = JLD2.load(CACHE_PATH)
sector_models = Dict{String,Any}()
per_ticker_models = Dict{String,Any}()
for (s, p) in cache["sector_payload"]
    psi_nn = build_psi_nn(p.n_obs, 2000)
    Flux.loadmodel!(psi_nn, p.state)
    ticker_idx = Dict(t => i for (i, t) in enumerate(p.tickers))
    sector_models[s] = (model = (psi_nn = psi_nn, log_theta = p.log_theta),
                        tickers = p.tickers, ticker_idx = ticker_idx)
end
for (t, p) in cache["per_ticker_payload"]
    psi_nn = build_psi_nn(p.n_obs, 5000)
    Flux.loadmodel!(psi_nn, p.state)
    per_ticker_models[t] = (model = (psi_nn = psi_nn, log_theta = p.log_theta),)
end
const POLY_BETA  = cache["poly_beta"]
const POLY_THETA = Dict(cache["poly_theta_pairs"])

qualified = Set(keys(per_ticker_models))

# ----------------------------------------------------------------------------
# Grid evaluators (same math as the canonical script)
# ----------------------------------------------------------------------------
eval_psi_poly(beta, ld, lm) =
    exp(beta[1]*ld + beta[2]*lm + beta[3]*ld*lm + beta[4]*lm^2 + beta[5]*ld^2)

function eval_per_ticker_iv_grid(ticker::String, m_range, dte_val::Float64)
    pm = per_ticker_models[ticker]
    psi_nn = pm.model.psi_nn
    log_theta = pm.model.log_theta[1]
    log_dte_s = Float32((log(max(dte_val, 1.0)) - MU_DTE) / SIGMA_DTE)
    log_m_vals = [Float32((log(m) - MU_M) / SIGMA_M) for m in m_range]
    x_grid = hcat(fill(log_dte_s, length(m_range)), log_m_vals)' |> Matrix{Float32}
    log_psi = vec(psi_nn(x_grid))
    ivs = exp.(Float32(0.5) .* (Float32(log_theta) .+ log_psi))
    return Float64.(ivs) .* 100
end

function eval_sector_iv_grid(ticker::String, m_range, dte_val::Float64)
    sector = get(SECTORS, ticker, "Other")
    sm = sector_models[sector]
    psi_nn = sm.model.psi_nn
    tidx = sm.ticker_idx[ticker]
    log_theta_t = sm.model.log_theta[tidx]
    log_dte_s = Float32((log(max(dte_val, 1.0)) - MU_DTE) / SIGMA_DTE)
    log_m_vals = [Float32((log(m) - MU_M) / SIGMA_M) for m in m_range]
    x_grid = hcat(fill(log_dte_s, length(m_range)), log_m_vals)' |> Matrix{Float32}
    log_psi = vec(psi_nn(x_grid))
    ivs = exp.(Float32(0.5) .* (Float32(log_theta_t) .+ log_psi))
    return Float64.(ivs) .* 100
end

function eval_poly_iv_grid(ticker::String, m_range, dte_val::Float64)
    theta_t = POLY_THETA[ticker]
    log_dte = log(max(dte_val, 1.0))
    [sqrt(max(theta_t * eval_psi_poly(POLY_BETA, log_dte, log(m)), 1e-10)) * 100 for m in m_range]
end

# ----------------------------------------------------------------------------
# Dark-theme palette: legible on a black slide background
# ----------------------------------------------------------------------------
const BG          = RGB(0.0, 0.0, 0.0)
const FG          = RGB(0.92, 0.92, 0.92)
const COL_CALL_BG = RGBA(0.45, 0.70, 1.00, 0.75)   # light blue
const COL_PUT_BG  = RGBA(1.00, 0.50, 0.55, 0.75)   # warm pink
const COL_POLY_BG = RGB(1.00, 0.70, 0.20)          # warm orange, dotted
const COL_SEC_BG  = RGB(0.70, 0.70, 0.70)          # light gray, dashed
const COL_PT_BG   = RGB(1.00, 1.00, 1.00)          # white, solid

panel_candidates = ["SPY", "NVDA", "MSFT", "LLY", "GS", "AVGO"]
panel_tickers = [t for t in panel_candidates if t in qualified]

println("Rendering black-background panels for: ", join(panel_tickers, ", "))

p_panels = Any[]
for (k, t) in enumerate(panel_tickers)
    slice = all_data[all_data.ticker .== t, :]
    avail_dtes = sort(unique(slice.actual_dte))
    target_dte = avail_dtes[max(1, length(avail_dtes) ÷ 2)]
    dte_slice = slice[slice.actual_dte .== target_dte, :]
    sector = get(SECTORS, t, "Other")
    show_legend = (k == 1)

    p = plot(title = "$t — $sector   (DTE = $target_dte)",
             xlabel = "Moneyness  K/S",
             ylabel = "Implied Volatility (%)",
             legend = show_legend ? :topright : false,
             legendfontsize = 8,
             titlefontsize = 11,
             guidefontsize = 10,
             tickfontsize = 9,
             framestyle = :box,
             grid = true,
             gridalpha = 0.25,
             foreground_color_grid = FG,
             background_color = BG,
             background_color_inside = BG,
             background_color_outside = BG,
             background_color_legend = BG,
             foreground_color_legend = FG,
             foreground_color_axis = FG,
             foreground_color_border = FG,
             foreground_color_text = FG,
             foreground_color_guide = FG,
             foreground_color_title = FG,
             legendfontcolor = FG,
             xlims = (0.78, 1.22),
             left_margin = 4mm,
             right_margin = 2mm,
             top_margin = 2mm,
             bottom_margin = 4mm)

    calls = dte_slice[dte_slice.type .== "call", :]
    puts  = dte_slice[dte_slice.type .== "put",  :]
    scatter!(p, calls.moneyness, Float64.(calls.implied_vol) .* 100,
             label = show_legend ? "Calls (market)" : "",
             marker = :circle, ms = 3.5, msw = 0.0, color = COL_CALL_BG)
    scatter!(p, puts.moneyness, Float64.(puts.implied_vol) .* 100,
             label = show_legend ? "Puts (market)" : "",
             marker = :diamond, ms = 3.5, msw = 0.0, color = COL_PUT_BG)

    m_grid = collect(range(0.85, 1.15, length = 100))
    poly_curve = eval_poly_iv_grid(t, m_grid, Float64(target_dte))
    sec_curve  = eval_sector_iv_grid(t, m_grid, Float64(target_dte))
    pt_curve   = eval_per_ticker_iv_grid(t, m_grid, Float64(target_dte))

    plot!(p, m_grid, poly_curve,
          label = show_legend ? "Polynomial (5β)" : "",
          lw = 2.0, color = COL_POLY_BG, ls = :dot)
    plot!(p, m_grid, sec_curve,
          label = show_legend ? "Sector NN" : "",
          lw = 2.0, color = COL_SEC_BG, ls = :dash)
    plot!(p, m_grid, pt_curve,
          label = show_legend ? "Per-ticker NN" : "",
          lw = 2.5, color = COL_PT_BG, ls = :solid)
    vline!(p, [1.0], label = "", ls = :dash, color = FG, alpha = 0.30)

    push!(p_panels, p)
end

p2 = plot(p_panels...,
          layout = (2, 3),
          size = (1500, 800),
          dpi = 200,
          background_color = BG,
          background_color_outside = BG,
          left_margin = 6mm,
          right_margin = 4mm,
          bottom_margin = 5mm,
          top_margin = 4mm)

out_pdf = joinpath(PRESENTATION_DIR, "ladder_per_ticker_nn_smile_panels_black_background.pdf")
out_png = joinpath(PRESENTATION_DIR, "ladder_per_ticker_nn_smile_panels_black_background.png")
savefig(p2, out_pdf)
savefig(p2, out_png)
println("\n[saved] $out_pdf")
println("[saved] $out_png")
