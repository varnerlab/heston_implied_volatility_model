"""
Pricing-error figure: model_price − market_mid for SPY, NVDA, LLY.

Loads the JLD2 cache produced by `calibrate_ladders_per_ticker_nn.jl`, prices
each contract via CRR American (n_steps=200, r=4.5%, q=0) using each of:
  - polynomial 5β baseline
  - sector NN
  - per-ticker NN
  - market-published IV (sanity reference)
and plots the dollar pricing error vs moneyness for one representative DTE
per ticker. Designed to load the trained models and re-render in seconds —
no retraining.

Run:
    julia --project=. examples/price_error_figure.jl
"""

using CSV
using DataFrames
using Flux
using JLD2
using Plots
using Plots.PlotMeasures
using Printf
using Statistics

using HestonIV  # for crr_american_price

const LADDER_DIR    = joinpath(@__DIR__, "..", "data", "ladder")
const PLOT_DIR      = joinpath(@__DIR__, "..", "figures")
const CACHE_PATH    = joinpath(PLOT_DIR, "calibrate_ladders_per_ticker_nn_cache.jld2")
const PAPER_FIG_DIR = abspath(joinpath(@__DIR__, "..", "..", "paper", "sections", "figures"))

const PANEL_TICKERS = ["SPY", "NVDA", "LLY", "GS"]
# Risk-free rate: ~3-month T-bill area as of the late-April 2026 capture window.
const R_FREE  = 0.0425
const N_STEPS = 200

# Continuous dividend yield by ticker. Set to zero by default: empirically the
# market-IV → CRR sanity check is closest to zero with q = 0, which matches the
# convention yfinance appears to use when publishing implied volatilities.
# This is also defensible on dividend-timing grounds — ex-div dates are
# discrete events, and over the late-April 2026 capture window neither SPY
# (next ex-div mid-June) nor NVDA has an ex-div inside the contract life.
# LLY's mid-May ex-div *does* fall inside its 29-day window, so a small q
# could be passed there; left at zero here for figure consistency.
const DIVIDEND_YIELD = Dict{String,Float64}(
    "SPY"  => 0.0,
    "NVDA" => 0.0,
    "LLY"  => 0.0,
)
default_q(t) = get(DIVIDEND_YIELD, t, 0.0)

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

isfile(CACHE_PATH) || error(
    "Cache missing at $(CACHE_PATH).\n" *
    "Run `julia --project=. examples/calibrate_ladders_per_ticker_nn.jl --retrain` first.")

# ============================================================================
# Step 1: Load the same ladder corpus the calibration script trained on
# ============================================================================

function load_ladder(filepath)
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

function load_all_ladders(dir)
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
        df = load_ladder(f)
        nrow(df) > 0 && push!(frames, df)
    end
    return vcat(frames...)
end

println("Loading ladder data...")
all_data = load_all_ladders(LADDER_DIR)
all_data[!, :sector] = [get(SECTORS, t, "Other") for t in all_data.ticker]
println("  $(nrow(all_data)) observations across $(length(unique(all_data.ticker))) tickers")

# Standardisation constants — identical recipe to the trained NNs
const MU_DTE    = mean(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const SIGMA_DTE = std(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const MU_M      = mean(log.(Float64.(all_data.moneyness)))
const SIGMA_M   = std(log.(Float64.(all_data.moneyness)))

# ============================================================================
# Step 2: Restore trained models from the JLD2 cache
# ============================================================================

build_psi_nn(n_obs::Integer, threshold::Integer) = n_obs >= threshold ?
    Chain(Dense(2 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 1)) :
    Chain(Dense(2 => 8,  tanh), Dense(8  => 8,  tanh), Dense(8  => 1))

println("Loading trained models from $(basename(CACHE_PATH))...")
cache = JLD2.load(CACHE_PATH)

sector_models = Dict{String,Any}()
for (s, p) in cache["sector_payload"]
    psi_nn = build_psi_nn(p.n_obs, 2000)
    Flux.loadmodel!(psi_nn, p.state)
    ticker_idx = Dict(t => i for (i, t) in enumerate(p.tickers))
    sector_models[s] = (psi_nn = psi_nn, log_theta = p.log_theta, ticker_idx = ticker_idx)
end

per_ticker_models = Dict{String,Any}()
for (t, p) in cache["per_ticker_payload"]
    psi_nn = build_psi_nn(p.n_obs, 5000)
    Flux.loadmodel!(psi_nn, p.state)
    per_ticker_models[t] = (psi_nn = psi_nn, log_theta = p.log_theta)
end

POLY_BETA  = cache["poly_beta"]
POLY_THETA = Dict(cache["poly_theta_pairs"])

# ============================================================================
# Step 3: IV evaluators (single-contract scalar) and CRR pricer wrapper
# ============================================================================

eval_psi_poly(beta, ld, lm) =
    exp(beta[1]*ld + beta[2]*lm + beta[3]*ld*lm + beta[4]*lm^2 + beta[5]*ld^2)

function poly_iv(t::String, K::Float64, S::Float64, dte_days)
    theta_t = POLY_THETA[t]
    ld = log(max(Float64(dte_days), 1.0))
    lm = log(K / S)
    return sqrt(max(theta_t * eval_psi_poly(POLY_BETA, ld, lm), 1e-10))
end

function _stdize_pair(K::Float64, S::Float64, dte_days)
    log_dte_s = Float32((log(max(Float64(dte_days), 1.0)) - MU_DTE) / SIGMA_DTE)
    log_m_s   = Float32((log(K / S) - MU_M) / SIGMA_M)
    return log_dte_s, log_m_s
end

function sector_iv(t::String, K::Float64, S::Float64, dte_days)
    s = get(SECTORS, t, "Other")
    sm = sector_models[s]
    log_theta_t = sm.log_theta[sm.ticker_idx[t]]
    ld_s, lm_s = _stdize_pair(K, S, dte_days)
    x = reshape(Float32[ld_s, lm_s], 2, 1)
    log_psi = first(sm.psi_nn(x))
    return Float64(exp(0.5f0 * (log_theta_t + log_psi)))
end

function per_ticker_iv(t::String, K::Float64, S::Float64, dte_days)
    pm = per_ticker_models[t]
    log_theta = pm.log_theta[1]
    ld_s, lm_s = _stdize_pair(K, S, dte_days)
    x = reshape(Float32[ld_s, lm_s], 2, 1)
    log_psi = first(pm.psi_nn(x))
    return Float64(exp(0.5f0 * (log_theta + log_psi)))
end

function american_price(S::Float64, K::Float64, σ::Float64, dte_days,
                        otype::Symbol, q::Float64)
    T_yrs = Float64(dte_days) / 365.0
    return crr_american_price(S, K, σ, R_FREE, T_yrs, N_STEPS, otype; q=q)
end

# ============================================================================
# Step 4: Build the figure
# ============================================================================

const COL_POLY = RGB(0.88, 0.55, 0.10)
const COL_SEC  = RGB(0.50, 0.50, 0.50)
const COL_PT   = RGB(0.00, 0.00, 0.00)
const COL_MKT  = RGBA(0.30, 0.55, 0.30, 0.55)

p_panels = Any[]
mae_summary = NamedTuple[]

for (k, t) in enumerate(PANEL_TICKERS)
    haskey(per_ticker_models, t) ||
        error("Cache has no per-ticker NN for $t — qualified set may have changed.")

    slice = all_data[all_data.ticker .== t, :]
    # Restrict to the most recent capture date so the slice has a single underlying
    # spot — pooling across 15 capture dates would otherwise mix contracts whose
    # actual_dte matches but whose expiration date and spot differ by days/dollars.
    latest_session = maximum(slice.und_session_date)
    slice = slice[slice.und_session_date .== latest_session, :]
    avail_dtes = sort(unique(slice.actual_dte))
    target_dte = avail_dtes[max(1, length(avail_dtes) ÷ 2)]
    dte_slice = slice[slice.actual_dte .== target_dte, :]
    isempty(dte_slice) && error("$t has no contracts at DTE $target_dte on $latest_session.")
    sector = get(SECTORS, t, "Other")

    q_t = default_q(t)
    rows = NamedTuple[]
    for r in eachrow(dte_slice)
        S = Float64(r.S)
        K = Float64(r.strike)
        otype = Symbol(r.type)
        mid = ismissing(r.mid) ? NaN : Float64(r.mid)
        bid = ismissing(r.bid) ? NaN : Float64(r.bid)
        ask = ismissing(r.ask) ? NaN : Float64(r.ask)
        (isnan(mid) || mid <= 0.0) && continue
        (isnan(bid) || isnan(ask) || ask < bid) && continue

        σ_poly = poly_iv(t, K, S, r.actual_dte)
        σ_sec  = sector_iv(t, K, S, r.actual_dte)
        σ_pt   = per_ticker_iv(t, K, S, r.actual_dte)
        σ_mkt  = Float64(r.implied_vol)

        push!(rows, (
            moneyness   = K / S,
            otype       = otype,
            mid         = mid,
            half_spread = max((ask - bid) / 2, 0.0),
            err_poly    = american_price(S, K, σ_poly, r.actual_dte, otype, q_t) - mid,
            err_sec     = american_price(S, K, σ_sec,  r.actual_dte, otype, q_t) - mid,
            err_pt      = american_price(S, K, σ_pt,   r.actual_dte, otype, q_t) - mid,
            err_mkt     = american_price(S, K, σ_mkt,  r.actual_dte, otype, q_t) - mid,
        ))
    end
    cdf = sort!(DataFrame(rows), :moneyness)
    calls = cdf[cdf.otype .== :call, :]
    puts  = cdf[cdf.otype .== :put,  :]

    inside(x, hs) = abs(x) <= hs
    pct_inside(col) = 100 * mean(inside.(cdf[!, col], cdf.half_spread))

    mae_poly = median(abs.(cdf.err_poly))
    mae_sec  = median(abs.(cdf.err_sec))
    mae_pt   = median(abs.(cdf.err_pt))
    mae_mkt  = median(abs.(cdf.err_mkt))
    push!(mae_summary, (ticker=t, dte=target_dte, n=nrow(cdf),
                        poly=mae_poly, sec=mae_sec, pt=mae_pt, mkt=mae_mkt,
                        in_poly=pct_inside(:err_poly),
                        in_sec=pct_inside(:err_sec),
                        in_pt=pct_inside(:err_pt),
                        in_mkt=pct_inside(:err_mkt),
                        median_half_spread=median(cdf.half_spread)))

    show_legend = (k == 1)

    p = plot(title  = "$t — $sector   (DTE = $target_dte)",
             xlabel = "Moneyness  K/S",
             ylabel = "Pricing error (\$, model − market mid)",
             legend = show_legend ? :topright : false,
             legendfontsize = 7,
             titlefontsize  = 11,
             guidefontsize  = 10,
             tickfontsize   = 9,
             framestyle = :box,
             grid = true, gridalpha = 0.25,
             foreground_color_grid = :gray,
             background_color = :white,
             xlims = (0.83, 1.17),
             left_margin = 4mm, right_margin = 2mm,
             top_margin = 2mm, bottom_margin = 4mm)

    # Per-K/S local ±½ bid–ask spread ribbon. We bin contracts into ~12
    # moneyness bins and take the median half-spread in each bin, then
    # interpolate onto a fine grid so the ribbon traces the actual quote
    # width at each strike (which typically widens at deep ITM/OTM wings).
    # This matches what the "% inside spread" metric in the printed table
    # uses on a per-contract basis.
    let nbin = 12
        m_min, m_max = extrema(cdf.moneyness)
        edges = range(m_min, m_max, length = nbin + 1)
        centers = [(edges[i] + edges[i+1]) / 2 for i in 1:nbin]
        bin_hs = Float64[]
        for i in 1:nbin
            in_bin = (cdf.moneyness .>= edges[i]) .& (cdf.moneyness .<= edges[i+1])
            push!(bin_hs, sum(in_bin) > 0 ? median(cdf.half_spread[in_bin]) : NaN)
        end
        keep = .!isnan.(bin_hs)
        if count(keep) >= 2
            xs = centers[keep]
            ys = bin_hs[keep]
            # Linear interpolation onto a fine moneyness grid
            m_grid = collect(range(xs[1], xs[end], length = 200))
            ribbon_hs = similar(m_grid)
            for (j, m) in enumerate(m_grid)
                k1 = searchsortedfirst(xs, m)
                k1 = clamp(k1, 2, length(xs))
                w = (m - xs[k1-1]) / (xs[k1] - xs[k1-1] + eps())
                ribbon_hs[j] = (1 - w) * ys[k1-1] + w * ys[k1]
            end
            med_hs_overall = median(cdf.half_spread)
            plot!(p, m_grid, zeros(length(m_grid)),
                  ribbon = (ribbon_hs, ribbon_hs),
                  fillalpha = 0.20, color = :gray, lw = 0,
                  label = show_legend ? @sprintf("±½ spread (median \$%.2f)", med_hs_overall) : "")
        end
    end

    hline!(p, [0.0], color = :black, ls = :solid, lw = 0.8, alpha = 0.7, label = "")
    vline!(p, [1.0], color = :gray,  ls = :dash,  lw = 0.8, alpha = 0.30, label = "")

    # Sanity reference: market IV → CRR (split by type so the line stays monotonic).
    if nrow(calls) > 0
        plot!(p, calls.moneyness, calls.err_mkt,
              label = show_legend ? "Market IV → CRR" : "",
              color = COL_MKT, lw = 1.2, ls = :dashdot, alpha = 0.85)
    end
    if nrow(puts) > 0
        plot!(p, puts.moneyness, puts.err_mkt,
              label = "", color = COL_MKT, lw = 1.2, ls = :dashdot, alpha = 0.85)
    end

    # Three model error series — calls and puts share color, marker shape distinguishes them.
    for (label, ydata, color) in (
            ("Polynomial (5β)", :err_poly, COL_POLY),
            ("Sector NN",       :err_sec,  COL_SEC),
            ("Per-ticker NN",   :err_pt,   COL_PT))
        if nrow(calls) > 0
            scatter!(p, calls.moneyness, calls[!, ydata],
                     label = show_legend ? label : "",
                     color = color, marker = :circle, ms = 4, msw = 0.0, alpha = 0.85)
        end
        if nrow(puts) > 0
            scatter!(p, puts.moneyness, puts[!, ydata],
                     label = "",
                     color = color, marker = :diamond, ms = 4, msw = 0.0, alpha = 0.85)
        end
    end

    if show_legend
        scatter!(p, Float64[], Float64[],
                 label = "Calls (●)",  color = :gray, marker = :circle,  ms = 4, msw = 0.0)
        scatter!(p, Float64[], Float64[],
                 label = "Puts (◆)",   color = :gray, marker = :diamond, ms = 4, msw = 0.0)
    end

    push!(p_panels, p)
end

p_fig = plot(p_panels...,
             layout = (2, 2),
             size = (1200, 900),
             dpi = 200,
             left_margin   = 6mm,
             right_margin  = 4mm,
             bottom_margin = 5mm,
             top_margin    = 4mm)

mkpath(PLOT_DIR)
out_pdf = joinpath(PLOT_DIR, "ladder_price_error_three_tickers.pdf")
out_png = joinpath(PLOT_DIR, "ladder_price_error_three_tickers.png")
savefig(p_fig, out_pdf)
savefig(p_fig, out_png)

mkpath(PAPER_FIG_DIR)
cp(out_pdf, joinpath(PAPER_FIG_DIR, "ladder_price_error_three_tickers.pdf"); force=true)
println("[promote] copied ladder_price_error_three_tickers.pdf -> paper/sections/figures/")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^78)
println("  PRICING ERROR PER CONTRACT — \$ medians and % inside the bid–ask spread")
println("="^78)
@printf("  %-5s %-4s %4s  %4s |   %5s   %5s   %5s   %5s\n",
        "tkr", "DTE", "N", "½sp", "poly", "sec", "p-tkr", "mkt-IV")
@printf("  %-5s %-4s %4s  %4s |   %5s   %5s   %5s   %5s\n",
        "", "", "", "(\$)", "Med|e|", "Med|e|", "Med|e|", "Med|e|")
println("  " * "-"^76)
for r in mae_summary
    @printf("  %-5s %-4d %4d  %4.2f |   %5.2f   %5.2f   %5.2f   %5.2f\n",
            r.ticker, r.dte, r.n, r.median_half_spread,
            r.poly, r.sec, r.pt, r.mkt)
    @printf("  %-5s %-4s %4s  %4s |   %4.0f%%   %4.0f%%   %4.0f%%   %4.0f%%   (inside ±½ spread)\n",
            "", "", "", "", r.in_poly, r.in_sec, r.in_pt, r.in_mkt)
end
println()
println("  Reading guide:")
println("    • Inside ±½ bid–ask spread = error indistinguishable from market quote.")
println("    • A market-maker would not requote outside the spread, so >1 spread = real miss.")
println("    • Mkt-IV column re-prices yfinance's own IV through CRR (sanity check).")
println()
println("  Pricing assumptions: CRR American, n_steps=$N_STEPS, r=$R_FREE")
println("  Per-ticker dividend yield (continuous):")
for t in PANEL_TICKERS
    @printf("    %-5s  q = %.4f\n", t, default_q(t))
end
