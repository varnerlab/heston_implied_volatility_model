"""
Temporal Holdout with Earnings-Aware Calibration: Three-Configuration Comparison

Trains the sector-NN psi model under three configurations on the same temporal
split (train 2026-04-14..04-22, test 04-23..04-24) to measure what an earnings
indicator buys us:

  A. Pooled control: 2 inputs (ln tau, ln m), no exclusion. Reproduces the
     existing temporal_holdout.jl sector-NN number (~13% test RMSE).
  B. Non-earnings baseline: 2 inputs, exclude all train+test rows where
     min(|d2e_self|, |d2e_peer_min|) <= 3. Reference for "what the model
     does on quiet days," with no event mechanism.
  C. Earnings-aware: 4 inputs (ln tau, ln m, d2e_self_clipped,
     d2e_peer_min_clipped), no exclusion. Same train+test as A.

d2e_self: signed days from observation date to nearest earnings print for
the row's ticker (negative = past, positive = upcoming, missing for ETFs).
d2e_peer_min: min |d2e| over same-sector equity peers (excluding self).
For ETFs (no own earnings), peer set = all 28 equities, and d2e_self is
set equal to d2e_peer_min so the feature is informative on event days.

Both d2e features are clipped to +/- 30 days before standardization.
"""

using CSV
using DataFrames
using Statistics
using Flux
using Printf
using Random
using Dates

include(joinpath(@__DIR__, "..", "src", "EarningsCalendar.jl"))
using .EarningsCalendar

# ============================================================================
# Configuration
# ============================================================================

const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladder")
const EARNINGS_CSV = joinpath(@__DIR__, "..", "data", "earnings", "earnings_calendar.csv")
const TRAIN_DAYS = ["options-04-14-2026", "options-04-15-2026",
                    "options-04-16-2026", "options-04-17-2026",
                    "options-04-21-2026", "options-04-22-2026"]
const TEST_DAYS  = ["options-04-23-2026", "options-04-24-2026"]
const EARNINGS_WINDOW = 3      # ±days for in-window exclusion / for sympathy
const D2E_CLIP = 30            # clamp days_to_earnings to this magnitude

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

const ETF_TICKERS = Set([t for (t, s) in SECTORS if s == "ETF"])
const EQUITY_TICKERS = Set([t for t in keys(SECTORS) if !(t in ETF_TICKERS)])

# Map a "options-MM-DD-YYYY" directory name to a Date
function dir_to_date(d::String)
    m = match(r"options-(\d{2})-(\d{2})-(\d{4})", d)
    Date(parse(Int, m.captures[3]), parse(Int, m.captures[1]), parse(Int, m.captures[2]))
end

# ============================================================================
# Data loading
# ============================================================================

function load_ladder(filepath::String, day_date::Date)
    df = CSV.read(filepath, DataFrame)
    ticker = string(df.underlying[1])
    S = df.und_close[1]
    df[!, :ticker] .= ticker
    df[!, :S] .= S
    df[!, :moneyness] = df.strike ./ S
    df[!, :obs_date] .= day_date
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

function load_split(day_dirs::Vector{String})
    frames = DataFrame[]
    for d in day_dirs
        full = joinpath(LADDER_DIR, d)
        day_date = dir_to_date(d)
        for f in readdir(full)
            endswith(f, ".csv") && occursin("_dte_ladder_", f) || continue
            df = load_ladder(joinpath(full, f), day_date)
            nrow(df) > 0 && push!(frames, df)
        end
    end
    out = vcat(frames...)
    out[!, :sector] = [get(SECTORS, t, "Other") for t in out.ticker]
    return out
end

println("Loading earnings calendar from $EARNINGS_CSV ...")
cal = load_earnings(EARNINGS_CSV)
println("  loaded $(length(cal)) tickers with earnings entries")

println("\nLoading train split: $TRAIN_DAYS")
train = load_split(TRAIN_DAYS)
println("  $(nrow(train)) observations across $(length(unique(train.ticker))) tickers")

println("\nLoading test split:  $TEST_DAYS")
test = load_split(TEST_DAYS)
println("  $(nrow(test)) observations across $(length(unique(test.ticker))) tickers")

tickers = sort(unique(train.ticker))
n_tickers = length(tickers)
sectors = sort(unique(train.sector))

# Drop test rows for tickers not in train
missing_in_train = setdiff(unique(test.ticker), tickers)
isempty(missing_in_train) || (test = test[[t in tickers for t in test.ticker], :])

# ============================================================================
# Earnings features
# ============================================================================

"""Return signed days-to-earnings for `ticker` on `date`, or `D2E_CLIP` if
the ticker has no calendar entry (ETFs)."""
function d2e_self_value(ticker::AbstractString, date::Date)
    v = days_to_earnings(cal, ticker, date)
    v === missing && return D2E_CLIP
    return clamp(v, -D2E_CLIP, D2E_CLIP)
end

"""Min |d2e| over same-sector equity peers (excluding self).
For ETFs the peer set is all equity tickers."""
function d2e_peer_min_value(ticker::AbstractString, sector::AbstractString,
                            date::Date)
    peers = if sector == "ETF"
        collect(EQUITY_TICKERS)
    else
        [t for (t, s) in SECTORS if s == sector && t != ticker && !(t in ETF_TICKERS)]
    end
    best = D2E_CLIP
    for p in peers
        v = days_to_earnings(cal, p, date)
        v === missing && continue
        a = abs(v)
        a < best && (best = a)
    end
    return clamp(best, 0, D2E_CLIP)  # peer-min is non-negative
end

function attach_earnings_features!(df::DataFrame)
    df[!, :d2e_self] = [d2e_self_value(t, d) for (t, d) in zip(df.ticker, df.obs_date)]
    df[!, :d2e_peer_min] = [d2e_peer_min_value(t, s, d) for (t, s, d)
                            in zip(df.ticker, df.sector, df.obs_date)]
    # For ETFs, replace d2e_self with d2e_peer_min (signed: 0 = today; positive future)
    for i in 1:nrow(df)
        if df.ticker[i] in ETF_TICKERS
            df.d2e_self[i] = df.d2e_peer_min[i]
        end
    end
    return df
end

println("\nAttaching earnings features...")
attach_earnings_features!(train)
attach_earnings_features!(test)

@printf("  train d2e_self        range: [%d, %d]   peer_min range: [%d, %d]\n",
        minimum(train.d2e_self), maximum(train.d2e_self),
        minimum(train.d2e_peer_min), maximum(train.d2e_peer_min))
@printf("  test  d2e_self        range: [%d, %d]   peer_min range: [%d, %d]\n",
        minimum(test.d2e_self), maximum(test.d2e_self),
        minimum(test.d2e_peer_min), maximum(test.d2e_peer_min))

# Mask of "near earnings" obs (own earnings within window OR peer within window)
near_earnings_mask(df) = (abs.(df.d2e_self) .<= EARNINGS_WINDOW) .|
                        (df.d2e_peer_min .<= EARNINGS_WINDOW)

n_near_train = sum(near_earnings_mask(train))
n_near_test  = sum(near_earnings_mask(test))
@printf("\n  Near-earnings rows: train %d / %d (%.1f%%); test %d / %d (%.1f%%)\n",
        n_near_train, nrow(train), 100*n_near_train/nrow(train),
        n_near_test,  nrow(test),  100*n_near_test/nrow(test))

# ============================================================================
# Helpers shared across configs
# ============================================================================

function rmse(pred, target)
    sqrt(mean((Float64.(pred) .- Float64.(target)).^2))
end

function compute_model_ivs_nn(psi_nn, log_theta, X, tidx)
    log_psi = psi_nn(X)
    log_theta_obs = log_theta[tidx]
    return exp.(Float32(0.5) .* (log_theta_obs .+ vec(log_psi)))
end

const SECTOR_EPOCHS = 2000
const SECTOR_PATIENCE = 200

function make_psi_nn(n_obs::Int, n_inputs::Int, seed::Int)
    Random.seed!(seed)
    h = n_obs >= 2000 ? 16 : 8
    return Chain(Dense(n_inputs => h, tanh), Dense(h => h, tanh), Dense(h => 1))
end

"""Train one sector NN. n_inputs is 2 (psi only) or 4 (psi + d2e features).
Returns (model, ticker_idx_local, training_history)."""
function train_sector_nn(sector::String, sector_train::DataFrame,
                         standardizer::NamedTuple, n_inputs::Int)
    sector_tickers = sort(unique(sector_train.ticker))
    s_tidx = Dict(t => i for (i, t) in enumerate(sector_tickers))
    n_obs = nrow(sector_train)

    log_dte = (log.(max.(Float64.(sector_train.actual_dte), 1.0)) .- standardizer.mu_dte) ./ standardizer.sigma_dte
    log_m   = (log.(Float64.(sector_train.moneyness)) .- standardizer.mu_m) ./ standardizer.sigma_m

    if n_inputs == 2
        Xs = hcat(Float32.(log_dte), Float32.(log_m))'
    else
        d2e_self = (Float64.(sector_train.d2e_self) .- standardizer.mu_d2e_self) ./ standardizer.sigma_d2e_self
        d2e_peer = (Float64.(sector_train.d2e_peer_min) .- standardizer.mu_d2e_peer) ./ standardizer.sigma_d2e_peer
        Xs = hcat(Float32.(log_dte), Float32.(log_m),
                  Float32.(d2e_self), Float32.(d2e_peer))'
    end
    ys = Float32.(sector_train.implied_vol)
    tidx_s = Int32[s_tidx[t] for t in sector_train.ticker]

    psi_s = make_psi_nn(n_obs, n_inputs, 42)
    log_theta_s = Float32[Float32(log(mean(Float64.(sector_train.implied_vol[sector_train.ticker .== t]))^2))
                          for t in sector_tickers]
    model_s = (psi_nn = psi_s, log_theta = log_theta_s)
    opt_s = Flux.setup(Flux.Adam(1e-3), model_s)

    @printf("  [%s] %d train obs, %d tickers, n_inputs=%d\n",
            sector, n_obs, length(sector_tickers), n_inputs)

    bl, bs, ni = Inf, nothing, 0
    for epoch in 1:SECTOR_EPOCHS
        l, g = Flux.withgradient(model_s) do m
            Flux.mse(compute_model_ivs_nn(m.psi_nn, m.log_theta, Xs, tidx_s), ys)
        end
        Flux.update!(opt_s, model_s, g[1])
        if l < bl; bl, bs, ni = l, Flux.state(model_s), 0; else; ni += 1; end
        ni >= SECTOR_PATIENCE && break
        epoch == 500  && Flux.adjust!(opt_s, 5e-4)
        epoch == 1000 && Flux.adjust!(opt_s, 2e-4)
        epoch == 1500 && Flux.adjust!(opt_s, 1e-4)
    end
    Flux.loadmodel!(model_s, bs)
    return model_s, s_tidx
end

"""Predict using the trained sector models. n_inputs must match training."""
function predict_sector_nn(df::DataFrame, sector_models::Dict,
                           standardizer::NamedTuple, n_inputs::Int)
    out = zeros(Float64, nrow(df))
    for sector in sectors
        mask = df.sector .== sector
        any(mask) || continue
        haskey(sector_models, sector) || continue
        sm = sector_models[sector]
        rows = df[mask, :]

        log_dte = (log.(max.(Float64.(rows.actual_dte), 1.0)) .- standardizer.mu_dte) ./ standardizer.sigma_dte
        log_m   = (log.(Float64.(rows.moneyness)) .- standardizer.mu_m) ./ standardizer.sigma_m
        if n_inputs == 2
            Xs = hcat(Float32.(log_dte), Float32.(log_m))'
        else
            d2e_self = (Float64.(rows.d2e_self) .- standardizer.mu_d2e_self) ./ standardizer.sigma_d2e_self
            d2e_peer = (Float64.(rows.d2e_peer_min) .- standardizer.mu_d2e_peer) ./ standardizer.sigma_d2e_peer
            Xs = hcat(Float32.(log_dte), Float32.(log_m),
                      Float32.(d2e_self), Float32.(d2e_peer))'
        end
        local_tidx = [get(sm.ticker_idx, t, 0) for t in rows.ticker]
        keep = local_tidx .> 0
        preds = fill(NaN, nrow(rows))
        if any(keep)
            tidx_local = Int32.(local_tidx[keep])
            preds[keep] .= Float64.(compute_model_ivs_nn(sm.model.psi_nn, sm.model.log_theta,
                                                        Xs[:, keep], tidx_local))
        end
        out[findall(mask)] .= preds
    end
    return out
end

"""Compute standardization stats from a TRAIN DataFrame, n_inputs aware."""
function compute_standardizer(df::DataFrame, n_inputs::Int)
    log_dte_v = log.(max.(Float64.(df.actual_dte), 1.0))
    log_m_v   = log.(Float64.(df.moneyness))
    s = (mu_dte = mean(log_dte_v), sigma_dte = std(log_dte_v),
         mu_m = mean(log_m_v),     sigma_m = std(log_m_v))
    if n_inputs == 4
        d2e_self = Float64.(df.d2e_self)
        d2e_peer = Float64.(df.d2e_peer_min)
        s = merge(s, (mu_d2e_self = mean(d2e_self), sigma_d2e_self = max(std(d2e_self), 1e-3),
                      mu_d2e_peer = mean(d2e_peer), sigma_d2e_peer = max(std(d2e_peer), 1e-3)))
    end
    return s
end

function run_config(label::String, train_df::DataFrame, test_df::DataFrame,
                    n_inputs::Int)
    println("\n" * "="^70)
    println("  CONFIGURATION $label  (n_inputs=$n_inputs, train=$(nrow(train_df)), test=$(nrow(test_df)))")
    println("="^70)

    standardizer = compute_standardizer(train_df, n_inputs)
    sector_models = Dict{String,Any}()
    for sector in sectors
        train_mask = train_df.sector .== sector
        sector_train = train_df[train_mask, :]
        nrow(sector_train) > 0 || continue
        model_s, ticker_idx_s = train_sector_nn(sector, sector_train,
                                                standardizer, n_inputs)
        sector_models[sector] = (model = model_s, ticker_idx = ticker_idx_s)
    end

    train_pred = predict_sector_nn(train_df, sector_models, standardizer, n_inputs)
    test_pred  = predict_sector_nn(test_df,  sector_models, standardizer, n_inputs)
    train_iv = Float64.(train_df.implied_vol)
    test_iv  = Float64.(test_df.implied_vol)
    train_rmse = rmse(train_pred, train_iv)
    test_rmse  = rmse(test_pred,  test_iv)

    @printf("\n  Train RMSE: %5.2f%%   Test RMSE: %5.2f%%   Gen gap: %+5.2f%%\n",
            train_rmse*100, test_rmse*100, (test_rmse - train_rmse)*100)

    println("\n  Per-sector test RMSE (%):")
    println("  Sector         N_test   RMSE")
    println("  " * "-"^35)
    for sector in sectors
        mask = test_df.sector .== sector
        n = sum(mask)
        n == 0 && continue
        @printf("  %-12s   %5d    %5.2f\n", sector, n,
                rmse(test_pred[mask], test_iv[mask])*100)
    end

    println("\n  Per-ticker test RMSE (%):")
    println("  Ticker  N_test   RMSE     d2e_self   d2e_peer")
    println("  " * "-"^55)
    rows = Tuple{String,Int,Float64,Float64,Float64}[]
    for t in tickers
        mask = test_df.ticker .== t
        n = sum(mask)
        n == 0 && continue
        r = rmse(test_pred[mask], test_iv[mask]) * 100
        # Show median d2e values for this ticker on test days
        d2e_s = median(Float64.(test_df.d2e_self[mask]))
        d2e_p = median(Float64.(test_df.d2e_peer_min[mask]))
        push!(rows, (t, n, r, d2e_s, d2e_p))
    end
    sort!(rows, by=r -> -r[3])
    for (t, n, r, d2es, d2ep) in rows
        @printf("  %-5s   %5d    %5.2f      %+5.0f       %5.0f\n", t, n, r, d2es, d2ep)
    end

    return (train_rmse = train_rmse, test_rmse = test_rmse,
            train_pred = train_pred, test_pred = test_pred,
            sector_models = sector_models, standardizer = standardizer)
end

# ============================================================================
# Run the three configurations
# ============================================================================

# Config A: pooled control (current 13.03% reproducer)
result_A = run_config("A: POOLED CONTROL (no exclusion, 2 inputs)",
                      train, test, 2)

# Config B: non-earnings baseline
keep_train_B = .!near_earnings_mask(train)
keep_test_B  = .!near_earnings_mask(test)
result_B = run_config("B: NON-EARNINGS BASELINE (exclude ±$EARNINGS_WINDOW d, 2 inputs)",
                      train[keep_train_B, :], test[keep_test_B, :], 2)

# Config C: earnings-aware
result_C = run_config("C: EARNINGS-AWARE (4 inputs: ln tau, ln m, d2e_self, d2e_peer)",
                      train, test, 4)

# ============================================================================
# Summary table
# ============================================================================

println("\n" * "="^70)
println("  THREE-CONFIG SUMMARY (sector NN)")
println("="^70)
println()
println("  Config                              N_train  N_test  Train(%)  Test(%)  GenGap(%)")
println("  " * "-"^85)
@printf("  A: pooled, 2 inputs                 %6d   %5d    %5.2f     %5.2f     %+5.2f\n",
        nrow(train), nrow(test),
        result_A.train_rmse*100, result_A.test_rmse*100,
        (result_A.test_rmse - result_A.train_rmse)*100)
@printf("  B: non-earnings, 2 inputs           %6d   %5d    %5.2f     %5.2f     %+5.2f\n",
        sum(keep_train_B), sum(keep_test_B),
        result_B.train_rmse*100, result_B.test_rmse*100,
        (result_B.test_rmse - result_B.train_rmse)*100)
@printf("  C: earnings-aware, 4 inputs         %6d   %5d    %5.2f     %5.2f     %+5.2f\n",
        nrow(train), nrow(test),
        result_C.train_rmse*100, result_C.test_rmse*100,
        (result_C.test_rmse - result_C.train_rmse)*100)
println()

# Tech-only spotlight (the blow-up sector)
println("\n  Tech-sector test RMSE comparison (the headline pivot):")
println("  Ticker  d2e_self  d2e_peer   A: pooled   C: earnings   delta")
println("  " * "-"^60)
for t in sort([k for (k, v) in SECTORS if v == "Tech"])
    mask = test.ticker .== t
    n = sum(mask)
    n == 0 && continue
    rA = rmse(result_A.test_pred[mask], Float64.(test.implied_vol[mask])) * 100
    rC = rmse(result_C.test_pred[mask], Float64.(test.implied_vol[mask])) * 100
    d2es = median(Float64.(test.d2e_self[mask]))
    d2ep = median(Float64.(test.d2e_peer_min[mask]))
    @printf("  %-5s   %+5.0f    %5.0f      %5.2f       %5.2f      %+5.2f\n",
            t, d2es, d2ep, rA, rC, rC - rA)
end

# Persist a tidy summary CSV for paper consumption
results_dir = joinpath(@__DIR__, "..", "figures")
mkpath(results_dir)
summary_path = joinpath(results_dir, "earnings_holdout_summary.csv")
summary_df = DataFrame(
    config = ["A_pooled_2input", "B_nonearnings_2input", "C_earnings_aware_4input"],
    n_train = [nrow(train), sum(keep_train_B), nrow(train)],
    n_test  = [nrow(test),  sum(keep_test_B),  nrow(test)],
    train_rmse = [result_A.train_rmse, result_B.train_rmse, result_C.train_rmse],
    test_rmse  = [result_A.test_rmse,  result_B.test_rmse,  result_C.test_rmse],
)
CSV.write(summary_path, summary_df)
println("\n  Wrote summary -> $summary_path")

println("\nDone.")
