"""
    TemporalFolds

Reusable data-loading, feature engineering, training, and evaluation for
ladder-corpus temporal holdouts. Used by:
- `examples/temporal_holdout_earnings.jl` — three-config (A/B/C) earnings ablation
- `examples/walk_forward_temporal.jl` — k=5..14 walk-forward validation

The module is intentionally standalone (not part of the HestonIV package) so
that downstream callers can `include` it from `examples/` without taking on
the full HestonIV dep graph.
"""
module TemporalFolds

using CSV
using DataFrames
using Statistics
using Flux
using Printf
using Random
using Dates

# EarningsCalendar lives in code/src; we include it relative to this file.
include("EarningsCalendar.jl")
using .EarningsCalendar

export SECTORS, ETF_TICKERS, EQUITY_TICKERS
export load_earnings_calendar, load_split, resolve_day, EXTENDED_DAYS,
       attach_earnings_features!, near_earnings_mask, compute_standardizer
export train_sector_nn, predict_sector_nn, run_fold
export rmse

# ============================================================================
# Static universe — kept in lockstep with the existing temporal_holdout script
# ============================================================================

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

const D2E_CLIP = 30          # clamp days-to-earnings to this magnitude
const EARNINGS_WINDOW = 3    # ±days for "near earnings" exclusion

# ============================================================================
# Extended corpus — 58 capture dates, 2026-04-14 .. 2026-07-17
#
# The first 15 entries are the frozen arm living in data/ladder; entries 16..58
# live in data/ladder_extended. Consumers resolve each name across both roots
# via `resolve_day`, so this list is layout-agnostic.
#
# Gaps are real: 04-30, 05-04, 05-05, 05-07, 06-05, 06-08 and 06-09 were not
# captured. 05-25 (Memorial Day) and 07-03 (Independence Day observed) are
# market holidays.
# ============================================================================

const EXTENDED_DAYS = [
    # --- frozen 15-date arm (data/ladder) -----------------------------------
    "options-04-14-2026", "options-04-15-2026", "options-04-16-2026",
    "options-04-17-2026", "options-04-21-2026", "options-04-22-2026",
    "options-04-23-2026", "options-04-24-2026", "options-04-27-2026",
    "options-04-28-2026", "options-04-29-2026", "options-05-01-2026",
    "options-05-06-2026", "options-05-08-2026", "options-05-11-2026",
    # --- extension (data/ladder_extended) -----------------------------------
    "options-05-12-2026", "options-05-13-2026", "options-05-14-2026",
    "options-05-15-2026", "options-05-18-2026", "options-05-19-2026",
    "options-05-20-2026", "options-05-21-2026", "options-05-22-2026",
    "options-05-26-2026", "options-05-27-2026", "options-05-28-2026",
    "options-05-29-2026", "options-06-01-2026", "options-06-02-2026",
    "options-06-03-2026", "options-06-04-2026", "options-06-10-2026",
    "options-06-11-2026", "options-06-12-2026", "options-06-15-2026",
    "options-06-16-2026", "options-06-17-2026", "options-06-18-2026",
    "options-06-22-2026", "options-06-23-2026", "options-06-24-2026",
    "options-06-25-2026", "options-06-26-2026", "options-06-29-2026",
    "options-06-30-2026", "options-07-01-2026", "options-07-02-2026",
    "options-07-06-2026", "options-07-07-2026", "options-07-08-2026",
    "options-07-09-2026", "options-07-10-2026", "options-07-13-2026",
    "options-07-14-2026", "options-07-15-2026", "options-07-16-2026",
    "options-07-17-2026",
]
@assert length(EXTENDED_DAYS) == 58

# ============================================================================
# Data loading
# ============================================================================

function dir_to_date(d::AbstractString)
    m = match(r"options-(\d{2})-(\d{2})-(\d{4})", String(d))
    Date(parse(Int, m.captures[3]), parse(Int, m.captures[1]), parse(Int, m.captures[2]))
end

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

"""
    resolve_day(roots, day) → String

Locate day-directory `day` across `roots` and return its full path. Raises if
it is absent from every root, and raises if more than one root supplies it —
both are silent corpus-corruption modes, so neither is a warn-and-continue.
"""
function resolve_day(roots::Vector{<:AbstractString}, day::AbstractString)
    hits = [joinpath(r, day) for r in roots if isdir(joinpath(r, day))]
    if isempty(hits)
        error("day directory '$(day)' not found in any root: " *
              join(roots, ", "))
    elseif length(hits) > 1
        error("day directory '$(day)' found in multiple roots: " *
              join(hits, ", "))
    end
    return hits[1]
end

"""
    load_split(roots::Vector{<:AbstractString}, day_dirs) → DataFrame

Canonical implementation. Resolves each entry in `day_dirs` across `roots`
via `resolve_day`, loads every `*_dte_ladder_*.csv` under each resolved
directory, and stacks the per-day frames into a single DataFrame with
`ticker, S, moneyness, obs_date, sector` columns added. The single-root
method below delegates to this one.
"""
function load_split(roots::Vector{<:AbstractString},
                    day_dirs::Vector{<:AbstractString})
    frames = DataFrame[]
    for d in day_dirs
        full = resolve_day(roots, d)
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

"""
    load_split(ladder_dir, day_dirs) → DataFrame

Load every `*_dte_ladder_*.csv` under each named day-directory and stack
into a single DataFrame with `ticker, S, moneyness, obs_date, sector` columns
added.
"""
# Single-root callers delegate, keeping one implementation of the load body.
load_split(ladder_dir::AbstractString, day_dirs::Vector{<:AbstractString}) =
    load_split([String(ladder_dir)], day_dirs)

load_earnings_calendar(path::AbstractString) = load_earnings(String(path))

# ============================================================================
# Earnings features
# ============================================================================

function d2e_self_value(cal, ticker::AbstractString, date::Date)
    v = days_to_earnings(cal, ticker, date)
    v === missing && return D2E_CLIP
    return clamp(v, -D2E_CLIP, D2E_CLIP)
end

function d2e_peer_min_value(cal, ticker::AbstractString, sector::AbstractString,
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
    return clamp(best, 0, D2E_CLIP)
end

"""
    attach_earnings_features!(df, cal) → df

Adds `:d2e_self` and `:d2e_peer_min` columns. For ETFs, `d2e_self` is set
equal to `d2e_peer_min` so the feature carries the sector-sympathy signal.
"""
function attach_earnings_features!(df::DataFrame, cal)
    df[!, :d2e_self] = [d2e_self_value(cal, t, d) for (t, d) in zip(df.ticker, df.obs_date)]
    df[!, :d2e_peer_min] = [d2e_peer_min_value(cal, t, s, d) for (t, s, d)
                            in zip(df.ticker, df.sector, df.obs_date)]
    for i in 1:nrow(df)
        if df.ticker[i] in ETF_TICKERS
            df.d2e_self[i] = df.d2e_peer_min[i]
        end
    end
    return df
end

near_earnings_mask(df::DataFrame) = (abs.(df.d2e_self) .<= EARNINGS_WINDOW) .|
                                    (df.d2e_peer_min .<= EARNINGS_WINDOW)

# ============================================================================
# Standardization + training + prediction
# ============================================================================

rmse(pred, target) = sqrt(mean((Float64.(pred) .- Float64.(target)).^2))

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

function train_sector_nn(sector::String, sector_train::DataFrame,
                         standardizer::NamedTuple, n_inputs::Int;
                         verbose::Bool=true, seed::Int=42)
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

    psi_s = make_psi_nn(n_obs, n_inputs, seed)
    log_theta_s = Float32[Float32(log(mean(Float64.(sector_train.implied_vol[sector_train.ticker .== t]))^2))
                          for t in sector_tickers]
    model_s = (psi_nn = psi_s, log_theta = log_theta_s)
    opt_s = Flux.setup(Flux.Adam(1e-3), model_s)

    verbose && @printf("  [%s] %d train obs, %d tickers, n_inputs=%d\n",
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

function predict_sector_nn(df::DataFrame, sector_models::Dict, sectors_list::Vector,
                           standardizer::NamedTuple, n_inputs::Int)
    out = zeros(Float64, nrow(df))
    for sector in sectors_list
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

# ============================================================================
# Public fold runner
# ============================================================================

"""
    run_fold(train_df, test_df, n_inputs; sectors_list=nothing, tickers_list=nothing,
             label="fold", verbose=true, seed=42) → NamedTuple

Train sector NNs on `train_df`, predict on both splits, return RMSE and
per-sector/per-ticker breakdowns. `n_inputs` is 2 (psi only) or 4 (psi + d2e
features). If `sectors_list` is `nothing`, derives from `train_df.sector`;
likewise for `tickers_list`.

Standardization is computed train-only per fold (the right thing for
walk-forward).
"""
function run_fold(train_df::DataFrame, test_df::DataFrame, n_inputs::Int;
                  sectors_list::Union{Nothing,AbstractVector}=nothing,
                  tickers_list::Union{Nothing,AbstractVector}=nothing,
                  label::AbstractString="fold", verbose::Bool=true,
                  seed::Int=42)
    sectors_use = sectors_list === nothing ? sort(unique(train_df.sector)) :
                  String.(sectors_list)
    tickers_use = tickers_list === nothing ? sort(unique(train_df.ticker)) :
                  String.(tickers_list)

    # Drop test rows whose ticker wasn't in train
    test_use = test_df[[t in tickers_use for t in test_df.ticker], :]

    verbose && println("\n" * "="^70)
    verbose && println("  FOLD: $label  (n_inputs=$n_inputs, train=$(nrow(train_df)), test=$(nrow(test_use)))")
    verbose && println("="^70)

    standardizer = compute_standardizer(train_df, n_inputs)
    sector_models = Dict{String,Any}()
    for sector in sectors_use
        train_mask = train_df.sector .== sector
        sector_train = train_df[train_mask, :]
        nrow(sector_train) > 0 || continue
        model_s, ticker_idx_s = train_sector_nn(sector, sector_train,
                                                standardizer, n_inputs;
                                                verbose=verbose, seed=seed)
        sector_models[sector] = (model = model_s, ticker_idx = ticker_idx_s)
    end

    train_pred = predict_sector_nn(train_df, sector_models, sectors_use, standardizer, n_inputs)
    test_pred  = predict_sector_nn(test_use, sector_models, sectors_use, standardizer, n_inputs)
    train_iv = Float64.(train_df.implied_vol)
    test_iv  = Float64.(test_use.implied_vol)
    train_rmse_v = rmse(train_pred, train_iv)
    test_rmse_v  = rmse(test_pred,  test_iv)

    per_sector_rmse = Dict{String,Float64}()
    for sector in sectors_use
        mask = test_use.sector .== sector
        sum(mask) > 0 || continue
        per_sector_rmse[sector] = rmse(test_pred[mask], test_iv[mask])
    end
    per_ticker_rmse = Dict{String,Float64}()
    for t in tickers_use
        mask = test_use.ticker .== t
        sum(mask) > 0 || continue
        per_ticker_rmse[t] = rmse(test_pred[mask], test_iv[mask])
    end

    if verbose
        @printf("\n  Train RMSE: %5.2f%%   Test RMSE: %5.2f%%   Gen gap: %+5.2f%%\n",
                train_rmse_v*100, test_rmse_v*100, (test_rmse_v - train_rmse_v)*100)
    end

    return (train_rmse = train_rmse_v, test_rmse = test_rmse_v,
            gen_gap = test_rmse_v - train_rmse_v,
            train_pred = train_pred, test_pred = test_pred,
            per_sector_rmse = per_sector_rmse,
            per_ticker_rmse = per_ticker_rmse,
            sector_models = sector_models, standardizer = standardizer,
            test_df = test_use)
end

end # module TemporalFolds
