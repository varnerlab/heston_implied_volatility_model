"""
Temporal Holdout: Per-Ticker NN vs Sector NN, Earnings-Excluded Subset

Companion to `temporal_holdout_earnings.jl`. The sector-NN three-config run
established that with ±3-day earnings windows excluded from train and test
(Configuration B) the sector NN reaches a 7.96% test RMSE with a -0.06%
generalization gap — i.e., once the earnings-driven obs are removed the
architecture generalizes faithfully out of sample. Per-ticker NN already
beats sector NN in-sample on qualified tickers (`calibrate_ladders_per_ticker_nn.jl`,
+0.33% IV pooled). This script asks the question that closes the loop:

    On the non-earnings subset, evaluated on the held-out 04-23/04-24 days,
    does per-ticker NN beat sector NN (7.96%)?

Pipeline:
  - Same train/test split: 04-14..04-22 train / 04-23..04-24 test (8-day corpus).
  - Same earnings exclusion: drop train and test rows where the row's ticker
    or any same-sector peer is within ±3 days of an earnings print.
  - Re-train sector NN under 2-input Configuration B on the surviving train
    rows. Reproduces the 7.96% reference.
  - For tickers with at least MIN_OBS_PER_TICKER non-earnings train rows,
    additionally train a 2->8->8->1 (or 2->16->16->1 for N>=5000) per-ticker
    psi network on that ticker's non-earnings train rows. Test on that ticker's
    non-earnings test rows.
  - Compute combined test RMSE: per-ticker NN where the ticker qualifies,
    sector NN otherwise. Report per-ticker test RMSE side by side.

Output: log + summary CSV in code/figures/.
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

const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladder")
const EARNINGS_CSV = joinpath(@__DIR__, "..", "data", "earnings", "earnings_calendar.csv")
const PLOT_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(PLOT_DIR)

const TRAIN_DAYS = ["options-04-14-2026", "options-04-15-2026",
                    "options-04-16-2026", "options-04-17-2026",
                    "options-04-21-2026", "options-04-22-2026"]
const TEST_DAYS  = ["options-04-23-2026", "options-04-24-2026"]
const EARNINGS_WINDOW = 3
const D2E_CLIP = 30
const MIN_OBS_PER_TICKER = 2000   # match calibrate_ladders_per_ticker_nn.jl

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

# ============================================================================
# Data loading
# ============================================================================

function dir_to_date(d::String)
    m = match(r"options-(\d{2})-(\d{2})-(\d{4})", d)
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

# ============================================================================
# Earnings features (need only the masking; per-ticker NN inputs stay 2-D)
# ============================================================================

function d2e_self_value(ticker::AbstractString, date::Date)
    v = days_to_earnings(cal, ticker, date)
    v === missing && return D2E_CLIP
    return clamp(v, -D2E_CLIP, D2E_CLIP)
end

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
    return clamp(best, 0, D2E_CLIP)
end

function attach_earnings_features!(df::DataFrame)
    df[!, :d2e_self] = [d2e_self_value(t, d) for (t, d) in zip(df.ticker, df.obs_date)]
    df[!, :d2e_peer_min] = [d2e_peer_min_value(t, s, d) for (t, s, d)
                            in zip(df.ticker, df.sector, df.obs_date)]
    for i in 1:nrow(df)
        if df.ticker[i] in ETF_TICKERS
            df.d2e_self[i] = df.d2e_peer_min[i]
        end
    end
    return df
end

println("\nAttaching earnings features to train and test ...")
attach_earnings_features!(train)
attach_earnings_features!(test)

near_earnings_mask(df) = (abs.(df.d2e_self) .<= EARNINGS_WINDOW) .|
                        (df.d2e_peer_min .<= EARNINGS_WINDOW)

train_keep = .!near_earnings_mask(train)
test_keep  = .!near_earnings_mask(test)

train_ne = train[train_keep, :]
test_ne  = test[test_keep, :]

@printf("\nNon-earnings subset: train %d / %d (%.1f%%); test %d / %d (%.1f%%)\n",
        nrow(train_ne), nrow(train), 100*nrow(train_ne)/nrow(train),
        nrow(test_ne),  nrow(test),  100*nrow(test_ne)/nrow(test))

# Drop test rows for tickers absent from non-earnings train
train_ne_tickers = Set(unique(train_ne.ticker))
keep_test_t = [t in train_ne_tickers for t in test_ne.ticker]
test_ne = test_ne[keep_test_t, :]
@printf("  test after restricting to train tickers: %d obs across %d tickers\n",
        nrow(test_ne), length(unique(test_ne.ticker)))

# ============================================================================
# Standardization (computed on non-earnings TRAIN; applied to both splits)
# ============================================================================

const MU_DTE    = mean(log.(max.(Float64.(train_ne.actual_dte), 1.0)))
const SIGMA_DTE = std(log.(max.(Float64.(train_ne.actual_dte), 1.0)))
const MU_M      = mean(log.(Float64.(train_ne.moneyness)))
const SIGMA_M   = std(log.(Float64.(train_ne.moneyness)))

function standardize_2d(df::DataFrame)
    log_dte = (log.(max.(Float64.(df.actual_dte), 1.0)) .- MU_DTE) ./ SIGMA_DTE
    log_m   = (log.(Float64.(df.moneyness)) .- MU_M) ./ SIGMA_M
    return hcat(Float32.(log_dte), Float32.(log_m))'
end

# ============================================================================
# Training helpers
# ============================================================================

const SECTOR_EPOCHS   = 2000
const SECTOR_PATIENCE = 200

function rmse(pred, target)
    sqrt(mean((Float64.(pred) .- Float64.(target)).^2))
end

function compute_iv_sector(psi_nn, log_theta, X, tidx)
    log_psi = psi_nn(X)
    return exp.(Float32(0.5) .* (log_theta[tidx] .+ vec(log_psi)))
end

function compute_iv_perticker(psi_nn, log_theta_scalar, X)
    log_psi = psi_nn(X)
    return exp.(Float32(0.5) .* (log_theta_scalar[1] .+ vec(log_psi)))
end

function train_with_schedule!(model, Xs, ys, loss_fn;
                              n_epochs::Int=SECTOR_EPOCHS,
                              patience::Int=SECTOR_PATIENCE,
                              lr_init::Float64=1e-3)
    opt = Flux.setup(Adam(Float32(lr_init)), model)
    bl, bs, ni = Inf, nothing, 0
    for epoch in 1:n_epochs
        l, g = Flux.withgradient(model) do m
            loss_fn(m, Xs, ys)
        end
        Flux.update!(opt, model, g[1])
        if l < bl
            bl, bs, ni = l, Flux.state(model), 0
        else
            ni += 1
        end
        ni >= patience && break
        epoch == 500  && Flux.adjust!(opt, 5f-4)
        epoch == 1000 && Flux.adjust!(opt, 2f-4)
        epoch == 1500 && Flux.adjust!(opt, 1f-4)
    end
    Flux.loadmodel!(model, bs)
    return model
end

# ============================================================================
# Train sector NN on non-earnings train (Config B reproducer)
# ============================================================================

println("\n" * "="^70)
println("  TRAINING SECTOR NN on non-earnings train (Config B baseline)")
println("="^70)

sectors_present = sort(unique(train_ne.sector))

sector_models = Dict{String,Any}()

for sector in sectors_present
    sector_train = train_ne[train_ne.sector .== sector, :]
    sector_tickers = sort(unique(sector_train.ticker))
    s_tidx = Dict(t => i for (i, t) in enumerate(sector_tickers))
    n_obs = nrow(sector_train)

    Xs = standardize_2d(sector_train)
    ys = Float32.(sector_train.implied_vol)
    tidx_s = Int32[s_tidx[t] for t in sector_train.ticker]

    Random.seed!(42)
    h = n_obs >= 2000 ? 16 : 8
    psi_nn = Chain(Dense(2 => h, tanh), Dense(h => h, tanh), Dense(h => 1))
    log_theta = Float32[Float32(log(mean(Float64.(sector_train.implied_vol[sector_train.ticker .== t]))^2))
                        for t in sector_tickers]
    model = (psi_nn = psi_nn, log_theta = log_theta)

    loss_fn(m, X, y) = Flux.mse(compute_iv_sector(m.psi_nn, m.log_theta, X, tidx_s), y)
    train_with_schedule!(model, Xs, ys, loss_fn)

    @printf("  [%-11s] %5d obs, %2d tickers, h=%d\n",
            sector, n_obs, length(sector_tickers), h)

    sector_models[sector] = (model = model, ticker_idx = s_tidx)
end

function predict_sector(df::DataFrame)
    out = fill(NaN, nrow(df))
    for sector in sectors_present
        mask = df.sector .== sector
        any(mask) || continue
        haskey(sector_models, sector) || continue
        sm = sector_models[sector]
        rows = df[mask, :]
        Xs = standardize_2d(rows)
        local_t = [get(sm.ticker_idx, t, 0) for t in rows.ticker]
        keep = local_t .> 0
        preds = fill(NaN, nrow(rows))
        if any(keep)
            preds[keep] .= Float64.(compute_iv_sector(sm.model.psi_nn, sm.model.log_theta,
                                                     Xs[:, keep], Int32.(local_t[keep])))
        end
        out[findall(mask)] .= preds
    end
    return out
end

sector_train_pred = predict_sector(train_ne)
sector_test_pred  = predict_sector(test_ne)
sector_train_iv   = Float64.(train_ne.implied_vol)
sector_test_iv    = Float64.(test_ne.implied_vol)

sector_train_rmse = rmse(sector_train_pred, sector_train_iv) * 100
sector_test_rmse  = rmse(sector_test_pred,  sector_test_iv)  * 100

@printf("\n  Sector NN (Config B): train %.2f%% | test %.2f%% | gap %+.2f%%\n",
        sector_train_rmse, sector_test_rmse, sector_test_rmse - sector_train_rmse)

# ============================================================================
# Per-ticker NN training on non-earnings train (Config B per-ticker)
# ============================================================================

println("\n" * "="^70)
println("  TRAINING PER-TICKER NN on non-earnings train")
println("="^70)

train_obs_by_ticker = Dict(t => sum(train_ne.ticker .== t) for t in unique(train_ne.ticker))
qualified = sort([t for (t, n) in train_obs_by_ticker if n >= MIN_OBS_PER_TICKER])
unqualified = sort([t for (t, n) in train_obs_by_ticker if n < MIN_OBS_PER_TICKER])

println("  Qualification threshold: N_train_ne >= $MIN_OBS_PER_TICKER")
println("  Qualified ($(length(qualified))):   ", join(qualified, ", "))
println("  Unqualified ($(length(unqualified))): ", join(unqualified, ", "))

per_ticker_models = Dict{String,Any}()

for t in qualified
    td = train_ne[train_ne.ticker .== t, :]
    n_obs = nrow(td)
    Xt = standardize_2d(td)
    yt = Float32.(td.implied_vol)

    Random.seed!(42)
    h = n_obs >= 5000 ? 16 : 8
    arch = "2->$h->$h->1"
    psi_nn = Chain(Dense(2 => h, tanh), Dense(h => h, tanh), Dense(h => 1))
    log_theta = Float32[Float32(log(mean(Float64.(td.implied_vol))^2))]
    model = (psi_nn = psi_nn, log_theta = log_theta)

    loss_fn(m, X, y) = Flux.mse(compute_iv_perticker(m.psi_nn, m.log_theta, X), y)
    train_with_schedule!(model, Xt, yt, loss_fn)

    train_pred_t = Float64.(compute_iv_perticker(model.psi_nn, model.log_theta, Xt))
    train_rmse_t = rmse(train_pred_t, Float64.(yt)) * 100

    per_ticker_models[t] = (model = model,)
    @printf("  %-5s [%-11s] N_train=%5d  arch=%s  train RMSE=%.2f%%\n",
            t, get(SECTORS, t, "Other"), n_obs, arch, train_rmse_t)
end

# ============================================================================
# Test-set evaluation: per-ticker for qualified, sector fallback otherwise
# ============================================================================

println("\n" * "="^70)
println("  TEST-SET EVALUATION (non-earnings)")
println("="^70)

# Build a combined test prediction vector
combined_test_pred = copy(sector_test_pred)
for (i, t) in enumerate(test_ne.ticker)
    haskey(per_ticker_models, t) || continue
    pm = per_ticker_models[t].model
    row_X = standardize_2d(test_ne[i:i, :])
    combined_test_pred[i] = Float64.(compute_iv_perticker(pm.psi_nn, pm.log_theta, row_X))[1]
end

combined_test_rmse = rmse(combined_test_pred, sector_test_iv) * 100

# Per-ticker test RMSE table: sector vs per-ticker on the SAME test rows
test_tickers = sort(unique(test_ne.ticker))
println("\n  Per-ticker test RMSE on non-earnings holdout (% IV):")
println("  Ticker  Sector       N_test   Sector%   Per-ticker%   Delta%   Qualified")
println("  " * "-"^75)
table_rows = NamedTuple[]
for t in test_tickers
    mask = test_ne.ticker .== t
    n = sum(mask)
    n == 0 && continue
    sec_r = rmse(sector_test_pred[mask], sector_test_iv[mask]) * 100
    pt_r  = NaN
    is_q = haskey(per_ticker_models, t)
    if is_q
        pm = per_ticker_models[t].model
        Xs = standardize_2d(test_ne[mask, :])
        preds = Float64.(compute_iv_perticker(pm.psi_nn, pm.log_theta, Xs))
        pt_r = rmse(preds, sector_test_iv[mask]) * 100
    end
    delta = is_q ? (sec_r - pt_r) : NaN
    push!(table_rows, (ticker=t, sector=get(SECTORS, t, "Other"), n_test=n,
                       sector_rmse=sec_r, per_ticker_rmse=pt_r, delta=delta, qualified=is_q))
end
sort!(table_rows, by=r -> -(isnan(r.delta) ? -Inf : r.delta))
for r in table_rows
    if r.qualified
        @printf("  %-5s   %-11s  %5d    %5.2f      %5.2f      %+5.2f     yes\n",
                r.ticker, r.sector, r.n_test, r.sector_rmse, r.per_ticker_rmse, r.delta)
    else
        @printf("  %-5s   %-11s  %5d    %5.2f         --          --     no\n",
                r.ticker, r.sector, r.n_test, r.sector_rmse)
    end
end

println("\n  Headline test RMSE (non-earnings, n_test=$(nrow(test_ne))):")
@printf("    Sector NN only:                          %5.2f%%\n", sector_test_rmse)
@printf("    Per-ticker NN (qualified) + sector NN:   %5.2f%%   delta %+.2f%%\n",
        combined_test_rmse, sector_test_rmse - combined_test_rmse)

# ============================================================================
# Persist artifacts
# ============================================================================

cmp_csv = joinpath(PLOT_DIR, "earnings_holdout_per_ticker_summary.csv")
CSV.write(cmp_csv, DataFrame(table_rows))
println("\n  -> wrote $cmp_csv")

println("\n[Config B strict] Summary:")
@printf("  Sector NN (Config B reproducer): test %.2f%% (gen gap %+.2f%%)\n",
        sector_test_rmse, sector_test_rmse - sector_train_rmse)
@printf("  Per-ticker NN + sector fallback: test %.2f%% on the same %d non-earnings rows\n",
        combined_test_rmse, nrow(test_ne))
n_qual_test = sum(r.n_test for r in table_rows if r.qualified; init=0)
n_total_test = sum(r.n_test for r in table_rows; init=0)
@printf("  Qualified tickers: %d / %d  (covering %.0f%% of test obs)\n",
        length(qualified), length(test_tickers),
        n_total_test == 0 ? 0.0 : 100 * n_qual_test / n_total_test)
if n_qual_test == 0
    println("  WARNING: no qualified ticker has non-earnings test rows. Per-ticker NN")
    println("           cannot be evaluated head-to-head under strict Config B at this")
    println("           corpus size. Continuing with Config B' (asymmetric).")
end

# ============================================================================
# CONFIG B': per-ticker trained on FULL train (matches the in-sample run
# that already qualified 18/31 tickers), evaluated on the SAME non-earnings
# test set as Config B. This is the apples-to-apples comparison: same non-
# earnings test rows, sector NN vs per-ticker NN.
# ============================================================================

println("\n" * "="^70)
println("  CONFIG B' (asymmetric): per-ticker on FULL train, test on non-earnings")
println("="^70)

# Standardizers re-derived from the FULL train so they match the per-ticker
# fits (separate from the non-earnings standardizers above).
const MU_DTE_F    = mean(log.(max.(Float64.(train.actual_dte), 1.0)))
const SIGMA_DTE_F = std(log.(max.(Float64.(train.actual_dte), 1.0)))
const MU_M_F      = mean(log.(Float64.(train.moneyness)))
const SIGMA_M_F   = std(log.(Float64.(train.moneyness)))

function standardize_2d_full(df::DataFrame)
    log_dte = (log.(max.(Float64.(df.actual_dte), 1.0)) .- MU_DTE_F) ./ SIGMA_DTE_F
    log_m   = (log.(Float64.(df.moneyness)) .- MU_M_F) ./ SIGMA_M_F
    return hcat(Float32.(log_dte), Float32.(log_m))'
end

# --- B': sector NN on full train ---
println("\n  Training sector NN on FULL train ...")
sector_models_full = Dict{String,Any}()
sectors_all = sort(unique(train.sector))
for sector in sectors_all
    sector_train = train[train.sector .== sector, :]
    sector_tickers = sort(unique(sector_train.ticker))
    s_tidx = Dict(t => i for (i, t) in enumerate(sector_tickers))
    n_obs = nrow(sector_train)

    log_dte = (log.(max.(Float64.(sector_train.actual_dte), 1.0)) .- MU_DTE_F) ./ SIGMA_DTE_F
    log_m   = (log.(Float64.(sector_train.moneyness)) .- MU_M_F) ./ SIGMA_M_F
    Xs = hcat(Float32.(log_dte), Float32.(log_m))'
    ys = Float32.(sector_train.implied_vol)
    tidx_s = Int32[s_tidx[t] for t in sector_train.ticker]

    Random.seed!(42)
    h = n_obs >= 2000 ? 16 : 8
    psi_nn = Chain(Dense(2 => h, tanh), Dense(h => h, tanh), Dense(h => 1))
    log_theta = Float32[Float32(log(mean(Float64.(sector_train.implied_vol[sector_train.ticker .== t]))^2))
                        for t in sector_tickers]
    model = (psi_nn = psi_nn, log_theta = log_theta)
    loss_fn(m, X, y) = Flux.mse(compute_iv_sector(m.psi_nn, m.log_theta, X, tidx_s), y)
    train_with_schedule!(model, Xs, ys, loss_fn)
    sector_models_full[sector] = (model = model, ticker_idx = s_tidx)
    @printf("    [%-11s] %5d obs, %2d tickers, h=%d\n",
            sector, n_obs, length(sector_tickers), h)
end

function predict_sector_full(df::DataFrame)
    out = fill(NaN, nrow(df))
    for sector in sectors_all
        mask = df.sector .== sector
        any(mask) || continue
        haskey(sector_models_full, sector) || continue
        sm = sector_models_full[sector]
        rows = df[mask, :]
        Xs = standardize_2d_full(rows)
        local_t = [get(sm.ticker_idx, t, 0) for t in rows.ticker]
        keep = local_t .> 0
        preds = fill(NaN, nrow(rows))
        if any(keep)
            preds[keep] .= Float64.(compute_iv_sector(sm.model.psi_nn, sm.model.log_theta,
                                                     Xs[:, keep], Int32.(local_t[keep])))
        end
        out[findall(mask)] .= preds
    end
    return out
end

sector_full_test_pred = predict_sector_full(test_ne)
sector_full_test_rmse = rmse(sector_full_test_pred, sector_test_iv) * 100
@printf("\n  Sector NN (full train) on non-earnings test: %.2f%%\n", sector_full_test_rmse)

# --- B': per-ticker NN on full train ---
println("\n  Training per-ticker NN on FULL train ...")
full_obs_by_ticker = Dict(t => sum(train.ticker .== t) for t in unique(train.ticker))
qualified_full = sort([t for (t, n) in full_obs_by_ticker if n >= MIN_OBS_PER_TICKER])
unqualified_full = sort([t for (t, n) in full_obs_by_ticker if n < MIN_OBS_PER_TICKER])
println("    Qualified ($(length(qualified_full))): ", join(qualified_full, ", "))
println("    Unqualified ($(length(unqualified_full))): ", join(unqualified_full, ", "))

per_ticker_full = Dict{String,Any}()
for t in qualified_full
    td = train[train.ticker .== t, :]
    n_obs = nrow(td)
    Xt = standardize_2d_full(td)
    yt = Float32.(td.implied_vol)
    Random.seed!(42)
    h = n_obs >= 5000 ? 16 : 8
    psi_nn = Chain(Dense(2 => h, tanh), Dense(h => h, tanh), Dense(h => 1))
    log_theta = Float32[Float32(log(mean(Float64.(td.implied_vol))^2))]
    model = (psi_nn = psi_nn, log_theta = log_theta)
    loss_fn(m, X, y) = Flux.mse(compute_iv_perticker(m.psi_nn, m.log_theta, X), y)
    train_with_schedule!(model, Xt, yt, loss_fn)
    train_pred_t = Float64.(compute_iv_perticker(model.psi_nn, model.log_theta, Xt))
    train_rmse_t = rmse(train_pred_t, Float64.(yt)) * 100
    per_ticker_full[t] = (model = model,)
    @printf("    %-5s [%-11s] N_train=%5d  arch=2->%d->%d->1  train RMSE=%.2f%%\n",
            t, get(SECTORS, t, "Other"), n_obs, h, h, train_rmse_t)
end

# --- B': head-to-head per-ticker test RMSE on the non-earnings test ---
println("\n  Per-ticker test RMSE on non-earnings holdout (% IV), Config B':")
println("  Ticker  Sector       N_test   Sector%   Per-ticker%   Delta%   Qualified")
println("  " * "-"^75)
table_rows_b2 = NamedTuple[]
combined_test_pred_b2 = copy(sector_full_test_pred)
for t in test_tickers
    mask = test_ne.ticker .== t
    n = sum(mask)
    n == 0 && continue
    sec_r = rmse(sector_full_test_pred[mask], sector_test_iv[mask]) * 100
    is_q = haskey(per_ticker_full, t)
    pt_r = NaN
    delta = NaN
    if is_q
        pm = per_ticker_full[t].model
        Xs = standardize_2d_full(test_ne[mask, :])
        preds = Float64.(compute_iv_perticker(pm.psi_nn, pm.log_theta, Xs))
        pt_r = rmse(preds, sector_test_iv[mask]) * 100
        delta = sec_r - pt_r
        # Splice into combined predictions
        idxs = findall(mask)
        combined_test_pred_b2[idxs] .= preds
    end
    push!(table_rows_b2, (ticker=t, sector=get(SECTORS, t, "Other"), n_test=n,
                          sector_rmse=sec_r, per_ticker_rmse=pt_r, delta=delta, qualified=is_q))
end
sort!(table_rows_b2, by=r -> -(isnan(r.delta) ? -Inf : r.delta))
for r in table_rows_b2
    if r.qualified
        @printf("  %-5s   %-11s  %5d    %5.2f      %5.2f      %+5.2f     yes\n",
                r.ticker, r.sector, r.n_test, r.sector_rmse, r.per_ticker_rmse, r.delta)
    else
        @printf("  %-5s   %-11s  %5d    %5.2f         --          --     no\n",
                r.ticker, r.sector, r.n_test, r.sector_rmse)
    end
end

combined_test_rmse_b2 = rmse(combined_test_pred_b2, sector_test_iv) * 100
n_qual_test_b2 = sum(r.n_test for r in table_rows_b2 if r.qualified; init=0)
n_total_test_b2 = sum(r.n_test for r in table_rows_b2; init=0)

println("\n  Headline test RMSE (Config B', n_test=$(nrow(test_ne))):")
@printf("    Sector NN (full train) on non-earnings test:                %5.2f%%\n",
        sector_full_test_rmse)
@printf("    Per-ticker NN (qualified) + sector NN on non-earnings test: %5.2f%%   delta %+.2f%%\n",
        combined_test_rmse_b2, sector_full_test_rmse - combined_test_rmse_b2)
@printf("    Qualified tickers in test: %d / %d  (covering %.0f%% of test obs)\n",
        sum(r.qualified for r in table_rows_b2), length(table_rows_b2),
        n_total_test_b2 == 0 ? 0.0 : 100 * n_qual_test_b2 / n_total_test_b2)

# Persist Config B' table too
cmp_csv_b2 = joinpath(PLOT_DIR, "earnings_holdout_per_ticker_b2_summary.csv")
CSV.write(cmp_csv_b2, DataFrame(table_rows_b2))
println("  -> wrote $cmp_csv_b2")

println("\n" * "="^70)
println("  FINAL TWO-CONFIG SUMMARY (per-ticker NN vs sector NN, non-earnings test)")
println("="^70)
@printf("  Strict Config B (non-earn train + non-earn test):\n")
@printf("    Sector NN:      %5.2f%%   (canonical reference)\n", sector_test_rmse)
@printf("    Per-ticker NN:  %5.2f%%   (qualified=%d, but %d covered in test)\n",
        combined_test_rmse, length(qualified), n_qual_test)
@printf("  Asymmetric Config B-prime (full train + non-earn test):\n")
@printf("    Sector NN:      %5.2f%%\n", sector_full_test_rmse)
@printf("    Per-ticker NN:  %5.2f%%   (qualified=%d, %d/%d test obs covered)\n",
        combined_test_rmse_b2, length(qualified_full), n_qual_test_b2, n_total_test_b2)
