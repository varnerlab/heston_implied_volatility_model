"""
Leave-one-date-out validation for the sector-NN calibration.

Holds out a single capture date as test, trains the six sector psi-networks
on the remaining dates with the same architecture and schedule used in
calibrate_ladders_sector_nn.jl, then reports (train RMSE, test RMSE, gap)
overall and per-sector. Confirms whether the 10.24% in-sample sector-NN
number reflects faithful generalization or in-sample overfit.

Default holdout = 2026-05-11 (latest capture). Override with --holdout DATE.

Usage:
    julia --project=. examples/holdout_validation_sector_nn.jl
    julia --project=. examples/holdout_validation_sector_nn.jl --holdout 2026-04-23
"""

using CSV
using DataFrames
using Dates
using Statistics
using Flux
using Printf
using Random

const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladder")

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

# ----- Parse --holdout flag -------------------------------------------------
holdout_date = Date("2026-05-11")
for (i, a) in enumerate(ARGS)
    if a == "--holdout" && i < length(ARGS)
        holdout_date = Date(ARGS[i+1])
    end
end

# ----- Load ladder ----------------------------------------------------------
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

println("Loading ladder corpus...")
all_data = load_all_ladders(LADDER_DIR)
all_data[!, :sector] = [get(SECTORS, t, "Other") for t in all_data.ticker]

unique_dates = sort(unique(all_data.und_session_date))
println("  $(nrow(all_data)) observations, $(length(unique(all_data.ticker))) tickers, $(length(unique_dates)) capture dates")
println("  date range: $(unique_dates[1]) → $(unique_dates[end])")

# ----- Train/test split -----------------------------------------------------
holdout_date in unique_dates ||
    error("Holdout date $holdout_date not in corpus. Available: $unique_dates")

train_data = all_data[all_data.und_session_date .!= holdout_date, :]
test_data  = all_data[all_data.und_session_date .== holdout_date, :]
@printf("\n  Holding out %s: %d test obs (%.1f%% of corpus); %d train obs across %d dates\n",
        holdout_date, nrow(test_data), 100*nrow(test_data)/nrow(all_data),
        nrow(train_data), length(unique_dates)-1)

# Standardisation constants come from TRAIN ONLY to avoid test leakage.
const MU_DTE = mean(log.(max.(Float64.(train_data.actual_dte), 1.0)))
const SIGMA_DTE = std( log.(max.(Float64.(train_data.actual_dte), 1.0)))
const MU_M   = mean(log.(Float64.(train_data.moneyness)))
const SIGMA_M = std( log.(Float64.(train_data.moneyness)))

sectors = sort(unique(all_data.sector))

# ----- Train one sector NN per sector on TRAIN, evaluate on both sets -------
function make_psi_nn(n_obs::Int)
    n_obs >= 2000 ?
        Chain(Dense(2 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 1)) :
        Chain(Dense(2 => 8,  tanh), Dense(8  => 8,  tanh), Dense(8  => 1))
end

function build_X(df)
    log_dte = Float32.((log.(max.(Float64.(df.actual_dte), 1.0)) .- MU_DTE) ./ SIGMA_DTE)
    log_m   = Float32.((log.(Float64.(df.moneyness)) .- MU_M) ./ SIGMA_M)
    return hcat(log_dte, log_m)'
end

function compute_ivs(psi_nn, log_theta, X, tidx)
    log_psi = psi_nn(X)
    return exp.(Float32(0.5) .* (log_theta[tidx] .+ vec(log_psi)))
end

println("\n" * "="^72)
println("  TRAINING SECTOR NNS ON TRAIN SPLIT (14 dates)")
println("="^72)

train_rmse_by_sector = Dict{String,Float64}()
test_rmse_by_sector  = Dict{String,Float64}()
train_n_by_sector    = Dict{String,Int}()
test_n_by_sector     = Dict{String,Int}()

for sector in sectors
    train_sec = train_data[train_data.sector .== sector, :]
    test_sec  = test_data[test_data.sector .== sector, :]
    sector_tickers = sort(unique(train_sec.ticker))
    ticker_idx = Dict(t => i for (i, t) in enumerate(sector_tickers))
    n_train = nrow(train_sec)
    n_test  = nrow(test_sec)

    X_train = build_X(train_sec)
    y_train = Float32.(train_sec.implied_vol)
    tidx_train = Int32[ticker_idx[t] for t in train_sec.ticker]

    Random.seed!(42)
    psi_nn = make_psi_nn(n_train)
    log_theta = Float32[Float32(log(mean(Float64.(train_sec.implied_vol[train_sec.ticker .== t]))^2))
                        for t in sector_tickers]
    model = (psi_nn = psi_nn, log_theta = log_theta)
    opt_state = Flux.setup(Adam(1f-3), model)
    best_loss = Inf; best_state = nothing; no_improve = 0
    for epoch in 1:2000
        l, grads = Flux.withgradient(model) do m
            Flux.mse(compute_ivs(m.psi_nn, m.log_theta, X_train, tidx_train), y_train)
        end
        Flux.update!(opt_state, model, grads[1])
        if l < best_loss
            best_loss = l; best_state = Flux.state(model); no_improve = 0
        else
            no_improve += 1
        end
        no_improve >= 200 && break
        epoch == 500  && Flux.adjust!(opt_state, 5f-4)
        epoch == 1000 && Flux.adjust!(opt_state, 2f-4)
        epoch == 1500 && Flux.adjust!(opt_state, 1f-4)
    end
    Flux.loadmodel!(model, best_state)

    # Train RMSE
    train_pred = Float64.(compute_ivs(model.psi_nn, model.log_theta, X_train, tidx_train))
    rmse_train = sqrt(mean((train_pred .- Float64.(y_train)).^2))

    # Test RMSE — only tickers seen in train; if test holds a ticker not in train, skip it.
    if n_test > 0
        test_tickers_ok = [t in keys(ticker_idx) for t in test_sec.ticker]
        if any(test_tickers_ok)
            test_keep = test_sec[test_tickers_ok, :]
            X_test = build_X(test_keep)
            y_test = Float32.(test_keep.implied_vol)
            tidx_test = Int32[ticker_idx[t] for t in test_keep.ticker]
            test_pred = Float64.(compute_ivs(model.psi_nn, model.log_theta, X_test, tidx_test))
            rmse_test = sqrt(mean((test_pred .- Float64.(y_test)).^2))
            test_rmse_by_sector[sector] = rmse_test
            test_n_by_sector[sector]    = nrow(test_keep)
        end
    end
    train_rmse_by_sector[sector] = rmse_train
    train_n_by_sector[sector]    = n_train

    if haskey(test_rmse_by_sector, sector)
        @printf("  [%-11s]  train n=%6d RMSE=%5.2f%%   |   test n=%5d RMSE=%5.2f%%   gap=%+5.2f%%\n",
                sector, n_train, rmse_train*100,
                test_n_by_sector[sector], test_rmse_by_sector[sector]*100,
                (test_rmse_by_sector[sector] - rmse_train)*100)
    else
        @printf("  [%-11s]  train n=%6d RMSE=%5.2f%%   |   (no test obs)\n",
                sector, n_train, rmse_train*100)
    end
end

# ----- Overall pooled RMSEs --------------------------------------------------
overall_train_sq = sum(train_n_by_sector[s] * train_rmse_by_sector[s]^2 for s in sectors if haskey(train_rmse_by_sector, s))
overall_train_n  = sum(values(train_n_by_sector))
overall_train_rmse = sqrt(overall_train_sq / overall_train_n)

overall_test_sq = sum(test_n_by_sector[s] * test_rmse_by_sector[s]^2 for s in sectors if haskey(test_rmse_by_sector, s))
overall_test_n  = sum(values(test_n_by_sector))
overall_test_rmse = overall_test_n > 0 ? sqrt(overall_test_sq / overall_test_n) : NaN

println("\n" * "="^72)
println("  SUMMARY — leave-one-date-out  (holdout = $holdout_date)")
println("="^72)
@printf("  Overall train RMSE: %5.2f%%   (n = %d)\n", overall_train_rmse*100, overall_train_n)
@printf("  Overall test  RMSE: %5.2f%%   (n = %d)\n", overall_test_rmse*100, overall_test_n)
@printf("  Generalization gap: %+5.2f%% IV\n", (overall_test_rmse - overall_train_rmse)*100)

println("\n  Per-sector gap:")
@printf("  %-11s   train(%%)   test(%%)   gap(%%)\n", "Sector")
println("  " * "-"^48)
for sector in sectors
    haskey(test_rmse_by_sector, sector) || continue
    @printf("  %-11s     %5.2f      %5.2f    %+5.2f\n",
            sector, train_rmse_by_sector[sector]*100,
            test_rmse_by_sector[sector]*100,
            (test_rmse_by_sector[sector] - train_rmse_by_sector[sector])*100)
end
