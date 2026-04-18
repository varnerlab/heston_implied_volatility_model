"""
Temporal Holdout Calibration: Out-of-Sample Generalization Test

Trains parametric, shared-NN, and sector-NN psi models on the first two
capture days (2026-04-14, 2026-04-15) and evaluates on the held-out next
two days (2026-04-16, 2026-04-17). The held-out days share the same 31
tickers, so per-ticker theta_base learned in training transfers directly.

This is the experiment the paper flagged as deferred — once a 3rd capture
day landed, generalization could be measured. With 4 days now available,
we can split 2-vs-2 for a more stable comparison.

Reports for each model:
  - Train RMSE (in-sample)
  - Test  RMSE (out-of-sample, the headline number)
  - Per-sector test RMSE
  - Per-ticker test RMSE

Standardization stats are computed from TRAIN ONLY and reused on TEST so
the test set is not allowed to influence the input scaling.
"""

using CSV
using DataFrames
using Statistics
using Optim
using Flux
using Printf
using Random

# ============================================================================
# Configuration
# ============================================================================

const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladder")
const TRAIN_DAYS = ["options-04-14-2026", "options-04-15-2026"]
const TEST_DAYS  = ["options-04-16-2026", "options-04-17-2026"]

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

# ============================================================================
# Data loading (same filters as production scripts)
# ============================================================================

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

function load_split(day_dirs::Vector{String})
    frames = DataFrame[]
    for d in day_dirs
        full = joinpath(LADDER_DIR, d)
        for f in readdir(full)
            endswith(f, ".csv") || continue
            df = load_ladder(joinpath(full, f))
            nrow(df) > 0 && push!(frames, df)
        end
    end
    out = vcat(frames...)
    out[!, :sector] = [get(SECTORS, t, "Other") for t in out.ticker]
    return out
end

println("Loading train split: $TRAIN_DAYS")
train = load_split(TRAIN_DAYS)
println("  $(nrow(train)) observations across $(length(unique(train.ticker))) tickers")

println("Loading test split:  $TEST_DAYS")
test = load_split(TEST_DAYS)
println("  $(nrow(test)) observations across $(length(unique(test.ticker))) tickers")

# Use the train-side ticker order for consistent indexing across train/test
tickers = sort(unique(train.ticker))
n_tickers = length(tickers)
ticker_idx = Dict(t => i for (i, t) in enumerate(tickers))
sectors = sort(unique(train.sector))

# Tickers in test but not train (would have no theta_base)
missing_in_train = setdiff(unique(test.ticker), tickers)
if !isempty(missing_in_train)
    println("  WARNING: dropping test obs for tickers not in train: $missing_in_train")
    test = test[[t in tickers for t in test.ticker], :]
end

# ============================================================================
# Train-side standardization stats (apply to test as-is)
# ============================================================================

const MU_DTE = mean(log.(max.(Float64.(train.actual_dte), 1.0)))
const SIGMA_DTE = std(log.(max.(Float64.(train.actual_dte), 1.0)))
const MU_M = mean(log.(Float64.(train.moneyness)))
const SIGMA_M = std(log.(Float64.(train.moneyness)))

# ============================================================================
# Helpers
# ============================================================================

function rmse(pred, target)
    sqrt(mean((Float64.(pred) .- Float64.(target)).^2))
end

function build_obs_arrays(df::DataFrame)
    log_dte = log.(max.(Float64.(df.actual_dte), 1.0))
    log_m = log.(Float64.(df.moneyness))
    iv = Float64.(df.implied_vol)
    tidx = [ticker_idx[t] for t in df.ticker]
    return log_dte, log_m, iv, tidx
end

train_log_dte, train_log_m, train_iv, train_tidx = build_obs_arrays(train)
test_log_dte,  test_log_m,  test_iv,  test_tidx  = build_obs_arrays(test)

# ============================================================================
# Model 1: Parametric (5 betas + per-ticker theta_base)
# ============================================================================

println("\n" * "="^70)
println("  MODEL 1: PARAMETRIC psi (5 betas)")
println("="^70)

eval_psi_param(beta, log_dte, log_m) =
    exp(beta[1]*log_dte + beta[2]*log_m + beta[3]*log_dte*log_m +
        beta[4]*log_m^2 + beta[5]*log_dte^2)

function param_objective(x)
    beta = @view x[1:5]
    log_theta = @view x[6:end]
    err = 0.0
    @inbounds for i in 1:length(train_iv)
        theta_base = exp(log_theta[train_tidx[i]])
        psi_val = eval_psi_param(beta, train_log_dte[i], train_log_m[i])
        sigma_model = sqrt(max(theta_base * psi_val, 1e-10))
        err += (sigma_model - train_iv[i])^2
    end
    return err / length(train_iv)
end

beta_init = [0.3, -1.0, 0.1, 2.0, -0.15]
theta_init = [mean(train_iv[train_tidx .== i])^2 for i in 1:n_tickers]
x0 = vcat(beta_init, log.(theta_init))

println("  Optimizing (Nelder-Mead, 100k iter x 2 passes)...")
r1 = optimize(param_objective, x0, NelderMead(),
              Optim.Options(iterations=100_000, show_trace=false))
r2 = optimize(param_objective, Optim.minimizer(r1), NelderMead(),
              Optim.Options(iterations=100_000, show_trace=false))
x_opt = Optim.minimizer(r2)
beta_opt = x_opt[1:5]
theta_opt = exp.(x_opt[6:end])

function predict_parametric(log_dte_v, log_m_v, tidx_v)
    out = Vector{Float64}(undef, length(log_dte_v))
    @inbounds for i in eachindex(out)
        psi_val = eval_psi_param(beta_opt, log_dte_v[i], log_m_v[i])
        out[i] = sqrt(max(theta_opt[tidx_v[i]] * psi_val, 1e-10))
    end
    return out
end

param_train_pred = predict_parametric(train_log_dte, train_log_m, train_tidx)
param_test_pred  = predict_parametric(test_log_dte,  test_log_m,  test_tidx)
param_train_rmse = rmse(param_train_pred, train_iv)
param_test_rmse  = rmse(param_test_pred,  test_iv)

@printf("  Train RMSE: %5.2f%%\n", param_train_rmse * 100)
@printf("  Test  RMSE: %5.2f%%\n", param_test_rmse  * 100)

# ============================================================================
# Model 2: Shared neural network psi
# ============================================================================

println("\n" * "="^70)
println("  MODEL 2: SHARED NEURAL NETWORK psi")
println("="^70)

function standardize(log_dte_v, log_m_v)
    s_dte = Float32.((log_dte_v .- MU_DTE) ./ SIGMA_DTE)
    s_m   = Float32.((log_m_v .- MU_M)   ./ SIGMA_M)
    return hcat(s_dte, s_m)'  # 2 x N
end

X_train = standardize(train_log_dte, train_log_m)
X_test  = standardize(test_log_dte,  test_log_m)
y_train = Float32.(train_iv)
y_test  = Float32.(test_iv)
tidx_train_i32 = Int32.(train_tidx)
tidx_test_i32  = Int32.(test_tidx)

function compute_model_ivs_nn(psi_nn, log_theta, X, tidx)
    log_psi = psi_nn(X)
    log_theta_obs = log_theta[tidx]
    return exp.(Float32(0.5) .* (log_theta_obs .+ vec(log_psi)))
end

Random.seed!(42)
psi_nn = Chain(Dense(2 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 1))
log_theta_nn = Float32[Float32(log(mean(train_iv[train_tidx .== i])^2)) for i in 1:n_tickers]
model_nn = (psi_nn = psi_nn, log_theta = log_theta_nn)
opt_nn = Flux.setup(Flux.Adam(1e-3), model_nn)

best_loss = Inf
best_state = nothing
no_improve = 0
const SHARED_EPOCHS = 2000
const SHARED_PATIENCE = 200

println("  Training shared NN (up to $SHARED_EPOCHS epochs, patience=$SHARED_PATIENCE)...")
for epoch in 1:SHARED_EPOCHS
    l, grads = Flux.withgradient(model_nn) do m
        Flux.mse(compute_model_ivs_nn(m.psi_nn, m.log_theta, X_train, tidx_train_i32), y_train)
    end
    Flux.update!(opt_nn, model_nn, grads[1])
    if l < best_loss
        global best_loss = l
        global best_state = Flux.state(model_nn)
        global no_improve = 0
    else
        global no_improve += 1
    end
    if epoch % 250 == 0
        @printf("    epoch %4d  train RMSE=%.2f%%  best=%.2f%%\n",
                epoch, sqrt(l)*100, sqrt(best_loss)*100)
    end
    no_improve >= SHARED_PATIENCE && (println("  Early stop at epoch $epoch"); break)
    epoch == 500  && Flux.adjust!(opt_nn, 5e-4)
    epoch == 1000 && Flux.adjust!(opt_nn, 2e-4)
    epoch == 1500 && Flux.adjust!(opt_nn, 1e-4)
end
Flux.loadmodel!(model_nn, best_state)

shared_train_pred = Float64.(compute_model_ivs_nn(model_nn.psi_nn, model_nn.log_theta, X_train, tidx_train_i32))
shared_test_pred  = Float64.(compute_model_ivs_nn(model_nn.psi_nn, model_nn.log_theta, X_test,  tidx_test_i32))
shared_train_rmse = rmse(shared_train_pred, train_iv)
shared_test_rmse  = rmse(shared_test_pred,  test_iv)

@printf("  Train RMSE: %5.2f%%\n", shared_train_rmse * 100)
@printf("  Test  RMSE: %5.2f%%\n", shared_test_rmse  * 100)

# ============================================================================
# Model 3: Sector-specific neural network psi
# ============================================================================

println("\n" * "="^70)
println("  MODEL 3: SECTOR-SPECIFIC NEURAL NETWORK psi")
println("="^70)

function make_psi_nn(n_obs::Int, seed::Int)
    Random.seed!(seed)
    if n_obs >= 2000
        return Chain(Dense(2 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 1))
    else
        return Chain(Dense(2 => 8, tanh), Dense(8 => 8, tanh), Dense(8 => 1))
    end
end

const SECTOR_EPOCHS = 2000
const SECTOR_PATIENCE = 200

# Per-sector models keyed by sector name. For each sector we keep:
#   - the trained Flux model (psi_nn + log_theta)
#   - its local sector_ticker_idx mapping
sector_models = Dict{String,Any}()

for sector in sectors
    train_mask = train.sector .== sector
    sector_train = train[train_mask, :]
    sector_tickers = sort(unique(sector_train.ticker))
    s_tidx = Dict(t => i for (i, t) in enumerate(sector_tickers))
    n_obs = nrow(sector_train)

    log_dte_s = Float32.((log.(max.(Float64.(sector_train.actual_dte), 1.0)) .- MU_DTE) ./ SIGMA_DTE)
    log_m_s = Float32.((log.(Float64.(sector_train.moneyness)) .- MU_M) ./ SIGMA_M)
    Xs = hcat(log_dte_s, log_m_s)'
    ys = Float32.(sector_train.implied_vol)
    tidx_s = Int32[s_tidx[t] for t in sector_train.ticker]

    psi_s = make_psi_nn(n_obs, 42)
    log_theta_s = Float32[Float32(log(mean(Float64.(sector_train.implied_vol[sector_train.ticker .== t]))^2))
                          for t in sector_tickers]
    model_s = (psi_nn = psi_s, log_theta = log_theta_s)
    opt_s = Flux.setup(Flux.Adam(1e-3), model_s)

    arch = n_obs >= 2000 ? "2->16->16->1" : "2->8->8->1"
    @printf("  [%s] %d train obs, %d tickers, arch=%s\n",
            sector, n_obs, length(sector_tickers), arch)

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

    sector_models[sector] = (model = model_s, ticker_idx = s_tidx)
end

# Score sector NN on both splits
function predict_sector_nn(df::DataFrame)
    out = zeros(Float64, nrow(df))
    for sector in sectors
        mask = df.sector .== sector
        any(mask) || continue
        sm = sector_models[sector]
        log_dte_s = Float32.((log.(max.(Float64.(df.actual_dte[mask]), 1.0)) .- MU_DTE) ./ SIGMA_DTE)
        log_m_s   = Float32.((log.(Float64.(df.moneyness[mask])) .- MU_M) ./ SIGMA_M)
        Xs = hcat(log_dte_s, log_m_s)'
        # Tickers in this sector, mapped via the sector model's local index.
        # Drop any rows whose ticker wasn't seen in this sector's training set.
        local_tidx = [get(sm.ticker_idx, t, 0) for t in df.ticker[mask]]
        keep = local_tidx .> 0
        if !all(keep)
            # Should not happen with the same 31 tickers across days, but guard anyway
            @warn "Sector $sector has $(sum(.!keep)) test obs with unknown ticker, scoring as NaN"
        end
        preds = fill(NaN, sum(mask))
        if any(keep)
            tidx_local = Int32.(local_tidx[keep])
            X_keep = Xs[:, keep]
            preds[keep] .= Float64.(compute_model_ivs_nn(sm.model.psi_nn, sm.model.log_theta, X_keep, tidx_local))
        end
        out[findall(mask)] .= preds
    end
    return out
end

sector_train_pred = predict_sector_nn(train)
sector_test_pred  = predict_sector_nn(test)
sector_train_rmse = rmse(sector_train_pred, train_iv)
sector_test_rmse  = rmse(sector_test_pred,  test_iv)

@printf("  Train RMSE: %5.2f%%\n", sector_train_rmse * 100)
@printf("  Test  RMSE: %5.2f%%\n", sector_test_rmse  * 100)

# ============================================================================
# Three-way comparison
# ============================================================================

println("\n" * "="^70)
println("  TEMPORAL HOLDOUT — THREE-WAY COMPARISON")
println("="^70)
println("\n  Train: $(TRAIN_DAYS), $(nrow(train)) obs")
println("  Test:  $(TEST_DAYS), $(nrow(test)) obs\n")

println("  Model                   Train RMSE(%)   Test RMSE(%)   Gen gap(%)")
println("  " * "-"^70)
@printf("  Parametric (5 betas)        %5.2f           %5.2f         %+5.2f\n",
        param_train_rmse*100, param_test_rmse*100, (param_test_rmse - param_train_rmse)*100)
@printf("  Shared NN (1 network)       %5.2f           %5.2f         %+5.2f\n",
        shared_train_rmse*100, shared_test_rmse*100, (shared_test_rmse - shared_train_rmse)*100)
@printf("  Sector NN (6 networks)      %5.2f           %5.2f         %+5.2f\n",
        sector_train_rmse*100, sector_test_rmse*100, (sector_test_rmse - sector_train_rmse)*100)

# Per-sector test RMSE
println("\n  Per-sector test RMSE (%):")
println("  Sector         N_test   Parametric   Shared NN   Sector NN")
println("  " * "-"^65)
for sector in sectors
    mask = test.sector .== sector
    n = sum(mask)
    n == 0 && continue
    p = rmse(param_test_pred[mask],  test_iv[mask])
    s = rmse(shared_test_pred[mask], test_iv[mask])
    sn = rmse(sector_test_pred[mask], test_iv[mask])
    @printf("  %-12s   %5d     %5.2f        %5.2f       %5.2f\n",
            sector, n, p*100, s*100, sn*100)
end

# Per-ticker test RMSE (sorted by sector NN test RMSE descending)
println("\n  Per-ticker test RMSE (%):")
println("  Ticker  N_test   Parametric   Shared NN   Sector NN  Sector")
println("  " * "-"^65)
ticker_rows = Tuple{String,Int,Float64,Float64,Float64,String}[]
for t in tickers
    mask = test.ticker .== t
    n = sum(mask)
    n == 0 && continue
    p = rmse(param_test_pred[mask],  test_iv[mask]) * 100
    s = rmse(shared_test_pred[mask], test_iv[mask]) * 100
    sn = rmse(sector_test_pred[mask], test_iv[mask]) * 100
    push!(ticker_rows, (t, n, p, s, sn, get(SECTORS, t, "Other")))
end
sort!(ticker_rows, by=r -> -r[5])
for (t, n, p, s, sn, sec) in ticker_rows
    @printf("  %-5s   %5d    %5.2f        %5.2f       %5.2f     %s\n", t, n, p, s, sn, sec)
end

println("\nDone.")
