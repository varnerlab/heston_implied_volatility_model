"""
Horvath-style direct neural IV-surface baseline (Reviewer 3).

Trains one global MLP that maps (ln τ, ln m, d2e_self, d2e_peer,
ticker-embedding) → IV directly, **without** the θ_i · ψ decomposition
used elsewhere in the paper. This is the head-to-head Reviewer 3 asked
for: it asks "is the θ·ψ decomposition adding value over a single dense
net with a ticker embedding?"

Same train/test split as Configuration A/C in temporal_holdout_earnings.jl
so the comparison against Config A (13.03%) and C (11.98%) is direct.

Architecture: (2 + 2 + 8) → 32 → 32 → 1 with tanh, ~1.5k params.
The 8-dim ticker embedding × 31 tickers = 248 embedding params, below
the per-ticker NN's 369 ψ params so capacity comparison stays honest.

Output:
  code/figures/horvath_direct_summary.csv
"""

using CSV
using DataFrames
using Statistics
using Flux
using Printf
using Random
using Dates

include(joinpath(@__DIR__, "..", "src", "TemporalFolds.jl"))
using .TemporalFolds

const LADDER_DIR   = joinpath(@__DIR__, "..", "data", "ladder")
const EARNINGS_CSV = joinpath(@__DIR__, "..", "data", "earnings", "earnings_calendar.csv")
const FIG_DIR      = joinpath(@__DIR__, "..", "figures")
const TRAIN_DAYS = ["options-04-14-2026", "options-04-15-2026",
                    "options-04-16-2026", "options-04-17-2026",
                    "options-04-21-2026", "options-04-22-2026"]
const TEST_DAYS  = ["options-04-23-2026", "options-04-24-2026"]
const EMBED_DIM  = 8
const HIDDEN     = 32
const N_EPOCHS   = 2000
const PATIENCE   = 200

println("Loading earnings calendar...")
cal = load_earnings_calendar(EARNINGS_CSV)

println("Loading train+test splits...")
train = load_split(LADDER_DIR, TRAIN_DAYS)
test  = load_split(LADDER_DIR, TEST_DAYS)
attach_earnings_features!(train, cal)
attach_earnings_features!(test,  cal)
println("  train: $(nrow(train)) obs    test: $(nrow(test)) obs")

# Ticker embedding indexing — built from train universe; test rows for
# unseen tickers are dropped (matches Configuration A semantics).
tickers = String.(sort(unique(train.ticker)))
tidx = Dict(t => i for (i, t) in enumerate(tickers))
test = test[[haskey(tidx, t) for t in test.ticker], :]
n_tickers = length(tickers)
@printf("  %d tickers in universe\n", n_tickers)

# Train-only standardization (4-input: ln_dte, ln_m, d2e_self, d2e_peer)
function _std_z(v, μ, σ); (Float64.(v) .- μ) ./ σ; end
mu_dte = mean(log.(max.(Float64.(train.actual_dte), 1.0)))
sg_dte = std(log.(max.(Float64.(train.actual_dte), 1.0)))
mu_m   = mean(log.(Float64.(train.moneyness)))
sg_m   = std(log.(Float64.(train.moneyness)))
mu_ds  = mean(Float64.(train.d2e_self));      sg_ds = max(std(Float64.(train.d2e_self)), 1e-3)
mu_dp  = mean(Float64.(train.d2e_peer_min));  sg_dp = max(std(Float64.(train.d2e_peer_min)), 1e-3)

function build_X(df)
    z_dte  = Float32.(_std_z(log.(max.(Float64.(df.actual_dte), 1.0)), mu_dte, sg_dte))
    z_m    = Float32.(_std_z(log.(Float64.(df.moneyness)), mu_m, sg_m))
    z_ds   = Float32.(_std_z(Float64.(df.d2e_self), mu_ds, sg_ds))
    z_dp   = Float32.(_std_z(Float64.(df.d2e_peer_min), mu_dp, sg_dp))
    X = hcat(z_dte, z_m, z_ds, z_dp)'      # 4 × N
    tids = Int32[tidx[t] for t in df.ticker]
    return X, tids
end

Xtr, tids_tr = build_X(train)
Xte, tids_te = build_X(test)
ytr = Float32.(train.implied_vol)
yte = Float32.(test.implied_vol)
@printf("  Xtr size: %s    ytr size: %s\n", size(Xtr), size(ytr))

# Embedding: each ticker → 8-dim vector. We use Flux.Embedding (32-dim hidden
# trunk takes 4 input features + 8 embedding dim = 12 input). The forward
# pass concatenates embedding(tickers) with X then runs the MLP.
Random.seed!(20260514)
embed = Embedding(n_tickers => EMBED_DIM)
trunk = Chain(
    Dense(4 + EMBED_DIM => HIDDEN, tanh),
    Dense(HIDDEN => HIDDEN, tanh),
    Dense(HIDDEN => 1),
)
model = (embed=embed, trunk=trunk)

function forward(m, X, tids)
    e = m.embed(tids)               # EMBED_DIM × N
    z = vcat(X, e)                  # (4+EMBED_DIM) × N
    return vec(m.trunk(z))
end

n_embed_params = sum(length, Flux.params(embed))
n_trunk_params = sum(length, Flux.params(trunk))
@printf("  embedding params: %d   trunk params: %d   total: %d\n",
        n_embed_params, n_trunk_params, n_embed_params + n_trunk_params)

function train_horvath!(model, Xtr, ytr, tids_tr; n_epochs::Int, patience_max::Int)
    opt = Flux.setup(Flux.Adam(1e-3), model)
    best_loss = Inf
    best_state = nothing
    patience = 0
    for epoch in 1:n_epochs
        l, g = Flux.withgradient(model) do m
            Flux.mse(forward(m, Xtr, tids_tr), ytr)
        end
        Flux.update!(opt, model, g[1])
        if l < best_loss
            best_loss = l
            best_state = Flux.state(model)
            patience = 0
        else
            patience += 1
        end
        if patience >= patience_max
            @printf("  early stop at epoch %d (patience %d)\n", epoch, patience_max)
            break
        end
        epoch == 500  && Flux.adjust!(opt, 5e-4)
        epoch == 1000 && Flux.adjust!(opt, 2e-4)
        epoch == 1500 && Flux.adjust!(opt, 1e-4)
        if epoch % 100 == 0
            @printf("  epoch %4d  train_mse=%.5f  best=%.5f\n", epoch, l, best_loss)
        end
    end
    best_state === nothing || Flux.loadmodel!(model, best_state)
end

println("\nTraining Horvath direct NN...")
train_horvath!(model, Xtr, ytr, tids_tr; n_epochs=N_EPOCHS, patience_max=PATIENCE)

# Evaluation
train_pred = Float64.(forward(model, Xtr, tids_tr))
test_pred  = Float64.(forward(model, Xte, tids_te))
train_iv   = Float64.(ytr)
test_iv    = Float64.(yte)
train_rmse_v = rmse(train_pred, train_iv)
test_rmse_v  = rmse(test_pred,  test_iv)

@printf("\n  Train RMSE: %5.2f%%   Test RMSE: %5.2f%%   Gen gap: %+5.2f%%\n",
        train_rmse_v*100, test_rmse_v*100, (test_rmse_v - train_rmse_v)*100)

# Per-sector breakdown
println("\nPer-sector test RMSE (%):")
sectors_sorted = sort(unique(test.sector))
per_sector = Dict{String,Float64}()
println("  Sector         N_test   RMSE")
println("  " * "-"^35)
for s in sectors_sorted
    mask = test.sector .== s
    n = sum(mask)
    n == 0 && continue
    r = rmse(test_pred[mask], test_iv[mask]) * 100
    per_sector[s] = r
    @printf("  %-12s   %5d    %5.2f\n", s, n, r)
end

# Persist tidy CSV row
mkpath(FIG_DIR)
out = joinpath(FIG_DIR, "horvath_direct_summary.csv")
row = DataFrame(
    config = ["D_horvath_direct_8d_embedding"],
    n_train = [nrow(train)],
    n_test  = [nrow(test)],
    train_rmse = [train_rmse_v],
    test_rmse  = [test_rmse_v],
    embed_dim  = [EMBED_DIM],
    hidden_dim = [HIDDEN],
    n_params   = [n_embed_params + n_trunk_params],
)
CSV.write(out, row)
@printf("\n[csv] wrote -> %s\n", out)

println("\nDone.")
