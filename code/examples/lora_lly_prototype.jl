"""
LoRA-adapter prototype for daily ψ_NN refitting on LLY 04-28.

Tests the user's "Variant 1 (rank-2 LoRA on shape + θ level update) with the
2+1 combined trigger" idea on a known failure case: the LLY 04-28 capture
where the time-averaged ψ_NN gives entry edges of about -\$1.35 and -\$1.80
against the market mid (see §5 / Supplementary Table tab:lly_scenario).

Pipeline:
1. Drift trigger:  base-ψ-vs-market RMSE on LLY for each of the 15 ladder
                   dates, z-scored against the trailing 4-date window.
                   If 04-28 is flagged as an outlier, the drift trigger
                   would have fired and requested a refit.

2. Variant 0:      Train ONLY \bar log θ_LLY on the 04-28 slice; ψ_NN frozen.
                   Single-scalar daily level update. Cheapest possible adapter.

3. Variant 1:      Train rank-2 LoRA adapters on each Dense layer of ψ_NN
                   + the same scalar log θ_LLY, on the 04-28 slice.
                   ~50 trainable params total. Base ψ remains frozen.

4. Re-price:       LR-American price each variant's IV at K_p=\$825 / K_c=\$940
                   and compare entry edge against the unchanged market mid.

Forward simulation property: the BASE ψ_NN stays frozen, so the §5 forward
scenarios are unaffected. The LoRA adapter is a mark-to-market overlay
applied only at calibration time for daily entry edges.

Run:
    julia --project=. examples/lora_lly_prototype.jl
"""

using CSV
using DataFrames
using Dates
using Flux
using JLD2
using Printf
using Random
using Statistics

using HestonIV
include(joinpath(@__DIR__, "..", "src", "ScenarioTemplate.jl"))
using .ScenarioTemplate

# ============================================================================
# Configuration
# ============================================================================

const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladder")
const NN_CACHE   = joinpath(@__DIR__, "..", "figures",
                            "calibrate_ladders_per_ticker_nn_cache.jld2")

const TICKER     = "LLY"
const SECTOR     = "Healthcare"
const ANCHOR     = Date("2026-04-28")
const K_PUT      = 825.0
const K_CALL     = 940.0
const MID_PUT    = 23.30
const MID_CALL   = 20.76
const IV_PUT_M   = 0.444
const IV_CALL_M  = 0.440
const T_DAYS     = 31
const R_FREE     = 0.0425
const Q_DIV      = 0.0
const N_LR       = 201

const ALL_DAYS = [
    "options-04-14-2026", "options-04-15-2026", "options-04-16-2026",
    "options-04-17-2026", "options-04-21-2026", "options-04-22-2026",
    "options-04-23-2026", "options-04-24-2026", "options-04-27-2026",
    "options-04-28-2026", "options-04-29-2026", "options-05-01-2026",
    "options-05-06-2026", "options-05-08-2026", "options-05-11-2026",
]

const RANK       = 2
const ALPHA      = 2.0     # LoRA scaling (typical: alpha = rank)
const N_EPOCHS   = 1500
const PATIENCE   = 150
const SEED       = 20260514

# ============================================================================
# Load LLY corpus + recompute standardisation (must match base ψ_NN training)
# ============================================================================

function _load_ladder(filepath)
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
        (df.actual_dte .> 0), :]
    return valid
end

function _load_all_ladders(dir)
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
        df = _load_ladder(f)
        nrow(df) > 0 && push!(frames, df)
    end
    return vcat(frames...)
end

println("Loading 31-ticker ladder corpus for standardisation constants...")
all_data = _load_all_ladders(LADDER_DIR)
const MU_DTE    = mean(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const SIGMA_DTE = std(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const MU_M      = mean(log.(Float64.(all_data.moneyness)))
const SIGMA_M   = std(log.(Float64.(all_data.moneyness)))

# Per-date LLY slices keyed by date directory
function load_lly_slice(date_dir)
    full = joinpath(LADDER_DIR, date_dir)
    isdir(full) || return DataFrame()
    files = filter(f -> startswith(f, TICKER * "_") && endswith(f, ".csv") &&
                        occursin("_dte_ladder_", f), readdir(full))
    isempty(files) && return DataFrame()
    return _load_ladder(joinpath(full, files[1]))
end

# Build (X, y) with the SAME standardisation as the base ψ_NN
function build_Xy(slice::DataFrame)
    z_dte = Float32.((log.(max.(Float64.(slice.actual_dte), 1.0)) .- MU_DTE) ./ SIGMA_DTE)
    z_m   = Float32.((log.(Float64.(slice.moneyness)) .- MU_M) ./ SIGMA_M)
    X = vcat(z_dte', z_m')           # 2 × N
    y = Float32.(slice.implied_vol)  # N
    return X, y
end

# ============================================================================
# Restore base ψ_NN^LLY
# ============================================================================

println("Restoring per-ticker ψ_NN^$(TICKER) from cache...")
nn_cache = JLD2.load(NN_CACHE)
base_psi, log_theta_base, src = ScenarioTemplate._restore_nn(
    nn_cache, TICKER; use_per_ticker=true, sector=SECTOR)
@assert src === :per_ticker  "expected per-ticker payload for LLY"
@printf("  source = %s   base log θ = %+.4f   base θ̄ = %.4f   base IV-level = %.1f%%\n",
        src, log_theta_base, exp(log_theta_base), 100*sqrt(exp(log_theta_base)))

# Base layers — Dense(2→16, tanh), Dense(16→16, tanh), Dense(16→1, identity)
const L1 = base_psi.layers[1]
const L2 = base_psi.layers[2]
const L3 = base_psi.layers[3]
@assert size(L1.weight) == (16, 2)  "expected per-ticker LLY ψ_NN architecture"
@assert size(L2.weight) == (16, 16)
@assert size(L3.weight) == (1, 16)

function base_log_psi(X)
    h1 = tanh.(L1.weight * X .+ L1.bias)
    h2 = tanh.(L2.weight * h1 .+ L2.bias)
    return vec(L3.weight * h2 .+ L3.bias)
end

function base_iv(X)
    return exp.(Float32(0.5) .* (Float32(log_theta_base) .+ base_log_psi(X)))
end

# ============================================================================
# Step 1: drift-trigger signal — per-date base-model RMSE on LLY
# ============================================================================

println("\n" * "="^78)
println("  Step 1: drift-trigger signal (base ψ_NN^LLY RMSE on each ladder date)")
println("="^78)
println("  Date          N_obs   RMSE(% IV)   z-score(vs trailing 4)   refit-recommended?")
println("  " * "-"^78)

function compute_drift_trigger()
    per_date_rmse = Float64[]
    per_date_n    = Int[]
    per_date_d    = String[]
    flagged       = Bool[]
    z_thresh      = 1.5
    for d in ALL_DAYS
        slice = load_lly_slice(d)
        nrow(slice) == 0 && (println("  $(d)   missing"); continue)
        X, y = build_Xy(slice)
        σ_b = Float64.(base_iv(X))
        rmse_d = sqrt(mean((σ_b .- Float64.(y)).^2))
        push!(per_date_rmse, rmse_d)
        push!(per_date_n, nrow(slice))
        push!(per_date_d, d)
        # Robust trigger: median + MAD over the trailing 4 dates (resistant to a
        # single outlier in the window, which a mean+std trigger fails on).
        if length(per_date_rmse) > 4
            trail = per_date_rmse[end-4:end-1]
            m = median(trail)
            mad = median(abs.(trail .- m))
            scaled_mad = 1.4826 * max(mad, 1e-4)   # MAD-to-σ consistency factor
            z = (rmse_d - m) / scaled_mad
            flag = z >= z_thresh
            @printf("  %s   %5d    %5.2f         %+6.2f                 %s\n",
                    d, nrow(slice), 100*rmse_d, z, flag ? "YES" : "no")
            push!(flagged, flag)
        else
            @printf("  %s   %5d    %5.2f         (warm-up)\n", d, nrow(slice), 100*rmse_d)
            push!(flagged, false)
        end
    end
    return per_date_d, per_date_n, per_date_rmse, flagged
end

per_date_d, per_date_n, per_date_rmse, flagged = compute_drift_trigger()

@printf("\n  Median per-date RMSE: %.2f%%   Max: %.2f%% on %s   Trigger: median + 1.5 × scaled-MAD\n",
        100*median(per_date_rmse), 100*maximum(per_date_rmse),
        per_date_d[argmax(per_date_rmse)])
@printf("  Refit-flagged dates: %s\n",
        join([per_date_d[i] for i in 1:length(flagged) if flagged[i]], ", "))

# ============================================================================
# Step 2: Variant 0 — level-only update (train just \bar log θ_LLY)
# ============================================================================

slice_28 = load_lly_slice("options-04-28-2026")
X28, y28 = build_Xy(slice_28)
S0_28 = Float64(slice_28.und_close[1])
y28_f64 = Float64.(y28)

# Base-only RMSE on 04-28
σ_base_28 = Float64.(base_iv(X28))
rmse_base_28 = sqrt(mean((σ_base_28 .- y28_f64).^2))

println("\n" * "="^78)
println("  Step 2: Variant 0 — daily \\bar log θ update only (ψ frozen)")
println("="^78)
@printf("  Base-only RMSE on 04-28 slice (%d obs): %.2f%% IV\n", nrow(slice_28), 100*rmse_base_28)

# Train a single scalar log_theta
function model_iv_v0(p, X)
    return exp.(Float32(0.5) .* (p.log_theta[1] .+ base_log_psi(X)))
end

function train_v0(X, y; lr=5e-3, n_epochs=500, patience_max=50)
    Random.seed!(SEED)
    log_θ = Float32[log_theta_base]
    params = (log_theta = log_θ,)
    opt = Flux.setup(Flux.Adam(lr), params)
    best, bs, pat = Inf, log_θ[1], 0
    for epoch in 1:n_epochs
        l, g = Flux.withgradient(params) do p
            Flux.mse(model_iv_v0(p, X), y)
        end
        Flux.update!(opt, params, g[1])
        if l < best; best = l; bs = params.log_theta[1]; pat = 0
        else; pat += 1; end
        pat >= patience_max && break
    end
    params.log_theta[1] = bs
    return params, best
end

log_θ_v0_param, _ = train_v0(X28, y28)
σ_v0_28 = Float64.(model_iv_v0(log_θ_v0_param, X28))
rmse_v0_28 = sqrt(mean((σ_v0_28 .- y28_f64).^2))
δ_log_θ_v0 = log_θ_v0_param.log_theta[1] - Float32(log_theta_base)
@printf("  After level-only update:     RMSE = %.2f%% IV    Δlog θ = %+.4f    new IV-level = %.1f%%\n",
        100*rmse_v0_28, δ_log_θ_v0, 100*sqrt(exp(Float64(log_θ_v0_param.log_theta[1]))))

# ============================================================================
# Step 3: Variant 1 — rank-2 LoRA on each Dense + the same daily \bar log θ
# ============================================================================

println("\n" * "="^78)
println("  Step 3: Variant 1 — rank-$(RANK) LoRA on each Dense + level update")
println("="^78)

# Adapter struct: low-rank deltas on each layer's weight matrix.
# Layer i:  W_new = W_base + (alpha/rank) * B_i * A_i
# Init:     A ~ Kaiming-small,  B = 0  → adapter is no-op at init.
mutable struct LoRAAdapter
    A1::Matrix{Float32};  B1::Matrix{Float32}
    A2::Matrix{Float32};  B2::Matrix{Float32}
    A3::Matrix{Float32};  B3::Matrix{Float32}
    log_theta::Vector{Float32}
end

Flux.@layer LoRAAdapter

function make_adapter(rng, log_theta_init::Real)
    init_A(d_in)  = Float32.(randn(rng, RANK, d_in) ./ Float32(sqrt(d_in)))
    init_B(d_out) = zeros(Float32, d_out, RANK)
    return LoRAAdapter(
        init_A(2),  init_B(16),     # Dense(2 → 16)
        init_A(16), init_B(16),     # Dense(16 → 16)
        init_A(16), init_B(1),      # Dense(16 → 1)
        Float32[log_theta_init],
    )
end

const SCALE = Float32(ALPHA / RANK)

function lora_log_psi(a::LoRAAdapter, X)
    h1 = tanh.(L1.weight * X .+ L1.bias .+ SCALE .* (a.B1 * (a.A1 * X)))
    h2 = tanh.(L2.weight * h1 .+ L2.bias .+ SCALE .* (a.B2 * (a.A2 * h1)))
    return vec(L3.weight * h2 .+ L3.bias .+ SCALE .* (a.B3 * (a.A3 * h2)))
end

function model_iv_v1(a::LoRAAdapter, X)
    return exp.(Float32(0.5) .* (a.log_theta[1] .+ lora_log_psi(a, X)))
end

# Sanity-check: at init (B = 0), adapter forward must equal base forward.
adapter_init = make_adapter(MersenneTwister(SEED), log_theta_base)
σ_init_check = Float64.(model_iv_v1(adapter_init, X28))
@assert maximum(abs.(σ_init_check .- σ_base_28)) < 1e-5  "LoRA init not a no-op"
@printf("  LoRA init is a no-op (max |σ_init − σ_base| = %.2e on 04-28 slice) ✓\n",
        maximum(abs.(σ_init_check .- σ_base_28)))

function train_v1!(adapter, X, y; lr=1e-3, n_epochs=N_EPOCHS, patience_max=PATIENCE)
    opt = Flux.setup(Flux.Adam(lr), adapter)
    best, bs, pat = Inf, deepcopy(Flux.state(adapter)), 0
    for epoch in 1:n_epochs
        l, g = Flux.withgradient(adapter) do a
            Flux.mse(model_iv_v1(a, X), y)
        end
        Flux.update!(opt, adapter, g[1])
        if l < best
            best = l
            bs = deepcopy(Flux.state(adapter))
            pat = 0
        else
            pat += 1
        end
        pat >= patience_max && (println("  early stop at epoch $(epoch)"); break)
        epoch == 500  && Flux.adjust!(opt, 5e-4)
        epoch == 1000 && Flux.adjust!(opt, 2e-4)
        epoch % 200 == 0 && @printf("    epoch %4d  loss = %.5f  best = %.5f\n", epoch, l, best)
    end
    Flux.loadmodel!(adapter, bs)
    return best
end

adapter = make_adapter(MersenneTwister(SEED), log_theta_base)
train_v1!(adapter, X28, y28)

σ_v1_28 = Float64.(model_iv_v1(adapter, X28))
rmse_v1_28 = sqrt(mean((σ_v1_28 .- y28_f64).^2))
δ_log_θ_v1 = adapter.log_theta[1] - Float32(log_theta_base)
@printf("\n  After rank-%d LoRA + level update:  RMSE = %.2f%% IV    Δlog θ = %+.4f\n",
        RANK, 100*rmse_v1_28, δ_log_θ_v1)
@printf("  Adapter param counts:  A1 %s, B1 %s, A2 %s, B2 %s, A3 %s, B3 %s   (total %d trainable)\n",
        size(adapter.A1), size(adapter.B1), size(adapter.A2), size(adapter.B2),
        size(adapter.A3), size(adapter.B3),
        length(adapter.A1) + length(adapter.B1) +
        length(adapter.A2) + length(adapter.B2) +
        length(adapter.A3) + length(adapter.B3) + 1)
@printf("  LoRA shape norms:  ‖B1A1‖_F = %.3f   ‖B2A2‖_F = %.3f   ‖B3A3‖_F = %.3f\n",
        sqrt(sum((adapter.B1 * adapter.A1).^2)),
        sqrt(sum((adapter.B2 * adapter.A2).^2)),
        sqrt(sum((adapter.B3 * adapter.A3).^2)))

# ============================================================================
# Step 4: re-price the LLY $825 put / $940 call under each variant
# ============================================================================

function strike_X(K, S, dte_days)
    z_dte = Float32((log(max(Float64(dte_days), 1.0)) - MU_DTE) / SIGMA_DTE)
    z_m   = Float32((log(K / S) - MU_M) / SIGMA_M)
    return reshape(Float32[z_dte, z_m], 2, 1)
end

iv_base_at(K, S, dte) = first(base_iv(strike_X(K, S, dte)))
function iv_v0_at(K, S, dte)
    p = (log_theta = log_θ_v0_param.log_theta,)
    return first(model_iv_v0(p, strike_X(K, S, dte)))
end
iv_v1_at(K, S, dte) = first(model_iv_v1(adapter, strike_X(K, S, dte)))

println("\n" * "="^78)
println("  Step 4: entry-edge comparison on the LLY 2026-05-29 \$825 put / \$940 call")
println("="^78)

put_b  = iv_base_at(K_PUT, S0_28, T_DAYS)
put_v0 = iv_v0_at(K_PUT, S0_28, T_DAYS)
put_v1 = iv_v1_at(K_PUT, S0_28, T_DAYS)
call_b  = iv_base_at(K_CALL, S0_28, T_DAYS)
call_v0 = iv_v0_at(K_CALL, S0_28, T_DAYS)
call_v1 = iv_v1_at(K_CALL, S0_28, T_DAYS)

@printf("\n  IV at strikes (anchor S_0 = \$%.2f):\n", S0_28)
@printf("    Put  K=\$%.0f T=%d:  base = %.1f%%   V0 (level) = %.1f%%   V1 (LoRA-r2) = %.1f%%   market = %.1f%%\n",
        K_PUT, T_DAYS, 100*put_b, 100*put_v0, 100*put_v1, 100*IV_PUT_M)
@printf("    Call K=\$%.0f T=%d:  base = %.1f%%   V0 (level) = %.1f%%   V1 (LoRA-r2) = %.1f%%   market = %.1f%%\n",
        K_CALL, T_DAYS, 100*call_b, 100*call_v0, 100*call_v1, 100*IV_CALL_M)

T_yrs = T_DAYS / 365.0
fv_put_b   = lr_american_price(S0_28, K_PUT,  Float64(put_b),  R_FREE, T_yrs, N_LR, :put;  q=Q_DIV)
fv_put_v0  = lr_american_price(S0_28, K_PUT,  Float64(put_v0), R_FREE, T_yrs, N_LR, :put;  q=Q_DIV)
fv_put_v1  = lr_american_price(S0_28, K_PUT,  Float64(put_v1), R_FREE, T_yrs, N_LR, :put;  q=Q_DIV)
fv_call_b  = lr_american_price(S0_28, K_CALL, Float64(call_b), R_FREE, T_yrs, N_LR, :call; q=Q_DIV)
fv_call_v0 = lr_american_price(S0_28, K_CALL, Float64(call_v0),R_FREE, T_yrs, N_LR, :call; q=Q_DIV)
fv_call_v1 = lr_american_price(S0_28, K_CALL, Float64(call_v1),R_FREE, T_yrs, N_LR, :call; q=Q_DIV)

@printf("\n  Entry-edge (model fair value − market mid) on the LR American pricer:\n")
@printf("    %-8s   %-10s   %-10s   %-10s   %-10s   %-10s   %-10s\n",
        "Contract", "Market", "Base FV", "V0 FV", "V1 FV", "Base edge", "V0 edge   V1 edge")
@printf("    %-8s   \$%-9.2f   \$%-9.2f   \$%-9.2f   \$%-9.2f   \$%+8.2f   \$%+8.2f   \$%+8.2f\n",
        "Put  $K_PUT",  MID_PUT,  fv_put_b,  fv_put_v0,  fv_put_v1,
        fv_put_b - MID_PUT, fv_put_v0 - MID_PUT, fv_put_v1 - MID_PUT)
@printf("    %-8s   \$%-9.2f   \$%-9.2f   \$%-9.2f   \$%-9.2f   \$%+8.2f   \$%+8.2f   \$%+8.2f\n",
        "Call $K_CALL", MID_CALL, fv_call_b, fv_call_v0, fv_call_v1,
        fv_call_b - MID_CALL, fv_call_v0 - MID_CALL, fv_call_v1 - MID_CALL)

# ============================================================================
# Persist a tidy CSV summary
# ============================================================================

out_dir = joinpath(@__DIR__, "..", "figures")
mkpath(out_dir)
summary_df = DataFrame(
    variant     = ["base", "V0 level-only", "V1 LoRA-r$(RANK) + level"],
    n_trainable = [0, 1,
                   length(adapter.A1) + length(adapter.B1) +
                   length(adapter.A2) + length(adapter.B2) +
                   length(adapter.A3) + length(adapter.B3) + 1],
    rmse_28_pct       = 100 .* [rmse_base_28, rmse_v0_28, rmse_v1_28],
    iv_put_pct        = 100 .* [put_b,  put_v0,  put_v1],
    iv_call_pct       = 100 .* [call_b, call_v0, call_v1],
    fv_put            = [fv_put_b,  fv_put_v0,  fv_put_v1],
    fv_call           = [fv_call_b, fv_call_v0, fv_call_v1],
    edge_put          = [fv_put_b - MID_PUT, fv_put_v0 - MID_PUT, fv_put_v1 - MID_PUT],
    edge_call         = [fv_call_b - MID_CALL, fv_call_v0 - MID_CALL, fv_call_v1 - MID_CALL],
)
out_csv = joinpath(out_dir, "lora_lly_prototype_summary.csv")
CSV.write(out_csv, summary_df)
@printf("\n[csv] wrote -> %s\n", out_csv)

# Drift-trigger persistence
trigger_df = DataFrame(
    date          = per_date_d,
    n_obs         = per_date_n,
    base_rmse_pct = 100 .* per_date_rmse,
    refit_flagged = flagged,
)
trig_csv = joinpath(out_dir, "lora_lly_drift_trigger.csv")
CSV.write(trig_csv, trigger_df)
@printf("[csv] wrote -> %s\n", trig_csv)

println("\nDone.")
