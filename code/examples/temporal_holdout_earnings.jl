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

Refactored 2026-05-14 to use TemporalFolds for the reusable train/predict
machinery. The three-config sequence here is preserved verbatim so the
existing 13.03 / 7.96 / 11.98 numbers reproduce.
"""

using CSV
using DataFrames
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "src", "TemporalFolds.jl"))
using .TemporalFolds

const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladder")
const EARNINGS_CSV = joinpath(@__DIR__, "..", "data", "earnings", "earnings_calendar.csv")
const TRAIN_DAYS = ["options-04-14-2026", "options-04-15-2026",
                    "options-04-16-2026", "options-04-17-2026",
                    "options-04-21-2026", "options-04-22-2026"]
const TEST_DAYS  = ["options-04-23-2026", "options-04-24-2026"]

println("Loading earnings calendar from $EARNINGS_CSV ...")
cal = load_earnings_calendar(EARNINGS_CSV)
println("  loaded $(length(cal)) tickers with earnings entries")

println("\nLoading train split: $TRAIN_DAYS")
train = load_split(LADDER_DIR, TRAIN_DAYS)
println("  $(nrow(train)) observations across $(length(unique(train.ticker))) tickers")

println("\nLoading test split:  $TEST_DAYS")
test = load_split(LADDER_DIR, TEST_DAYS)
println("  $(nrow(test)) observations across $(length(unique(test.ticker))) tickers")

println("\nAttaching earnings features...")
attach_earnings_features!(train, cal)
attach_earnings_features!(test,  cal)

@printf("  train d2e_self        range: [%d, %d]   peer_min range: [%d, %d]\n",
        minimum(train.d2e_self), maximum(train.d2e_self),
        minimum(train.d2e_peer_min), maximum(train.d2e_peer_min))
@printf("  test  d2e_self        range: [%d, %d]   peer_min range: [%d, %d]\n",
        minimum(test.d2e_self), maximum(test.d2e_self),
        minimum(test.d2e_peer_min), maximum(test.d2e_peer_min))

n_near_train = sum(near_earnings_mask(train))
n_near_test  = sum(near_earnings_mask(test))
@printf("\n  Near-earnings rows: train %d / %d (%.1f%%); test %d / %d (%.1f%%)\n",
        n_near_train, nrow(train), 100*n_near_train/nrow(train),
        n_near_test,  nrow(test),  100*n_near_test/nrow(test))

# Universe lists shared across configs
sectors = sort(unique(train.sector))
tickers = sort(unique(train.ticker))

# ============================================================================
# Run the three configurations
# ============================================================================

result_A = run_fold(train, test, 2; sectors_list=sectors, tickers_list=tickers,
                    label="A: POOLED CONTROL (no exclusion, 2 inputs)")

keep_train_B = .!near_earnings_mask(train)
keep_test_B  = .!near_earnings_mask(test)
result_B = run_fold(train[keep_train_B, :], test[keep_test_B, :], 2;
                    sectors_list=sectors, tickers_list=tickers,
                    label="B: NON-EARNINGS BASELINE (exclude ±3 d, 2 inputs)")

result_C = run_fold(train, test, 4; sectors_list=sectors, tickers_list=tickers,
                    label="C: EARNINGS-AWARE (4 inputs)")

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
        result_A.gen_gap*100)
@printf("  B: non-earnings, 2 inputs           %6d   %5d    %5.2f     %5.2f     %+5.2f\n",
        sum(keep_train_B), sum(keep_test_B),
        result_B.train_rmse*100, result_B.test_rmse*100,
        result_B.gen_gap*100)
@printf("  C: earnings-aware, 4 inputs         %6d   %5d    %5.2f     %5.2f     %+5.2f\n",
        nrow(train), nrow(test),
        result_C.train_rmse*100, result_C.test_rmse*100,
        result_C.gen_gap*100)
println()

# Tech-only spotlight (the blow-up sector)
println("\n  Tech-sector test RMSE comparison (the headline pivot):")
println("  Ticker  d2e_self  d2e_peer   A: pooled   C: earnings   delta")
println("  " * "-"^60)
for t in sort([k for (k, v) in TemporalFolds.SECTORS if v == "Tech"])
    mask = test.ticker .== t
    n = sum(mask)
    n == 0 && continue
    test_iv_local = Float64.(test.implied_vol[mask])
    rA = sqrt(mean((result_A.test_pred[mask] .- test_iv_local).^2)) * 100
    rC = sqrt(mean((result_C.test_pred[mask] .- test_iv_local).^2)) * 100
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
