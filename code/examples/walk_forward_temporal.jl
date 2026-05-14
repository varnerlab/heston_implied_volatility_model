"""
Walk-forward temporal validation across the 15-date ladder corpus (Reviewer 2).

For k = 5..14, train Configuration-A (pooled, 2-input ψ) on the first k
sorted ladder dates and test on date k+1. Reports the per-fold generalization
gap and a CSV / line plot.

The k=6 fold (train days 1..6 = 04-14..04-22, test day 7 = 04-23) should
reproduce the Configuration-A 13.03% test RMSE and +4.99% gen-gap from
`tab:earnings_holdout` — this is a built-in regression check.

Run:
    julia --project=. examples/walk_forward_temporal.jl
"""

using CSV
using DataFrames
using Dates
using Plots
using Plots.PlotMeasures
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "src", "TemporalFolds.jl"))
using .TemporalFolds

const LADDER_DIR    = joinpath(@__DIR__, "..", "data", "ladder")
const EARNINGS_CSV  = joinpath(@__DIR__, "..", "data", "earnings", "earnings_calendar.csv")
const FIG_DIR       = joinpath(@__DIR__, "..", "figures")
const N_INPUTS      = 2     # Configuration-A: pooled control, no earnings features

# Sorted 15-date corpus
const ALL_DAYS = [
    "options-04-14-2026", "options-04-15-2026",
    "options-04-16-2026", "options-04-17-2026",
    "options-04-21-2026", "options-04-22-2026",
    "options-04-23-2026", "options-04-24-2026",
    "options-04-27-2026", "options-04-28-2026",
    "options-04-29-2026", "options-05-01-2026",
    "options-05-06-2026", "options-05-08-2026",
    "options-05-11-2026",
]
@assert length(ALL_DAYS) == 15

# Folds: k = 5..14 → train days 1:k, test day k+1
const K_RANGE = 5:14

println("Loading earnings calendar from $(EARNINGS_CSV) ...")
cal = load_earnings_calendar(EARNINGS_CSV)
println("  loaded $(length(cal)) tickers")

println("Loading full 15-date ladder corpus ...")
all_data = load_split(LADDER_DIR, ALL_DAYS)
attach_earnings_features!(all_data, cal)
println("  $(nrow(all_data)) total obs across $(length(unique(all_data.ticker))) tickers, " *
        "$(length(unique(all_data.obs_date))) dates")

universe_sectors = sort(unique(all_data.sector))
universe_tickers = sort(unique(all_data.ticker))

function _split_by_dirnames(df, dirs)
    target_dates = Set(TemporalFolds.dir_to_date(d) for d in dirs)
    return df[[d in target_dates for d in df.obs_date], :]
end

function _test_window_has_earnings(test_df)
    # Flag fold if any test row has |d2e_self| ≤ 3 or d2e_peer_min ≤ 3
    return any(abs.(test_df.d2e_self) .<= 3) || any(test_df.d2e_peer_min .<= 3)
end

rows = NamedTuple[]
for k in K_RANGE
    train_dirs = ALL_DAYS[1:k]
    test_dirs  = ALL_DAYS[k+1:k+1]
    train_df = _split_by_dirnames(all_data, train_dirs)
    test_df  = _split_by_dirnames(all_data, test_dirs)
    test_date = TemporalFolds.dir_to_date(test_dirs[1])

    fold = run_fold(train_df, test_df, N_INPUTS;
                    sectors_list=universe_sectors,
                    tickers_list=universe_tickers,
                    label="WF k=$(k) train=1..$(k), test=$(k+1) ($(test_date))",
                    verbose=true, seed=42)

    has_e = _test_window_has_earnings(test_df)

    push!(rows, (
        k = k,
        n_train_days = k,
        n_train_obs  = nrow(train_df),
        test_date    = test_date,
        n_test_obs   = nrow(test_df),
        train_rmse   = fold.train_rmse,
        test_rmse    = fold.test_rmse,
        gen_gap      = fold.gen_gap,
        has_earnings_in_test_window = has_e,
    ))

    @printf("\n  FOLD k=%2d  train_obs=%6d  test_obs=%5d  train=%5.2f%%  test=%5.2f%%  gap=%+5.2f%%  earnings=%s\n",
            k, nrow(train_df), nrow(test_df),
            fold.train_rmse*100, fold.test_rmse*100, fold.gen_gap*100,
            has_e ? "Y" : "N")
end

# ============================================================================
# Write CSV
# ============================================================================
mkpath(FIG_DIR)
df = DataFrame(rows)
csv_path = joinpath(FIG_DIR, "walk_forward_summary.csv")
CSV.write(csv_path, df)
@printf("\n[csv] wrote -> %s\n", csv_path)

# Summary
println("\n" * "="^80)
println("  WALK-FORWARD SUMMARY")
println("="^80)
println("   k    test_date    n_train    n_test    train(%)    test(%)    gap(%)    e?")
println("  " * "-"^75)
for r in rows
    @printf("  %2d   %s   %7d    %5d    %6.2f      %6.2f    %+6.2f    %s\n",
            r.k, r.test_date, r.n_train_obs, r.n_test_obs,
            r.train_rmse*100, r.test_rmse*100, r.gen_gap*100,
            r.has_earnings_in_test_window ? "Y" : "N")
end
@printf("\n  median test RMSE: %5.2f%%   median gap: %+5.2f%%\n",
        median([r.test_rmse for r in rows])*100,
        median([r.gen_gap for r in rows])*100)
@printf("  mean   test RMSE: %5.2f%%   mean   gap: %+5.2f%%\n",
        mean([r.test_rmse for r in rows])*100,
        mean([r.gen_gap for r in rows])*100)

# ============================================================================
# Figure: gap and test RMSE vs k
# ============================================================================
println("\nRendering walk-forward gap figure...")
ks   = [r.k for r in rows]
gaps = [r.gen_gap*100 for r in rows]
tests = [r.test_rmse*100 for r in rows]
earns = [r.has_earnings_in_test_window for r in rows]

p1 = plot(ks, tests; marker=:o, lw=2, ms=6,
          color=RGB(0.10, 0.35, 0.65),
          label="Test RMSE", legend=:topleft,
          xlabel="Fold (train days = 1..k, test = k+1)",
          ylabel="RMSE (% IV)",
          title="Walk-forward: per-fold test RMSE and generalization gap",
          framestyle=:box, grid=true, gridalpha=0.25,
          titlefontsize=12, guidefontsize=11, tickfontsize=10,
          left_margin=10mm, right_margin=4mm, top_margin=4mm, bottom_margin=8mm)
plot!(p1, ks, gaps; marker=:s, lw=2, ms=6,
      color=RGB(0.78, 0.20, 0.20), label="Gen gap")
# Annotate earnings-flagged folds
for (k, t, e) in zip(ks, tests, earns)
    if e
        annotate!(p1, k, t + 0.6, text("⚡", 10, RGB(0.8, 0.5, 0.0)))
    end
end
hline!(p1, [0.0], color=:gray, ls=:dash, alpha=0.6, label="")

p_full = plot(p1, size=(1100, 600), dpi=200)
out_pdf = joinpath(FIG_DIR, "walk_forward_gap.pdf")
out_png = joinpath(FIG_DIR, "walk_forward_gap.png")
savefig(p_full, out_pdf); savefig(p_full, out_png)
@printf("[fig] wrote -> %s / .png\n", out_pdf)

println("\nDone.")
