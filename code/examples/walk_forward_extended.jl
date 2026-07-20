"""
Expanding-window walk-forward across the 58-date extended ladder corpus.

For k = 5..57, train on the first k capture dates and test on the unseen date
k+1, giving 53 folds. By default only the 2-input sector model runs per fold
(matching `walk_forward_temporal.jl`); the 4-input earnings-aware model is
deferred because the current corpus window falls in the trough between
earnings seasons and cannot support a replication claim for it. Set
WF_CONFIGS=both to also run the 4-input config once the corpus covers the
Q2 earnings season.

Reads two roots — `data/ladder` for the 15-date arm and `data/ladder_extended`
for the 43 new dates. `walk_forward_temporal.jl` is untouched and still
reproduces the 15-date result independently.

Environment:
    WF_K_MAX    caps the number of folds (default 57, the full sweep). The
                corpus load happens before the fold loop, so this does NOT
                reduce corpus loading time -- a capped run still loads the
                full 58-date, ~986k-observation corpus, which must be
                present regardless of the cap.
    WF_CONFIGS  "sector" (default) runs only the 2-input sector model;
                "both" also runs the 4-input earnings-aware model

Run:
    julia --project=. examples/walk_forward_extended.jl
    WF_K_MAX=16 julia --project=. examples/walk_forward_extended.jl
    WF_CONFIGS=both julia --project=. examples/walk_forward_extended.jl
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

const LADDER_ROOTS = [
    joinpath(@__DIR__, "..", "data", "ladder"),
    joinpath(@__DIR__, "..", "data", "ladder_extended"),
]
const EARNINGS_CSV = joinpath(@__DIR__, "..", "data", "earnings",
                              "earnings_calendar.csv")
const FIG_DIR = joinpath(@__DIR__, "..", "figures")

# The 2026-05-12..2026-07-17 corpus window sits in the trough between
# quarterly earnings seasons: only 11 earnings events total, 6 of them on the
# final three capture days (monthly counts: Apr 17, May 9, Jun 2, Jul 18). The
# Q2 season lands 2026-07-22..2026-08-05, just past the end of the corpus, so
# the 4-input earnings-aware config cannot support a replication claim here.
# It is deferred until the corpus covers the Q2 season; set WF_CONFIGS=both
# to re-enable it.
const ALL_CONFIGS = [
    (name = "sector_2in",   n_inputs = 2),
    (name = "earnings_4in", n_inputs = 4),
]
const CONFIGS = get(ENV, "WF_CONFIGS", "sector") == "both" ? ALL_CONFIGS : ALL_CONFIGS[1:1]

const K_MAX = parse(Int, get(ENV, "WF_K_MAX", "57"))
const K_RANGE = 5:K_MAX
@assert 5 <= K_MAX <= length(EXTENDED_DAYS) - 1 "WF_K_MAX must be in 5:$(length(EXTENDED_DAYS) - 1); K_MAX < 5 produces an empty K_RANGE"

println("Loading earnings calendar from $(EARNINGS_CSV) ...")
cal = load_earnings_calendar(EARNINGS_CSV)
println("  loaded $(length(cal)) tickers")

println("Loading extended corpus across $(length(LADDER_ROOTS)) roots ...")
all_data = load_split(LADDER_ROOTS, EXTENDED_DAYS)
attach_earnings_features!(all_data, cal)
@printf("  %d obs, %d tickers, %d dates\n", nrow(all_data),
        length(unique(all_data.ticker)), length(unique(all_data.obs_date)))

universe_sectors = sort(unique(all_data.sector))
universe_tickers = sort(unique(all_data.ticker))

function _split_by_dirnames(df, dirs)
    target_dates = Set(TemporalFolds.dir_to_date(d) for d in dirs)
    return df[[d in target_dates for d in df.obs_date], :]
end

function _test_window_has_earnings(test_df)
    return any(abs.(test_df.d2e_self) .<= 3) || any(test_df.d2e_peer_min .<= 3)
end

rows = NamedTuple[]
for k in K_RANGE
    train_df = _split_by_dirnames(all_data, EXTENDED_DAYS[1:k])
    test_df  = _split_by_dirnames(all_data, EXTENDED_DAYS[k+1:k+1])
    test_date = TemporalFolds.dir_to_date(EXTENDED_DAYS[k+1])

    # Guard against a malformed or truncated capture day, not an expected
    # occurrence: across the completed 53-fold sweep this never fired -- the
    # minimum observed n_test_obs was 11,379.
    if nrow(test_df) < 500
        @printf("  SKIP k=%2d (%s): only %d test obs\n",
                k, test_date, nrow(test_df))
        continue
    end

    has_e = _test_window_has_earnings(test_df)

    for cfg in CONFIGS
        fold = run_fold(train_df, test_df, cfg.n_inputs;
                        sectors_list = universe_sectors,
                        tickers_list = universe_tickers,
                        label = "WF-ext k=$(k) test=$(test_date) [$(cfg.name)]",
                        verbose = false, seed = 42)

        push!(rows, (
            k = k,
            config = cfg.name,
            n_inputs = cfg.n_inputs,
            n_train_days = k,
            n_train_obs = nrow(train_df),
            test_date = test_date,
            n_test_obs = nrow(test_df),
            train_rmse = fold.train_rmse,
            test_rmse = fold.test_rmse,
            gen_gap = fold.gen_gap,
            has_earnings_in_test_window = has_e,
        ))

        @printf("  k=%2d %-13s test=%s train=%5.2f%% test=%5.2f%% gap=%+5.2f%% e=%s\n",
                k, cfg.name, test_date, fold.train_rmse*100,
                fold.test_rmse*100, fold.gen_gap*100, has_e ? "Y" : "N")
    end
end

# ============================================================================
# Write CSV
# ============================================================================
mkpath(FIG_DIR)
df = DataFrame(rows)
csv_path = joinpath(FIG_DIR, "walk_forward_extended_summary.csv")
CSV.write(csv_path, df)
@printf("\n[csv] wrote -> %s\n", csv_path)

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^80)
println("  EXTENDED WALK-FORWARD SUMMARY  ($(length(K_RANGE)) folds)")
println("="^80)
for cfg in CONFIGS
    sub = df[df.config .== cfg.name, :]
    nrow(sub) == 0 && continue
    @printf("  %-13s  median test %5.2f%%   median gap %+5.2f%%   mean gap %+5.2f%%\n",
            cfg.name, median(sub.test_rmse)*100,
            median(sub.gen_gap)*100, mean(sub.gen_gap)*100)
end

# ============================================================================
# Figure
# ============================================================================
plt = plot(size=(900, 460), left_margin=6mm, bottom_margin=6mm,
           xlabel="test date", ylabel="test RMSE (%)",
           title="Expanding-window walk-forward, 58-date corpus",
           legend=:topright)
for cfg in CONFIGS
    sub = df[df.config .== cfg.name, :]
    nrow(sub) == 0 && continue
    plot!(plt, sub.test_date, sub.test_rmse .* 100,
          label=cfg.name, marker=:circle, markersize=3, linewidth=1.5)
end
for ext in ("pdf", "png")
    p = joinpath(FIG_DIR, "walk_forward_extended_gap.$(ext)")
    savefig(plt, p)
    @printf("[fig] wrote -> %s\n", p)
end

println("\nDone.")
