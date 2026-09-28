"""
Sensitivity of the April earnings holdout to the stored-DTE offset.

The snapshot labeled April 23 was downloaded before the open on April 24, and
its `actual_dte` counts from the download date, one day short of the April 23
session. That snapshot is one of the two test dates of the earnings holdout
(`examples/temporal_holdout_earnings.jl`). Training rows (April 14-22) are
unaffected, so each configuration is trained once, exactly as in the driver,
and the same test rows are scored twice: with the stored DTE and with the
April 23 rows counted from the session date (stored DTE + 1).

Run:
    julia --project=. scripts/check_holdout_dte_offset.jl
"""

using CSV
using DataFrames
using Dates
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "src", "TemporalFolds.jl"))
using .TemporalFolds

const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladder")
const EARNINGS_CSV = joinpath(@__DIR__, "..", "data", "earnings", "earnings_calendar.csv")
const TRAIN_DAYS = ["options-04-14-2026", "options-04-15-2026",
                    "options-04-16-2026", "options-04-17-2026",
                    "options-04-21-2026", "options-04-22-2026"]
const TEST_DAYS  = ["options-04-23-2026", "options-04-24-2026"]
const OFFSET_DATE = Date(2026, 4, 23)
const OUT_CSV = joinpath(@__DIR__, "..", "results", "holdout_dte_offset.csv")

cal = load_earnings_calendar(EARNINGS_CSV)
train = load_split(LADDER_DIR, TRAIN_DAYS)
test  = load_split(LADDER_DIR, TEST_DAYS)
attach_earnings_features!(train, cal)
attach_earnings_features!(test,  cal)

# Confirm the offset before correcting it: every April 23 row should store a
# DTE one day shorter than expiration minus the session date.
off = test.obs_date .== OFFSET_DATE
session_dte = Dates.value.(Date.(test.expiration[off]) .- Date.(test.und_session_date[off]))
@assert all(session_dte .- test.actual_dte[off] .== 1) "April 23 offset is not uniformly one day"
@printf("April 23 test rows: %d of %d; stored DTE = session DTE - 1 for all of them\n",
        sum(off), nrow(test))

test_fixed = copy(test)
test_fixed.actual_dte = copy(test.actual_dte)
test_fixed.actual_dte[off] .+= 1

sectors = sort(unique(train.sector))
tickers = sort(unique(train.ticker))

keepB_tr = .!near_earnings_mask(train)
keepB_te = .!near_earnings_mask(test)
configs = [("A", train, test, test_fixed, 2),
           ("B", train[keepB_tr, :], test[keepB_te, :], test_fixed[keepB_te, :], 2),
           ("C", train, test, test_fixed, 4)]

rows = NamedTuple[]
for (name, tr, te, te_fixed, n_inputs) in configs
    r = run_fold(tr, te, n_inputs; sectors_list=sectors, tickers_list=tickers,
                 label=name, verbose=false)
    pred_fixed = predict_sector_nn(te_fixed, r.sector_models, sectors, r.standardizer, n_inputs)
    y = Float64.(r.test_df.implied_vol)
    d = r.test_df.obs_date .== OFFSET_DATE
    push!(rows, (config=name, n_train=nrow(tr), n_test=nrow(r.test_df), n_test_0423=sum(d),
                 train_rmse=100r.train_rmse,
                 test_rmse_stored=100rmse(r.test_pred, y), test_rmse_session=100rmse(pred_fixed, y),
                 test0423_stored=100rmse(r.test_pred[d], y[d]), test0423_session=100rmse(pred_fixed[d], y[d]),
                 test0424=100rmse(r.test_pred[.!d], y[.!d])))
end

df = DataFrame(rows)
mkpath(dirname(OUT_CSV))
CSV.write(OUT_CSV, df)
println()
println("Config  N_test  Train   Test(stored)  Test(session)  Change | 04-23 stored  04-23 session | 04-24")
for r in eachrow(df)
    @printf("  %s     %6d  %5.2f   %6.2f        %6.2f        %+5.2f  |   %6.2f        %6.2f     | %6.2f\n",
            r.config, r.n_test, r.train_rmse, r.test_rmse_stored, r.test_rmse_session,
            r.test_rmse_session - r.test_rmse_stored, r.test0423_stored, r.test0423_session, r.test0424)
end
println("\nWrote $OUT_CSV")
