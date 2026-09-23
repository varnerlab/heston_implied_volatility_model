# Earnings-aware refit of the July-cutoff surface

Written 2026-09-23, before any earnings-aware fit was run.

## Question

The Discussion attributes LLY's held-out overprediction (August 6 to September 4)
to its August 5 earnings report: a single ticker IV level fitted through July 31
cannot represent IV both before and after a report. If that is right, giving the
surface the time to each ticker's scheduled report should reduce LLY's
post-report overprediction.

## Configurations

Both use `training.csv` (1,173,538 rows, underlying sessions through
2026-07-31) and `surface_test.csv` (425,061 rows, 2026-08-04 to 2026-09-04)
unchanged, the six sector networks, the `TemporalFolds` architecture rule,
2,000-epoch limit, patience 200, learning-rate schedule, and training-only
standardization.

- **A2 (current):** two inputs, log calendar DTE and log moneyness.
- **E4 (earnings-aware):** A2's inputs plus the two existing earnings inputs from
  `TemporalFolds.attach_earnings_features!`: signed calendar days to the
  ticker's nearest report, clipped to [-30, 30], and the minimum absolute days to
  a same-sector peer's nearest report, clipped to 30 (ETFs use the peer value
  for both).

Seeds 42, 43, and 44 for each configuration. A2 seed 42 is the published fit
(`../surface_model.jld2`); it is re-predicted, not refitted, and must reproduce
`../surface_scores_by_date.csv`.

The observation date for the earnings inputs is the underlying session, the
date convention of the July-cutoff experiment. (The April holdout used the
capture-directory date.) The calendar is `code/data/earnings/earnings_calendar.csv`
as saved: LLY reports 2026-04-30 and 2026-08-05; GS 2026-04-13, 2026-07-14,
and 2026-10-13. Scheduled report dates are public in advance, so the inputs use
no information unavailable at a forecast date.

## Outcomes

Bias is observation-weighted mean predicted minus observed IV, in volatility
points.

- **Primary:** LLY held-out bias, sessions 2026-08-06 to 2026-09-04.
- **Secondary:** LLY bias on 2026-08-04; GS held-out bias 2026-08-06 to
  2026-09-04; LLY and GS held-out RMSE; pooled held-out and training RMSE.

## Decision rule (fixed now)

- **Supported:** the E4 seed-mean absolute LLY primary bias is at most half the
  A2 seed-mean, and every E4 seed's absolute LLY primary bias is below every A2
  seed's.
- **Not supported:** the E4 seed-mean absolute LLY primary bias is not below
  the A2 seed-mean.
- **Partial:** anything in between.

GS and pooled errors are reported whatever the outcome. All results go in the
supplement, and the Discussion is revised to match the outcome, including
removing the earnings explanation if the result is not supported.
