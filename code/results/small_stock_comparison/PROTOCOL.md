# Small stock comparison

Recorded before fitting the new candidates or calculating their test scores.
This follows the inspected August–September 2026 misses and is exploratory model
development. The 2025 stock file supplies an additional chronological evaluation;
neither 2025 nor 2026 outcomes select settings in this experiment.

## Four methods

1. The saved 2014–2024 JumpHMM with its original stationary initialization and
   independent-pilot shift to 10% annual growth. Retain the established settings.
2. Unchanged stock price, evaluated as a point forecast.
3. A zero-log-drift Gaussian stock model whose daily variance is updated by an
   exponentially weighted moving average (EWMA) of squared daily log returns.
4. The same variance model with one simple directional regression. Predict the
   next daily return, divided by its origin volatility, from the stock's last
   one, five and twenty available daily returns and SPY's last five available
   daily returns. Normalize cumulative features by volatility and square-root
   window length. Fit a ridge regression without an intercept; zero features
   therefore imply zero predicted log drift. Hold the origin drift and variance
   fixed over each one- or five-session forecast, so no future market returns
   enter and both horizons describe the same stock process.

This is one adaptive variance rule and one directional model, not a search over
many architectures. Directional coefficients are ticker-specific. Select one
shared EWMA decay and one shared ridge penalty by averaging ticker/year/horizon
validation scores, to avoid deploying different rules chosen on GS/LLY test wins.

## Earlier-data selection

Use annual expanding validation blocks 2019–2024, each with earlier observations
only for coefficient fitting and scaling. EWMA variance initializes from the
first 60 training daily returns and then updates causally. Candidate decays:
0.80, 0.90, 0.94, 0.97, 0.99. Choose decay by Gaussian predictive log loss in
log-return space, averaged equally across tickers, validation years and horizons
1 and 5. EWMA forecasts h times origin daily variance at horizon h. There is no
variance clipping except a numerical floor of 1e-12.

With the selected decay, choose ridge penalty from 0.01, 0.1, 1, 10, 100, 1000,
and the zero-drift limit. Minimize squared cumulative-log-return error divided by
h times origin variance, again equally weighted across ticker/year/horizon.
Training feature scale uses training rows only; the ridge objective is mean
squared normalized one-session error plus penalty times coefficient norm squared.
Training responses must end before the fold's validation year. Validation
endpoints must stay inside the validation year. Break ties toward stronger
shrinkage (or slower decay for the volatility rule). Refit the final regression
using 2014–2024 only and freeze its coefficients and scaling.

## Forecast observations and simulation

Historical GS, LLY and SPY close series come from the local 2014–2024 and 2025
JLD2 files. Align on the recorded SPY session grid and audit ticker dates. The
2025 test uses each recorded origin with an observed endpoint inside that year.
Variance and lag features update with observed returns through each origin,
while fitted parameters remain fixed. The 2026 test uses the existing audited
stock closes and identical original origins/endpoints, supplemented with SPY
closes from the same selected capture manifest. Use no option prices.

The 2026 series starts in April after a data gap. Restart its EWMA from the
2014–2024 variance; allow the April–July observations to update it before testing.
Do not join December 2025 to April 2026 into one daily return. Within 2026,
calculate a daily return only when both adjacent exchange sessions are observed.
Missing daily observations leave variance unchanged. Feature windows mean the
last N valid daily returns, which may span more than N sessions when data are
missing. Record feature age and counts. Do not fill missing stock closes.

Simulate 10,000 paths per origin with three fixed seeds (202609091–202609093).
Share Gaussian innovations between adaptive and directional variants. Save all
replicates, and average replicate scores without treating simulation replicates
as independent market dates. Freeze candidate settings in a separate file before
opening test targets for scoring. Match all methods to identical ticker, origin,
endpoint and observed price. Preserve the previous results.

## Evaluation and interpretation

Report mean RMSE, median MAE, CRPS, central 90% interval coverage and width, signed
bias, and errors relative to origin price. Point forecasts have no meaningful
uncertainty interval; do not reward the unchanged-price benchmark for zero width.
Report 2025 and 2026 separately, both tickers, both horizons and every candidate.
Check whether any gain holds across seeds and dates, and separate improved
directional prediction from improved distribution calibration. An apparent win
on the already inspected 2026 period is not independent confirmation. Do not
automatically replace the manuscript's original model or rebuild its submission
package around whichever candidate performs best.
