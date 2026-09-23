# Stock forecast diagnosis, September 7, 2026

This is an exploratory follow-up motivated by the already observed forecast
misses. It is not a new untouched holdout or a replacement of the first results.
Keep the 2014–2024 fitted GS/LLY marginal parameters fixed. Use only the saved
stock session observations; option values do not enter these experiments.

Before calculating candidate scores, fix these comparisons:
1. Existing stationary-start library simulation, with the original pilot shift.
2. Stationary state followed by a genuine next-session transition, including the
   jump mechanism, to isolate the forecast boundary convention.
3. State filtering using observations available through each origin, followed by
   a next-session transition. Track remaining jump duration as a latent variable
   so the filter matches the simulator's multi-session jump mechanism.
4. The same filtered forecasts with future centered daily returns scaled by the
   standard deviation of the last 20 available one-session returns divided by
   the original pilot standard deviation. This changes future dispersion only;
   retain the same filtered state probabilities to isolate the scale change.
5. Zero-log-drift Gaussian random walk with the fixed pilot volatility.
6. Zero-log-drift Gaussian random walk with the same recent 20-return volatility.
7. Unchanged stock price as a point benchmark.

Use all original August 4–September 4 origins and exactly the same observed
one- and five-session endpoints. No selecting origins by performance or option
availability. Use 10,000 paths and three fixed seeds (202609081–202609083), with
common random numbers where possible. Keep original 1,000-path scores separately.
Check reconstruction of the old first-origin forecasts before interpreting new
results. Report mean-based RMSE, median-based MAE, CRPS, central 90% coverage and
width, signed error, and point forecasts relative to an unchanged-stock baseline.
Report every candidate and simulation replicate, without selecting a winning seed.

Filtering begins at the first recorded April stock session. At each subsequent
exchange session, advance the latent state distribution. Apply an observation
likelihood only when both that session and its immediate predecessor have a
recorded close. Do not treat multi-session returns as one-day observations or
fill missing prices. The likelihood subtracts the fixed pilot growth shift from
observed annualized growth, consistent with the simulated observation model.
Filtering never reads a future return. Recent volatility uses the last 20 valid
one-session returns available at the origin, without searching window lengths.

Poisson jump durations in the filter retain all but at most 1e-12 probability
and fold the remaining mass into the final duration. The simulator can sample
that finite representation. Document that the library forward_filter ignores
the explicit jump mechanism; do not use it as an exact filter for this test.
Numerically test latent propagation and observation updates on a small model.
Check whether filtering effects disappear with horizon, whether observed returns
are more variable than the fixed forecast, and whether a few dates dominate.
Treat any apparent improvement as exploratory evidence requiring later validation.
