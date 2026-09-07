# Why the stock forecasts missed

The investigation found no daily-return scaling or price-reconstruction error.
It found limited directional information in the fitted state transitions and a
volatility mismatch in LLY. Initializing from recent returns did not resolve the
five-session misses. Adjusting future volatility using past returns improved
LLY's forecast distribution, but did not meaningfully improve its central price
forecast and made the GS distribution worse. These are stock-only results;
option quotes and the IV module played no role in them. The comparisons preserve
the fitted 2014–2024 models and the original test dates. They are exploratory
follow-ups after seeing the initial misses, not a fresh holdout or evidence of a
production-ready correction.

## What was checked

The stored observations are annualized excess log returns. Both marginals have
dt=1/252 and rf=0, and the simulator reconstructs prices by exponentiating daily
log returns once. The first-origin 1,000-path forecasts for both tickers reproduced
the saved means at both horizons to numerical precision. The 10% annual growth
prior adds roughly 0.2% expected log growth across five sessions, so that drift
assumption cannot explain an 8.5% realized move. State propagation, observation
likelihoods, and the first future transition were tested on small models. The
new filter tracks remaining jump duration as well as the return state. This is
necessary because the package's ordinary forward filter omits its separate jump
mechanism. Missing daily closes advance the state distribution without treating
an accumulated multi-day return as a one-day observation. Truncating all input
history at the first forecast origin reproduced that origin's filtered state
exactly, confirming that its inference did not use later prices.

## What explains the behavior

The 50 states represent bins of daily return values rather than persistent
market-volatility regimes. Conditioning on the observed return history changes
the next-state distribution, but those changes quickly disappear under the
fitted transition matrix. Across the examined origins, the average total
variation distance between the conditioned and unconditioned latent forecasts
fell from about 0.30 at one session to 0.044 at two, 0.009 at three, and below
0.004 at five for both tickers. Here total variation measures the difference
between two probability distributions on a scale from zero to one. Consequently,
correctly initializing the current state barely changes five-session central
forecasts. The unchanged-price benchmark performs at least as well on point
error as the original model in these dates. This does not prove that stock moves
are unpredictable; it shows that this fitted model and its available inputs did
not supply a useful directional signal in the examined period.

LLY's forecast dispersion also lagged the observed period. The pilot daily
log-return standard deviation was 1.70%, close to the historical sample's 1.69%.
The 21 available adjacent-session returns in August–September had a standard
deviation of 2.19%. Five-session observed returns ranged from -7.12% to +8.54%,
and five of the seventeen endpoints fell outside the original 90% intervals.
The August 4 LLY example is direct: the stock started at 1117.47 and reached
1212.92 on August 11, while the original forecast mean was 1118.87 and its upper
90% bound was 1190.78. Large subsequent declines also generated major misses;
the problem was not simply a forecast biased against an upward trend. New jump
episodes are rare under the stored epsilon=0.0001 setting. The original five-step
simulator supplies four opportunities to start a jump, a probability of about
0.04% in total. Ordinary Student-t emissions can still generate large moves,
so this probability must not be interpreted as the model's entire tail risk.
The rare explicit jump mechanism is not an event-conditioned earnings model.

GS behaved differently. Its pilot daily standard deviation was 1.80%, compared
with 1.40% in the 21 available adjacent-session test returns. Its original
five-session intervals contained every observed endpoint, while LLY's contained
only about 71%. Widening both models therefore addressed the wrong problem for
GS. The recent 20-return estimate still included larger earlier GS moves and
made its already broad intervals wider. This explains why the tested adjustment
cannot be recommended as a common default. The historical Student-t emission
scale convention also adds a small amount of variance relative to using within-
state sample standard deviations directly; the implied daily standard deviations
are about 1.80% versus 1.76% for GS and 1.74% versus 1.70% for LLY. This is not
an order-of-magnitude scaling problem and does not explain LLY's higher recent
variability.

## Controlled five-session comparisons

Each row below averages three fixed runs of 10,000 paths, using the same seventeen
observed endpoints per ticker. Prices, errors, and interval widths are dollars
per share. MAE evaluates the forecast median; RMSE evaluates the mean. CRPS scores
the full distribution, with smaller values better. Coverage refers to central
90% intervals. A point benchmark has no uncertainty interval, so its interval
coverage is not a competing uncertainty estimate. The full files also report
one-session results, the stationary first-transition control, and every seed.

| Ticker | Method | Median MAE | Mean RMSE | CRPS | Coverage | Width |
|---|---|---:|---:|---:|---:|---:|
| GS | Original initialization | 16.28 | 21.72 | 12.99 | 100.0% | 129.38 |
| GS | Filtered state | 16.35 | 22.61 | 13.20 | 100.0% | 127.75 |
| GS | Filtered state + recent volatility | 16.21 | 22.85 | 15.62 | 100.0% | 176.58 |
| GS | Fixed-volatility random walk | 15.66 | 21.05 | 13.62 | 100.0% | 137.99 |
| GS | Recent-volatility random walk | 15.72 | 21.31 | 16.76 | 100.0% | 190.51 |
| GS | Unchanged price | 15.51 | 20.71 | 15.51 | — | — |
| LLY | Original initialization | 49.58 | 58.65 | 35.69 | 70.6% | 147.21 |
| LLY | Filtered state | 49.24 | 57.94 | 35.28 | 70.6% | 150.74 |
| LLY | Filtered state + recent volatility | 49.14 | 57.99 | 34.23 | 94.1% | 202.45 |
| LLY | Fixed-volatility random walk | 49.34 | 58.33 | 34.68 | 76.5% | 151.40 |
| LLY | Recent-volatility random walk | 49.32 | 58.55 | 34.00 | 94.1% | 203.38 |
| LLY | Unchanged price | 49.37 | 58.17 | 49.37 | — | — |

For LLY, filtered initialization plus recent volatility reduced CRPS by about
4.1% and increased coverage from 12/17 to 16/17. It did so with wider intervals,
not by predicting the observed directions. The simpler recent-volatility random
walk had a slightly lower CRPS still. In GS, the same adjustment raised CRPS by
about 20%, and state initialization alone did not help. These patterns persisted
across the three seeds. A better-looking interval is not by itself sufficient:
the width and CRPS make the cost of widening explicit. Seventeen overlapping
endpoints remain a small, dependent market sample, regardless of path count.

## How to address the issue

The immediate correction is to distinguish predicting the central stock price
from estimating the range and likelihood of future prices. The current model
has not beaten an unchanged-price forecast on the central prediction in this
sample. It can still be evaluated as a distributional scenario generator, but
it should not be presented as having demonstrated directional forecasting skill.
State conditioning is now implemented and testable; this experiment rules out
assuming that initialization alone repairs the five-session problem. Recent
volatility is a more promising target for improving LLY's uncertainty estimates,
but the GS result rules out adopting this particular 20-return rule wholesale
or turning it on only for the ticker on which it happened to win.

The next model-development question is a causal volatility update that responds
to both increases and decreases without an excessive lag. Its choices should
be selected on earlier rolling periods and frozen before later evaluation.
Scheduled-event inputs could be examined separately if available at the forecast
origin; raising jump frequency after seeing a large move would not establish
event prediction. A different state model would be needed to test persistent
volatility regimes, since the present return-bin transitions lose state
information quickly. None of these changes should be assumed to improve the
conditional mean. The present arXiv draft should report the stock benchmark and
the limitations of its actual simulation setup. These exploratory stock results
have been saved separately and have not silently replaced the paper's original
scores or its already-built submission package.

## Reproduction

Run from the repository root:

```sh
julia --project=code --startup-file=no code/test/test_stock_forecast_diagnosis.jl
julia --project=code --startup-file=no code/scripts/diagnose_stock_forecasts.jl
python3 code/scripts/summarize_stock_diagnosis.py
```

The protocol fixes the candidate set and evaluation before candidate scoring.
`manifest.toml` records input and source hashes. `scores.csv` contains every
endpoint, candidate, and replicate; `summary_by_replicate.csv` exposes simulation
variation; `summary.csv` averages those replicates. `state_memory.csv` records
state propagation, `origin_diagnostics.csv` records causal volatility estimates,
and `training_variance_audit.csv` records historical scale checks. The focused
unit suite passed eight tests and the experiment passed eight reconstruction
and causality checks. The integrated Julia suite passed all 227 tests, and
`git diff --check` passed. No fitted portfolio or original forecast file was changed.
