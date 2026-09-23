# Results of the small stock comparison

The small comparison found an improvement in LLY's forecast distribution, but
did not establish a general improvement in the central stock-price forecast.
Adaptive volatility improved LLY's five-session CRPS in both 2025 and the
previously examined August–September 2026 period. It did not improve GS's CRPS
in either period. Earlier-data selection preferred zero drift over every tested
directional regression setting, so the selected directional candidate became
identical to the adaptive-volatility model. This is a result of the selection
procedure, not an omitted directional experiment. The original JumpHMM and the
unchanged-price benchmark remain necessary comparisons.

## Design and selection

The candidate set and selection rules were recorded in PROTOCOL.md before fitting
or scoring the new candidates. The comparison used the saved 2014–2024 JumpHMM,
an unchanged-price point forecast, a Gaussian stock model with EWMA variance, and
the same variance model with a regularized directional regression. The regression
used the stock's most recent one-, five-, and twenty-return information and SPY's
most recent five-return information, normalized by origin volatility. Both drift
and variance were fixed within each forecast horizon and updated only when a
new origin was reached. No realized future stock or market path was supplied.

Settings were selected using expanding annual validation blocks from 2019 through
2024. Regression fits and feature scaling used earlier observations only, with
training responses required to end before each validation year. One shared EWMA
decay and one shared ridge penalty were selected by averaging ticker, year, and
one-/five-session validation scores equally. The decay candidates were 0.80,
0.90, 0.94, 0.97, and 0.99. The selected decay was 0.97, giving each newly observed
squared daily return a 3% weight in the variance update. The directional penalty
grid ranged from 0.01 to 1000 and included the zero-drift limit. Zero drift had
the lowest validation loss; none of the finite-penalty fits improved the chosen
validation objective. Final settings and fitted coefficients were saved before
test forecast scoring. Neither 2025 nor 2026 outcomes selected those settings.

The additional 2025 stock file supplied 249 one-session and 245 five-session
origins per ticker. The 2026 comparison used exactly the original 21 one-session
and 17 five-session stock outcomes per ticker. All four candidates were scored
against identical origins, endpoints, starting prices, and observed prices.
Historical price series were aligned and checked across GS, LLY, and SPY. For
2026, daily returns required both adjacent exchange sessions to be observed;
missing prices were not filled or converted into accumulated one-day returns.
Feature windows used the last N valid daily returns and recorded their age.
The oldest stock/market input at the examined 2026 origins was four calendar days
old because an adjacent close was missing. The 2025 features had no such gap.
These limitations were preserved rather than selecting only favorable origins.

## Five-session findings

The table reports averages over three fixed simulation replicates, each using
10,000 paths per origin. Median MAE evaluates the forecast median and mean RMSE
evaluates the mean. CRPS evaluates the entire predictive distribution; lower
values are better. Errors and interval widths are dollars per share. Coverage
refers to the central 90% interval. The unchanged-price forecast is a point
benchmark and has no useful uncertainty interval. Adaptive volatility and the
selected directional model are exactly identical because selection chose zero
drift; both rows remain in the machine-readable outputs.

| Period | Ticker | Method | Median MAE | Mean RMSE | CRPS | 90% coverage | Width |
|---|---|---|---:|---:|---:|---:|---:|
| 2025 | GS | JumpHMM | 22.31 | 28.42 | 16.22 | 84.4% | 84.72 |
| 2025 | GS | Adaptive / selected directional | 22.62 | 28.57 | 16.32 | 89.8% | 94.86 |
| 2025 | GS | Unchanged | 22.61 | 28.75 | 22.61 | — | — |
| 2025 | LLY | JumpHMM | 39.10 | 50.16 | 29.60 | 69.9% | 100.41 |
| 2025 | LLY | Adaptive / selected directional | 39.35 | 50.24 | 28.57 | 84.4% | 151.37 |
| 2025 | LLY | Unchanged | 39.37 | 50.36 | 39.37 | — | — |
| 2026 | GS | JumpHMM | 16.25 | 21.71 | 13.03 | 100.0% | 129.06 |
| 2026 | GS | Adaptive / selected directional | 15.48 | 21.07 | 15.58 | 100.0% | 176.18 |
| 2026 | GS | Unchanged | 15.51 | 20.71 | 15.51 | — | — |
| 2026 | LLY | JumpHMM | 49.44 | 58.62 | 35.60 | 70.6% | 147.27 |
| 2026 | LLY | Adaptive / selected directional | 49.31 | 58.35 | 33.81 | 94.1% | 191.38 |
| 2026 | LLY | Unchanged | 49.37 | 58.17 | 49.37 | — | — |

For LLY, the adaptive model reduced five-session CRPS by approximately 3.5% in
2025 and 5.0% in 2026. Coverage increased substantially, with wider intervals.
The 2025 coverage of 84.4% still fell short of the nominal 90%. The lower CRPS
indicates that the distributional improvement was not merely an unpenalized
reward for widening the intervals. Its date-level CRPS was lower on 131 of 245
origins in 2025 and 10 of 17 in 2026; it did not win on every date. Its central
price errors remained close to those of an unchanged-price forecast. The three
simulation replicates agreed on the direction of the distribution-score change.

For GS, adaptive volatility raised five-session CRPS by approximately 0.6% in
2025 and 19.6% in 2026. In 2025, coverage moved closer to 90%, but the gain in
coverage came with wider intervals and a slightly worse overall distribution
score. In 2026, both methods covered every endpoint, and the adaptive intervals
were substantially wider. Adaptive CRPS improved on only 103 of 245 dates in
2025 and 2 of 17 in 2026. Central-error improvements of the adaptive candidate
over JumpHMM in 2026 did not beat the unchanged-price benchmark consistently
across MAE and RMSE, and cannot be attributed to a learned directional signal
because the selected drift was zero.

## One-session findings and interpretation

The shorter horizon did not establish a directional benefit either. In 2025,
adaptive versus JumpHMM CRPS was 6.74 versus 6.66 for GS and 10.63 versus 10.67
for LLY. Corresponding 90% coverage was 93.4% versus 87.8% for GS and 93.3%
versus 77.5% for LLY. In 2026, CRPS was 8.72 versus 8.00 for GS and 14.47 versus
14.73 for LLY. The adaptive and unchanged-price median errors remained nearly
identical, as expected for a zero-log-drift Gaussian forecast whose population
median equals its starting price. Their tiny differences in the reported MAE
come from simulation precision and are not evidence of directional forecasting.
The full one-session errors, widths, and seed-specific results are in summary.csv
and summary_by_replicate.csv.

These results strengthen the case that LLY needs a changing volatility estimate,
but do not support replacing the stock model with this EWMA rule for every ticker.
They also do not identify a successful repair for the central stock-price misses.
The tested recent-return and market features failed to justify a directional
adjustment in earlier validation. This is evidence about the particular simple
model and features tested, not proof that no directional information exists.
The 2025 period supplies substantially more observations than the 2026 case study,
but overlapping forecasts remain dependent and 2026 had already informed the
development question. No significance claim or winning-seed selection is made.

The appropriate near-term use is to report adaptive volatility as an alternative
stock-distribution benchmark and describe its ticker-dependent behavior. A rule
that chooses the stock model differently across tickers would itself need to be
chosen and validated on earlier data; selecting it from these test outcomes
would overstate the evidence. The current experiment has been saved separately
and does not replace the original manuscript scores or submission package.

## Reproduction and checks

Run from the repository root:

```sh
julia --project=code --startup-file=no code/scripts/export_small_stock_data.jl
python3 code/test/test_small_stock_models.py
python3 code/scripts/fit_small_stock_models.py
python3 code/scripts/prepare_small_stock_forecasts.py
julia --project=code --startup-file=no code/scripts/run_small_stock_comparison.jl
python3 code/scripts/summarize_small_stock_comparison.py
```

Four Python tests checked that future prices do not alter origin features, gaps
do not become one-day returns, training response cutoffs are enforced, and
training scaling/zero-drift shrinkage behave correctly. Thirty-two Julia checks
verified positive finite forecasts and agreement with analytic lognormal means
at the first origins of both periods. Aggregation asserted exact outcome matching
across every candidate and replicate and exact equality of the selected
directional and adaptive forecasts. Input, source, and settings hashes are saved
in data_manifest.toml, frozen_settings.json, and run_manifest.toml. The raw and
previous forecast files were preserved.
