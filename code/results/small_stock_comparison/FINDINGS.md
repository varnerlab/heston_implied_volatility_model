# Stock comparison after emission truncation

The September 14 rerun uses Student-t residuals conditional on absolute value
at most 10 fitted scale units. It retains the 2014–2024 state parameters,
historical selection, origins, seeds, path counts, and Gaussian benchmarks.
Chronological pilot constants were recomputed under the same cutoff. The
underlying mathematical price moments are now finite. The cutoff is a stated
stress assumption, not a fitted tail boundary.

The substantive conclusion is unchanged: central forecasts remain close to
an unchanged-price benchmark. Adaptive volatility improves LLY's five-session
CRPS in 2025 and 2026 but does not improve GS's. Earlier validation still selects
zero directional drift. `FINDINGS-before-truncation.md` preserves the historical
analysis; its numerical tables describe the earlier untruncated results.
The following table is regenerated from the current `summary.csv`; errors and
widths are dollars per share, and coverage is a fraction.

| period | ticker | method | n_dates | median_mae | rmse | crps | coverage | width |
|---|---|---|---|---|---|---|---|---|
| 2025 | GS | Adaptive volatility | 245.000 | 22.618 | 28.567 | 16.324 | 0.898 | 94.864 |
| 2025 | GS | Directional | 245.000 | 22.618 | 28.567 | 16.324 | 0.898 | 94.864 |
| 2025 | GS | JumpHMM | 245.000 | 22.309 | 28.424 | 16.221 | 0.844 | 84.714 |
| 2025 | GS | Unchanged | 245.000 | 22.614 | 28.755 | 22.614 | 0.000 | 0.000 |
| 2025 | LLY | Adaptive volatility | 245.000 | 39.354 | 50.235 | 28.572 | 0.844 | 151.371 |
| 2025 | LLY | Directional | 245.000 | 39.354 | 50.235 | 28.572 | 0.844 | 151.371 |
| 2025 | LLY | JumpHMM | 245.000 | 39.098 | 50.158 | 29.604 | 0.699 | 100.397 |
| 2025 | LLY | Unchanged | 245.000 | 39.369 | 50.362 | 39.369 | 0.000 | 0.000 |
| 2026 | GS | Adaptive volatility | 17.000 | 15.483 | 21.073 | 15.578 | 1.000 | 176.181 |
| 2026 | GS | Directional | 17.000 | 15.483 | 21.073 | 15.578 | 1.000 | 176.181 |
| 2026 | GS | JumpHMM | 17.000 | 16.255 | 21.707 | 13.026 | 1.000 | 129.049 |
| 2026 | GS | Unchanged | 17.000 | 15.508 | 20.710 | 15.508 | 0.000 | 0.000 |
| 2026 | LLY | Adaptive volatility | 17.000 | 49.310 | 58.349 | 33.805 | 0.941 | 191.384 |
| 2026 | LLY | Directional | 17.000 | 49.310 | 58.349 | 33.805 | 0.941 | 191.384 |
| 2026 | LLY | JumpHMM | 17.000 | 49.440 | 58.628 | 35.602 | 0.706 | 147.259 |
| 2026 | LLY | Unchanged | 17.000 | 49.366 | 58.172 | 49.366 | 0.000 | 0.000 |

Reproduction uses the commands in the repository README and the unchanged
experiment protocol. The emission correction and wider-bound comparison are
recorded in `../truncated_emissions/PROTOCOL.md` and `FINDINGS.md`.
