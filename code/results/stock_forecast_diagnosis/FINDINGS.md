# Stock diagnosis after emission truncation

The September 14 rerun uses Student-t residuals conditional on absolute value
at most 10 fitted scale units in both simulation and the state-inference
likelihood. The model's state locations, scales, transitions, and jump mechanism
are retained. Chronological pilot constants were regenerated first. The forward
sampler uses one uniform quantile draw per emission, preserving common draws
across initialization variants and cutoff sensitivity cases.

Eight reconstruction and causality checks pass. Recent-state conditioning
still supplies little five-session central-forecast improvement. The trained
state-transition memory and historically selected stock benchmark settings
are unaffected. `FINDINGS-before-truncation.md` preserves the earlier analysis;
its numerical tables describe the untruncated run. Current five-session scores
follow, with errors and widths in dollars per share and coverage as a fraction.

| ticker | method | n_dates | median_mae | rmse | crps | coverage | width |
|---|---|---|---|---|---|---|---|
| GS | filtered_state | 17.000 | 16.358 | 22.589 | 13.214 | 1.000 | 128.318 |
| GS | filtered_state_vol20 | 17.000 | 16.188 | 22.834 | 15.636 | 1.000 | 177.339 |
| GS | legacy_stationary | 17.000 | 16.278 | 21.716 | 12.993 | 1.000 | 129.374 |
| GS | rw_fixed | 17.000 | 15.658 | 21.054 | 13.617 | 1.000 | 137.994 |
| GS | rw_vol20 | 17.000 | 15.719 | 21.307 | 16.758 | 1.000 | 190.513 |
| GS | stationary_transition | 17.000 | 16.213 | 21.639 | 13.012 | 1.000 | 129.401 |
| GS | unchanged | 17.000 | 15.508 | 20.710 | 15.508 | 0.000 | 0.000 |
| LLY | filtered_state | 17.000 | 49.402 | 58.023 | 35.354 | 0.706 | 150.981 |
| LLY | filtered_state_vol20 | 17.000 | 49.360 | 58.091 | 34.334 | 0.941 | 202.762 |
| LLY | legacy_stationary | 17.000 | 49.584 | 58.648 | 35.687 | 0.706 | 147.203 |
| LLY | rw_fixed | 17.000 | 49.339 | 58.325 | 34.680 | 0.765 | 151.402 |
| LLY | rw_vol20 | 17.000 | 49.315 | 58.550 | 33.996 | 0.941 | 203.382 |
| LLY | stationary_transition | 17.000 | 49.644 | 58.653 | 35.674 | 0.706 | 147.096 |
| LLY | unchanged | 17.000 | 49.366 | 58.172 | 49.366 | 0.000 | 0.000 |

Reproduction uses the commands in the repository README and the unchanged
experiment protocol. The emission correction and wider-bound comparison are
recorded in `../truncated_emissions/PROTOCOL.md` and `FINDINGS.md`.
