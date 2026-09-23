# Truncated Student-t emissions: completed results

The primary and wider-cutoff reruns completed on September 14, 2026. All six
experiment groups were regenerated at each cutoff: fitted GS/LLY illustrations,
paired dynamic-IV ablation, monthly and shorter-maturity chronological forecasts,
stock-initialization diagnosis, and the frozen 2025/2026 stock comparison.
The original fitted neural surfaces, trained state parameters, dates, contracts,
seeds, and path counts were retained. Pilot and ensemble normalization constants
were recomputed under each emission law.

## Specification and finite moments

The primary residual law is Student-t conditional on |Z| <= 10 fitted scale
units, with |Z| <= 20 as sensitivity. Both cutoffs were recorded before these
reruns as stress assumptions, not selected by forecast scores. Each state's
location and scale come from the 2014–2024 fit. Rejected emission proposals are
redrawn independently before price reconstruction; prices and P&L are not clipped.
The same state and jump paths are retained for seeded single-asset simulations.
State filtering uses the corresponding conditional emission likelihood.

For t(5), the primary bound excludes 0.017095% of the original residual mass;
the wider bound excludes 0.000578%. With the chronological pilot shift, the
primary daily percentage-return support over all states is -26.47% to +38.99%
for GS and -21.23% to +38.98% for LLY. The wider supports are -42.73% to +81.83%
and -38.22% to +82.11%, respectively. These are model support limits, not
confidence intervals or empirically validated maximum returns. The exact values
and conditional residual variances are in `bounds.csv`.

Finite support over finitely many states makes all fixed-horizon stock-price
moments finite. Consequently, the divergent stock mean and short-call expected
loss problem of the unbounded log-Student-t generator is removed. This property
does not demonstrate that either support limit matches future market extremes.

## Sensitivity results

Widening the bound from 10 to 20 scale units changed terminal short-call mean
P&L by -$0.0185/share for GS and -$0.0101/share for LLY. Corresponding changes
in 5% expected shortfall were -$0.0624 and -$0.0520. At the ten-transition
coupled-IV ablation endpoint, call mean P&L changed by less than $0.004/share
and expected shortfall by less than $0.02/share. See `call_tail_sensitivity.csv`.

Five-session coupled option MAE was $4.193743 for GS and $11.587739 for LLY
at the primary cutoff, versus $4.200236 and $11.587747 at the wider cutoff.
The largest absolute change was below $0.007/share. The largest change in the
reported five-session stock CRPS across 2025/2026 and GS/LLY was below
$0.002/share. See `forecast_sensitivity.csv`. These are comparisons of the
specified finite simulation ensembles, not a proof of general tail insensitivity.

The main conclusions remain: the paired IV variants change interim marks but
agree at expiration; coupling offers only marginal option forecast improvements;
central stock forecasts remain close to an unchanged-price benchmark; and
adaptive volatility improves LLY's distribution scores but not GS's. The stock
benchmark selection still chooses zero directional drift.

## Verification and reproduction

The full Julia suite passes 251 tests, including conditional-distribution,
no-boundary-mass, state/jump preservation, reproducibility, exponential-moment,
copula support, unsupported-observation, and cache-cutoff checks. Six Python
tests and 136 saved-forecast reconstruction checks pass. Each stock comparison
passes 32 analytic-price/finite-value checks, and each stock diagnosis passes
eight reconstruction/causality checks. Source/input hashes and emission metadata
are recorded in the per-experiment manifests. Primary results occupy their
usual directories; the wider runs are under `wide/`.

The manuscript includes generated support, call-loss, and forecast-sensitivity
tables. `audit_truncated_emissions.jl` generates the bounds; the Python helper
`summarize_truncated_emissions.py` verifies completed run manifests and generates
the comparison tables. The repository README provides the full rerun sequence.
Scenario cache version 3 rejects earlier untruncated or different-cutoff caches.

The bounded library API supports single-asset JumpHMM and copula portfolios.
Factor portfolios now fail explicitly because their separate market/residual
construction needs its own finite-moment implementation. The manuscript uses
single-ticker simulations. Older exported scenario clock and variance-engine
limitations remain separately documented in the submission review.
