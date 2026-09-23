# Fixed chronological validation protocol

Recorded 2026-09-07 before calculating held-out forecast errors. This is a
retrospective chronological evaluation, not a claim of preregistered prospective
validation. The raw data window ends on 2026-09-04 even if collection continues.

## Data and cutoff

Preserve the original 15-date and 43-date corpora. Copy missing SDK captures into
the extended root. Resolve the latest eligible close capture per ticker/session;
then retain one row per exact option symbol. Retain an audit of every source and
exclusion. Use the underlying session as the valuation date and recompute calendar
DTE from expiration. Retrieval must be after that session's 20:00 UTC and before
the next exchange session's 13:30 UTC. Dates run April–September under
daylight-saving time.
The collector explicitly records UTC. Quote-level exchange timestamps are absent,
so this rule does not establish quote freshness or exact synchronization.

Training uses only captures available by 2026-07-31 23:59:59 UTC, with session dates
through July 31. No August–September outcomes select models, windows, parameters,
seeds, contract rules, or the illustrative case. Intraday repeated captures are
excluded. Quotes require finite positive bid, ask >= bid, and positive displayed
sizes; IV fitting also requires 0.01 < IV < 2, 0.8 <= K/S <= 1.2, and positive DTE.
Original row-level exclusions and the historical 53-fold results remain separately
identified; that earlier experiment is not silently replaced by the new filters.

## Fitted model and initialization

Use the existing two-input sector neural model (log calendar DTE, log moneyness),
ticker variance levels, seed 42, and existing fixed training schedule (up to 2,000
epochs; training-loss checkpoint). Fit six sector models once through July 31.
Save the training-only standardizer with model weights and input hashes. There is
no new architecture/window search and no claim of robustness to neural seed.

For dynamic comparisons, all variants start at the selected contract's observed
origin IV squared. Multiply its frozen surface target by origin variance divided
by the surface value at the origin. This origin anchoring is a distinct evaluation
variant and is described explicitly; the original paper's ablation is unchanged.
Compare frozen IV, anchored direct surface, deterministic relaxation, uncoupled
factor, and coupled factor. Retain kappa=15, sigma_v=0.5, rho=-0.6, floor=0.005^2,
r=0.0425, q=0, and the 201-step American LR/CRR pricer. Forecast initialization
at successive origins uses available observations but does not update parameters.

## Forecast origins, contracts, and simulation

Origins are observed GS/LLY sessions August 4–September 4. Primary horizon is five
exchange sessions; one session is secondary. An endpoint must fall within the fixed
window. Select one put and call per ticker/origin with 25–45 calendar DTE, choosing
the expiry nearest 31 DTE, then strike/spot nearest 0.95 (put) or 1.05 (call).
Break ties by earlier expiry then symbol. Select using only origin quotes and IV;
never use future match availability. Report selected, eligible, matched, and
unmatched counts. Match the same symbol at the exact endpoint session; never
substitute another contract or the next captured date. Do not require endpoint
moneyness or IV for a valid price observation. Endpoint IV is a separate diagnostic.

Use the fixed 2014–2024 JumpHMM marginal, 1,000 evaluation paths per ticker/origin,
and seeds deterministically derived from ticker and origin date. A separate
10,000-path pilot sets the shift to the existing 10% annual growth assumption and
one-session shock normalization. No completed evaluation ensemble is recentered.
Share paths and normal innovations among IV variants. Record a simple stock
random-walk benchmark using pilot return volatility and zero log drift. Include
an observed-stock-path diagnostic only when every intervening session is available;
this diagnostic is explicitly conditional on the realized path, not a stock forecast.

## Outcomes and display

Primary point score: mean absolute error of predicted mean option value against
observed bid–ask midpoint. Also report signed bias, RMSE, empirical CRPS, 90%
interval coverage and width, and whether the point prediction is inside the quote.
Average contracts within ticker/origin before pooling dates; report per ticker
and horizon. Compare paired model scores on the same eligible observations.
Report stock scores separately. Reprice endpoint reported IV in the same lattice
as a pricing diagnostic. Do not interpret model marks as executable fills.

The illustrative figure uses GS at the earliest origin with both origin contracts
and both five-session endpoint quotes available, regardless of forecast error.
Show stock, put, and call observations with the coupled forecast mean and 90%
bands, plus frozen-IV mean for option panels. Tables include all eligible origins.
Confidence statements are descriptive: overlapping dates and shared markets do
not supply independent replicates. Simulation count is not the market sample size.
Run meaningful pipeline/invariant tests and lattice/Monte Carlo checks on fixed
small subsets. Any implementation correction is logged; results are not selected
by whether coupling wins.

## Availability amendment before inspecting aggregate forecast errors

The original 25–45 DTE cohort has zero exact-symbol five-session quote matches
for either ticker (0/36 eligible contracts each). This follows from the collector's
rolling maturity buckets, not forecast accuracy. Retain its one-session results
and explicitly report the failed five-session endpoint. Do not silently replace
its contracts. Add a separately labeled 10–21 DTE cohort, choosing expiry nearest
14 DTE with the same origin-only strike and tie rules. Its availability audit found
34/36 GS and 30/36 LLY five-session matches. Use that cohort for the five-session
case study, with all other fitting, simulation, scoring, and illustration rules
unchanged. This amendment is retrospective and based on availability; it is not
presented as a fully prespecified untouched experiment. No aggregate forecast
errors were inspected to choose the additional cohort. Continuing data collection
must track selected symbols through time to validate the original monthly horizon.
