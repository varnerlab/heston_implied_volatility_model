# Corpus extension: 15 → 58 capture dates via expanding-window walk-forward

Date: 2026-07-20

## Problem

The `alpaca-markets-sdk` sibling repository has continued its EOD options pull
(`scripts/pull_options_eod.sh`, 16:30 ET weekdays) since the Heston corpus was
frozen. Its `data/` directory now holds 58 usable capture days spanning
2026-04-14 through 2026-07-17, in the same 31-ticker DTE-ladder format the
project already consumes.

The project corpus is a frozen 15-date subset ending 2026-05-11 (234,549
observations). That leaves 43 uncaptured dates — a corpus expansion of roughly
3.9x, extending the observation window from four weeks to fourteen.

The goal is to use the additional dates for both training and out-of-sample
evaluation through an expanding-window walk-forward, testing whether the
sector model's edge holds or decays over a materially longer horizon.

## Data inventory

43 new dates, 2026-05-12 through 2026-07-17, 31 tickers on every one. No
partial captures.

Coverage is 58 of 65 expected trading days. Genuine gaps are 04-30, 05-04,
05-05, 05-07, 06-05, 06-08 and 06-09 — seven weekdays, scattered rather than
clustered, so no single week is lost. 05-25 and 07-03 are Memorial Day and the
Independence Day observance and are correctly absent.

The window reaches the leading edge of Q2 earnings season: BAC, GS and JPM
report 07-14 and JNJ 07-15. It is not a full season, and describing it as one
would overstate the result.

## Approach

Two data roots, keeping the 15-date arm independently runnable.

The scripts behind the published pooled fits — `calibrate_ladders.jl:76`,
`calibrate_ladders_sector_nn.jl:69`, `calibrate_ladders_per_ticker_nn.jl:76` —
do not use a date list. They `walkdir` the ladder root and consume every
`_dte_ladder_` CSV they find. Adding 43 directories to that root would silently
redefine their corpus: the 234,549-observation fits behind 12.48 / 11.47 /
10.24 / 9.73 would become ~900k-observation fits on the next run, and the
15-date arm would survive only in log files.

That matters independently of publication status. The claim this work produces
is comparative — sector NN scores 10.24% on 15 dates, X on 58 — and that
comparison requires both arms to remain runnable on demand.

All 20 scripts resolve the literal path `data/ladder` with no wildcard and no
globbing of `data/`, so a sibling directory is invisible to every one of them.
The project already uses this pattern: `ladder_excluded/` sits beside `ladder/`
to keep the 04-20 partial capture out of the walkdir root.

Rejected alternatives:

- **One root with explicit date pinning.** The better end state, since it also
  removes the 15-date list and `SECTORS` dict duplicated across five files. It
  requires touching every calibration script, and each edit is an opportunity to
  perturb a number that currently reproduces. Worth doing as a separate cleanup;
  not worth bundling into a data extension. The two-root layout does not
  preclude it.
- **A standalone script carrying its own 58-date list.** Cheapest today, but it
  adds a seventh copy of the date list and leaves the disk-inference hazard armed
  for whoever extends the corpus next.

## Data layout

Three siblings under `code/data/`:

    ladder/            15 dirs — the 15-date arm, left alone
    ladder_excluded/   04-20 partial capture, unchanged
    ladder_extended/   43 new dirs, options-05-12-2026 … options-07-17-2026

The split is reversible. Pointing the pooled fits at both roots, or moving the
directories together, is a later decision that costs nothing now.

## Components

### `code/scripts/sync_ladder_extended.jl`

Copies capture directories from the SDK sibling into `ladder_extended/`,
renaming the two-digit year the SDK emits (`options-05-12-26`) to the
four-digit form this project uses (`options-05-12-2026`). No script in either
repository currently performs this transfer; the original was a manual copy.

Only directories matching `options-MM-DD-YY` are considered. The SDK's `data/`
also holds `options/` and `options-partial/`, which do not match and are
ignored.

Idempotent. Refuses to write into `ladder/`. Skips dates on or before 05-11 so
the 15-date arm cannot be re-seeded by accident — this also excludes the SDK's
04-20 capture, which is partial at 23 tickers and already held out in
`ladder_excluded/`. The source path is a parameter defaulting to the SDK
location, since students will not have it at the same path.

Julia, matching `promote_figures.jl`; the repository has no shell scripts.

### `TemporalFolds.jl` — `EXTENDED_DAYS`

The 58-date list, defined once. The walk-forward script imports it rather than
carrying a copy, adding no new duplication to the five files that already hold
15-date lists.

### `TemporalFolds.jl` — multi-root `load_split`

`load_split(ladder_dir::AbstractString, day_dirs)` at `src/TemporalFolds.jl:96`
gains a sibling method taking a vector of roots. Julia dispatch binds the four
existing callers to the original method, so their behavior is unchanged.

A `resolve_day` helper maps each day directory to its root. It raises on a day
found in no root, and on a day found in more than one. Both are silent
corpus-corruption modes, so neither is a warning-and-continue.

### `code/examples/walk_forward_extended.jl`

Modeled on `walk_forward_temporal.jl`, which is left exactly as it stands so its
published figure and its median gap of +2.02% remain reproducible.

Roots `[ladder/, ladder_extended/]`, days `EXTENDED_DAYS`, `K_RANGE = 5:57` for
53 folds. Each fold trains on days `1..k` and tests on the unseen day `k+1`.

Two configurations run per fold: the sector model with `n_inputs=2`, matching
what the current walk-forward reports, and the earnings-aware `n_inputs=4`
configuration. The new window carries an independent set of earnings events, so
the second configuration tests whether the earnings feature's contribution
replicates on events it has never seen. At present that contribution rests on a
single cluster, 04-23 and 04-24, dominated by INTC's +2192% surprise.

The earnings features are proximity-based — `TemporalFolds.jl:117` and `:131`
call `days_to_earnings` for the ticker and its sector peers — so missing
`eps_actual` values do not break them. Missing *scheduled dates* would silently
make earnings-adjacent days look quiet, which is why the calendar refresh below
is a prerequisite rather than an improvement.

### Earnings calendar refresh

`code/data/earnings/earnings_calendar.csv` was fetched 2026-04-26 and is a
stale snapshot. Comparable nine-week windows hold 24–27 events; the 05-12 to
07-17 window holds 10, every one with an empty `eps_actual`. The fetcher's own
docstring explains why: yfinance returns roughly four upcoming dates per ticker
beyond its history window.

The fetcher is re-run in place. The April snapshot remains recoverable from git
history, which is sufficient. Expect the window to reach roughly 25 events with
actuals populated.

## Outputs

`walk_forward_extended_gap.pdf`, `.png`, and a per-fold
`walk_forward_extended_results.csv`, all in `code/figures/`.

`promote_figures()` is not called. Nothing is written to `paper-arxiv/**` or
`paper-jcf/**`. Whether any of this reaches a paper is a decision to make after
the drift curve exists.

## Execution order

Each step gates the next.

1. **Sync.** Verify 43 directories in `ladder_extended/`, 31 CSVs in each, and
   `ladder/` still holding exactly 15 — a guard against the script writing to
   the wrong root.
2. **Refresh earnings.** Verify the 05-12 to 07-17 window grew from 10 events to
   roughly 25 with actuals. Diff historical rows against the git HEAD copy and
   surface any revision rather than absorbing it.
3. **Control-arm check.** Re-run `calibrate_ladders_sector_nn.jl` and confirm it
   still reports 234,549 observations and 10.24%. This establishes the two-root
   insulation empirically rather than by inspection.
4. **Smoke test.** `K_RANGE = 5:16` only. This covers the critical seam: fold
   k=15 trains on days 1–15, all in `ladder/`, and tests on day 16, 05-12, in
   `ladder_extended/` — the one fold exercising the cross-root resolver on a real
   boundary. It also measures wall-clock.
5. **Full sweep.** 53 folds × 2 configurations, once step 4 passes and the
   runtime is known.

## Scale

53 folds × 2 configurations × 6 sector models is 636 network fits, against 60
today, with later folds carrying roughly four times the data. The result is
somewhere near 30–40x current walk-forward cost. The current wall-clock is
unmeasured, which is why the smoke test precedes the full sweep rather than
following it.

Committing the data adds roughly 275 MB, bringing the repository near 370 MB.
This is well under GitHub's 1 GB soft warning, and individual CSVs are about
250 KB, so no file-size limit applies. Committing preserves clone-and-run
reproducibility for students, who otherwise would need the SDK repository and
their own Alpaca credentials.

## Verification

- Directory and ticker counts after sync, and `ladder/` unchanged at 15 dirs.
- Earnings window event count before and after refresh; historical-row diff.
- Control arm reproduces 234,549 observations and 10.24%.
- Seam fold k=15 resolves across both roots.
- `walk_forward_extended.jl` audited for `:S[end]` and single-date filtering
  before any number it produces is trusted. Three figure and scenario scripts
  previously failed this way when the 15-date corpus landed; the existing
  scripts are insulated by the two-root split, but the new script is written
  against a 58-date corpus and is exactly the kind of code that bites.

## Out of scope

Per-ticker and LoRA configurations across the folds, the SABR and Horvath
baselines, consolidating the duplicated date lists and `SECTORS` dictionary,
and any paper edit.
