# code/figures/

This directory is **regenerable build output**, not the canonical source.

## Canonical figures live in `paper/sections/figures/`

The paper builds against `paper/sections/figures/`. Every PDF that lands there is byte-identical to the corresponding output in this directory, copied by `code/scripts/promote_figures.jl`. If the two ever disagree, the paper is the source of truth and `code/figures/` is stale build output.

## Conventions

- **Every figure-producing script must call `promote_figures()` at the end.** This scans `paper/sections/*.tex` for `\includegraphics{sections/figures/<file>}` and copies each referenced PDF here into `paper/sections/figures/`.
- **Only PDFs are canonical.** PNGs occasionally produced by scripts are previews, not paper inputs, and are not preserved here.
- **Unreferenced PDFs are stale.** If a script writes a PDF that no `.tex` section references, `promote_figures.jl` will warn after each sync. Either cite it in the paper or delete it.

## Regenerating from scratch

```bash
cd code
julia --project=. examples/calibrate_ladders_sector_nn.jl
julia --project=. examples/calibrate_ladders_per_ticker_nn.jl
julia --project=. examples/temporal_holdout_earnings.jl
# ... etc, see examples/ for the full set
julia --project=. scripts/promote_figures.jl
```

## Reproducible artifacts (logs and CSVs) kept here

- `earnings_holdout_full.log` + `earnings_holdout_summary.csv` — three-config sector NN temporal holdout (`temporal_holdout_earnings.jl`)
- `earnings_holdout_per_ticker_full.log` + `earnings_holdout_per_ticker_summary.csv` + `earnings_holdout_per_ticker_b2_summary.csv` — per-ticker NN temporal holdout (`temporal_holdout_per_ticker_earnings.jl`)
