# Simulating American Option Prices with Dynamic Implied Volatility

This repository connects JumpHMM physical stock paths, fitted IV surfaces,
contract-specific square-root factors, and American-option lattices. The
contract factors are scenario IV inputs, not a jointly calibrated Heston
stock/variance model.

The revised manuscript is in [`paper-arxiv/`](paper-arxiv/). The separate
`paper-jcf/` draft has not been synchronized with this arXiv revision.

## Reproduce the arXiv scenarios and dynamic ablation

Run from the repository root with the supplied `code/Project.toml` and local
`code/Manifest.toml`. The experiments were run with Julia 1.12.7.

```sh
julia --project=code -e 'using Pkg; Pkg.instantiate()'
julia --project=code --startup-file=no -e 'include("code/test/runtests.jl")'
julia --project=code --startup-file=no code/examples/reproduce_arxiv_scenarios.jl
julia --project=code --startup-file=no code/examples/dynamic_iv_ablation.jl
python3 code/scripts/promote_arxiv_ablation.py
julia --project=code --startup-file=no code/scripts/export_manuscript_figure_data.jl
python3 code/scripts/render_manuscript_figures.py
make -C paper-arxiv all
```

The first experiment regenerates the corrected GS/LLY illustrations and
their table bodies. The second runs five IV variants on shared stock paths,
with three seeds and 1,000 paths per seed for each ticker. A separate pilot
fixes return normalization before evaluation. Its primary endpoint is
liquidation after ten trading transitions. See the
[fixed analysis design](code/experiments/dynamic-ablation-design.md).

For a quick pipeline check, add `--smoke` to the ablation command. To rebuild
its summaries and figures from saved path marks, use `--render-only` instead.
Smoke outputs have a separate directory and are never promoted to the paper.

The ablation writes `code/results/dynamic_ablation/`, including path marks,
per-seed and pooled summaries, adjacent-variant contrasts, lattice-depth
checks, strike audits, pilot constants, input hashes, a per-file corpus
manifest, and a snapshot of the computational sources. The promotion script
copies generated tables and figures into the self-contained arXiv source
tree; the paper does not read files outside `paper-arxiv/` when compiled.

The final export and rendering commands rebuild the revised manuscript figures
from saved results without refitting or drawing new paths. They require Python
with pandas. Run these commands after promotion, which
otherwise copies the original ablation plot. The Julia exporter checks the
neural-cache and frozen-corpus hashes and writes its inputs to
`code/results/figure_revision/`. The renderer checks plotted statistics against
the saved experiment summaries. It summarizes stock and IV paths and reports
the May 11 smile errors at observed contracts.
The smile points show one date; their fitted curves pool all fifteen snapshots.
The Python renderer calls `code/scripts/render_scenario_figures.jl` for the
stock, option-value, and IV panels. These use the Plots/GR style of the
HMM-w-jumps-paper figures: boxed panels, sans-serif type, inset legends, and
the navy/red palette. Gridlines are disabled. To rebuild only these panels,
run `julia --project=code --startup-file=no code/scripts/render_scenario_figures.jl`.
The same renderer invokes `render_ablation_figure.jl` for paired coupling effects
conditioned on stock return (ten equal-count groups, with medians and interquartile
bands; all paths retained)
and `render_smile_figure.jl` for the six calibration panels. Figure text is limited
to panel letters, legends, and axes; dates and maturities are given in the caption.
The terminal P&L distributions are reported in the scenario tables rather than
a separate manuscript figure. The optional `render_terminal_pnl_figure.jl`
diagnostic writes to `code/results/figure_revision/` and checks its statistics
against the frozen scenario summaries. These scripts can also be run directly
with `julia --project=code --startup-file=no`.

## Required local inputs

- `code/data/ladder/`: the frozen 457-file, 234,549-filtered-row calibration
  corpus, collected from Alpaca Markets using a free-tier account. The runner
  checks the row count and records file hashes. Do not
  substitute the larger `ladder_extended/` corpus without retraining.
- `code/figures/calibrate_ladders_per_ticker_nn_cache.jld2`: the fitted neural
  surfaces. The calibration scripts under `code/examples/calibrate_ladders*`
  generate the static fitting hierarchy.
- `code/data/pretrained-portfolio-surrogate.jld2`: pretrained JumpHMM marginals.

These are research inputs; installation alone does not download them. The
JumpHMM training window was 2014–2024. Saved portfolio metadata records 424
tickers, 50 states, 2,767 training days, `dt=1/252`, and `rf=0`.
Alpaca's [Market Data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq)
documents Black–Scholes calculations with a Vega-based iterative IV solver.
The experiments fit the reported IV field directly.

## Interpretation

The ablation measures sensitivity of interim marks and model liquidation
P&L. It does not establish predictive superiority or trading profitability.
All monetary option values and P&L are per share. Residual GS strike-condition
violations are reported explicitly; passing the finite LLY grid does not
establish an arbitrage-free market. See [SETUP.md](SETUP.md) for the older
environment and earnings-calendar setup instructions.

## Chronological evaluation for the arXiv revision

The completed evaluation is recorded in
`code/results/chronological_validation/PROTOCOL.md`, including the availability
amendment. It uses a July 31 training cutoff and a fixed August 4–September 4,
2026 evaluation window. The original calibration and ablation outputs remain
separate. Source hashes, capture exclusions, selected contracts, exact endpoint
matches, fitted weights and preprocessing, path distributions, and date-level
scores are saved with the new experiment.

From the repository root:

```sh
julia --project=code --startup-file=no code/scripts/sync_ladder_extended.jl
python3 code/scripts/prepare_chronological_validation.py
python3 code/scripts/select_short_maturity_validation.py
julia --project=code --startup-file=no code/scripts/fit_chronological_surface.jl
julia --project=code --startup-file=no code/scripts/run_chronological_forecasts.jl
FORECAST_COHORT=short julia --project=code --startup-file=no code/scripts/run_chronological_forecasts.jl
python3 code/scripts/summarize_chronological_validation.py
julia --project=code --startup-file=no code/scripts/check_forecast_outputs.jl
julia --project=code --startup-file=no code/scripts/render_forecast_validation.jl
make -C paper-arxiv all
```

Training checkpoints validate their input and trainer hashes before reuse. The
large derived `training.csv`, `surface_test.csv`, and `surface_predictions.csv`
tables in `code/results/chronological_validation/` are excluded from Git. The
preparation script regenerates the first two from the committed source snapshots;
the surface-fitting script regenerates predictions using the saved checkpoints.
The source snapshots, fitted models, forecast outputs, and summary scores are retained.
The runner accepts `--smoke` for a separate 32-path check. Python requires pandas and
NumPy; Julia uses the existing project. The full test suite includes forecast
calendar, scoring, and anchoring checks. Quote-filter checks run with
`python3 code/test/test_prepare_chronological_validation.py`.

The roughly monthly cohort has no exact-contract five-session matches because
the collector's maturity buckets rotate. Its one-session results are retained.
A separately identified 10–21 DTE cohort supplies five-session outcomes; it was
added after an availability audit, not selected by forecast accuracy. Continuing
collection should preserve selected option symbols across their forecast horizons.
Simply accumulating maturity ladders does not produce complete contract histories.

The forecast results do not establish a consistent benefit from the fixed coupled
factor over frozen IV. The manuscript reports that outcome, the conditional
observed-stock diagnostic, and the limited, overlapping set of market dates.
The archived runner versions document the addition of the cohort selector during
the original run; the numerical procedure was unchanged.

After rebuilding the manuscript, `python3 code/scripts/package_arxiv_source.py`
creates `paper-arxiv/arxiv-source-v2.1.tar.gz` from the reachable TeX sources,
figures, bibliography, and local style file. It excludes raw data and unrelated
submission notes. The completed package was compiled in an isolated directory;
its PDF text matched the reviewed manuscript exactly. Packaging also refreshes
`paper-arxiv/abstract-v2.1.txt` directly from the manuscript abstract and refuses
to package sources edited after the PDF was built.

## Follow-up stock forecast diagnosis

The author's questions prompted a separate stock-only investigation, recorded in
`code/results/stock_forecast_diagnosis/FINDINGS.md`. It compares the original
initialization, causal state filtering, recent volatility, random walks, and an
unchanged-price benchmark on identical dates. Run
`julia --project=code --startup-file=no code/scripts/diagnose_stock_forecasts.jl`
and then `python3 code/scripts/summarize_stock_diagnosis.py` to reproduce it.
The original portfolio and forecast outputs are preserved. The initialization
check is now incorporated into the manuscript, with the broader exploratory
outputs retained separately.

The subsequent four-method comparison is documented in
`code/results/small_stock_comparison/FINDINGS.md`, with reproduction commands.
It selects EWMA and directional-regression settings using 2014–2024 data, freezes
them, and evaluates both 2025 and the original 2026 stock outcomes. Historical
selection preferred zero directional drift. Adaptive volatility improved LLY's
five-session distribution score in both test periods, but did not improve GS's;
it did not establish a general improvement in central price forecasts. All
candidate, date, horizon, and simulation-replicate scores are retained.

## Manuscript prepared for the author's prose pass

The current 36-page `paper-arxiv/main.pdf` integrates the 2025 and 2026 stock
comparisons in Table 4 and the supplement. The abstract, introduction, results,
discussion, conclusion, and forecast captions distinguish illustrative scenarios,
forecasts evaluated against later observations, and diagnostics supplied with
future stock information. JumpHMM remains the original worked example. The
results support controlled comparisons of IV assumptions; they do not establish
a consistent stock or option forecasting advantage.

After editing the manuscript prose, regenerate the submission files with:

```sh
python3 code/scripts/promote_stock_validation_tables.py
make -C paper-arxiv all
python3 code/scripts/package_arxiv_source.py
```

The resulting files are `paper-arxiv/main.pdf`, `paper-arxiv/abstract-v2.1.txt`,
and `paper-arxiv/arxiv-source-v2.1.tar.gz`. The source archive includes only the
reachable manuscript sources, required figures, bibliography, and local style;
it does not include old README files, raw data, or submission notes. Check the
build log and rendered PDF again after a prose pass. The audit for this prepared
version is recorded in `paper-arxiv/submission-check-v2.1.md`.
