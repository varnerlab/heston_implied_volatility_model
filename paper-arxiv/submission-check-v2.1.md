# September 14 update: truncated return emissions

The current manuscript is 40 pages. All six affected simulation experiment groups
were regenerated with Student-t residuals truncated at ±10 fitted scale units and
with ±20 for sensitivity. The fitted state parameters and neural surfaces were
retained; pilot normalization constants were recomputed. Three new supplementary
tables report return bounds and sensitivity. The Methods and quote-comparison
captions identify Alpaca's free indicative feed.

Verification passed: 251 Julia tests, six Python tests, 136 saved-forecast
reconstruction checks, and 102 recorded source/input hashes across the twelve
primary/sensitivity runs. All 44 figure/table labels are unique and cited in the
Results. The strict latexmk build has no undefined references or layout warnings;
the existing font-substitution warning remains. All 40 pages were inspected in
rendered overviews, with the new appendix and changed figures checked separately.

The refreshed source archive contains 55 files. Their hashes matched the package
manifest, and an isolated strict build produced identical 40-page PDF text. The
plain abstract is synchronized with the manuscript. Review artifacts are under
`tmp/pdfs/truncated-emissions/`; numerical findings and verification metadata are
in `code/results/truncated_emissions/`. No commit, push, or arXiv submission was
performed. The historical checks below describe the earlier manuscript.

---

# Manuscript check before the author's prose pass

Title update, September 7, 2026: the manuscript is now titled “Simulating American
Option Prices with Dynamic Implied Volatility.” The shorter title changed page
flow, producing a 36-page PDF with Table 4 on page 16. The PDF and source archive
were rebuilt, and the archive compiled independently with identical extracted
PDF text. Pages 1, 2, 16, 33, and 36 were visually checked after the title change.
There were no layout warnings or undefined references. The existing font
substitution warning remains. Verification artifacts for this update are under
`tmp/pdfs/title-update/`; the source manifest records the refreshed hashes.
The detailed integration record below refers to the preceding 37-page version.

Prepared September 7, 2026. This records the completed integration and submission
checks for the 37-page `main.pdf`. No additional model experiments were run during
this integration. JumpHMM and the original IV experiments were retained.

## Integrated evidence and interpretation

- Main Table 4 (page 17) reports the five-session stock comparison for 2025 and
  August–September 2026. Each ticker has 245 and 17 forecast origins, respectively.
  The larger 2025 evaluation concerns stocks, not option prices.
- Supplementary Table S23 retains the prespecified option mean-forecast MAE and
  adds median-forecast MAE and mean-forecast RMSE on the same matched contracts.
- Supplementary Tables S24–S27 document causal stock-state initialization,
  historical parameter selection, and complete five- and one-session stock scores.
  The selected EWMA decay was 0.97. The directional model selected zero drift.
- The abstract, introduction, Results, Discussion, and Conclusion distinguish
  controlled IV comparisons from evidence of forecasting accuracy. The stock
  comparison does not establish a general improvement in central forecasts.
  Adaptive volatility improved LLY's five-session distribution score in both
  periods but did not improve GS's.
- The April–May scenarios are identified as illustrations. The chronological
  forecasts use information available at their origins. The observed-stock-path
  diagnostic explicitly uses future stock information and is not a usable
  forward-forecast improvement. Retrospective choice of the benchmark classes
  after inspecting the 2026 misses is disclosed.

## Numerical, editorial, and visual checks

- Six new tables were generated from completed outputs. Frozen selection-input
  hashes, simulation-source hashes, and the six generated-table hashes were
  verified. Their manifest is
  `code/results/small_stock_comparison/manuscript_table_manifest.json`.
- Option point-summary scores were recomputed from the saved contract-level
  forecasts, averaging contracts within each origin and then averaging origins.
  Captions distinguish scoring of means from scoring of medians and identify
  forecast sample sizes and simulation replication where applicable.
- All 41 main and supplementary figure/table labels are unique and are cited in
  the Results prose. Every figure caption begins with a descriptive lead.
- The narrative follows `HOUSE_STYLE.md`: general-to-specific Introduction with
  a developed final paragraph beginning “In this study,” substantial narrative
  paragraphs, and a consolidated limitations paragraph ending the Discussion.
- Rendered pages 1, 2, 7, 8, 14–19, and 32–37 were inspected at manuscript size.
  New tables, affected captions, and narrative pages have no clipping or overlap.
  The final caption clarification changed only page 33, which was re-rendered
  and inspected again.
- `make -C paper-arxiv all` passed. There are no undefined references, overfull
  or underfull boxes, or LaTeX errors. The existing author-email font substitution
  warning (`OMS/cmtt/m/n`) remains and did not produce a visible layout problem.
- `git diff --check` passed. Previously completed model tests were not rerun for
  this manuscript integration.

## Submission files

- `main.pdf`: reviewed manuscript, 37 pages.
- `abstract-v2.1.txt`: 242-word plain-text abstract generated from the current
  manuscript abstract, with no LaTeX commands.
- `arxiv-source-v2.1.tar.gz`: 51 reachable source/dependency files. It includes
  the required figures, bibliography, compiled bibliography, and local style.
  It excludes README files, raw data, build logs, and submission notes.
- `code/results/chronological_validation/arxiv_source_manifest.json` records
  hashes for the archive, reviewed PDF, plain abstract, and all packaged files.
- The final archive was extracted into a fresh isolated directory and compiled
  with pdflatex, BibTeX, and two further pdflatex passes. All extracted source
  hashes matched the manifest. The resulting 37-page PDF's layout-preserving
  text extraction matched the reviewed PDF exactly.

The isolated verification files are under
`tmp/pdfs/arxiv-stock-integration-final-check/`. The repository README contains
the updated artifact locations and regeneration commands. No commit, push, or
submission was performed as part of this integration.

## After the prose pass

Rebuild the PDF and refresh both submission derivatives:

```sh
make -C paper-arxiv all
python3 code/scripts/package_arxiv_source.py
```

Inspect changed pages and check the build log again. The package helper rejects
sources newer than the PDF. This audit describes the version prepared for the
prose pass; its visual and package checks must be repeated for later revisions.
