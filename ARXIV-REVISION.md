# ArXiv revision: changes and remaining evidence

## Current submission revision: September 28, 2026

The current manuscript has 45 pages and its source archive has 63 files. The
final prose pass preserves the requested paragraph structure and frames the
experiments as fixed-horizon valuation comparisons. Supporting checks include
six-ticker pricing errors, the network-size diagnostic, the stored-DTE offset,
and paired-bootstrap uncertainty for expected-shortfall differences. The PDF,
plain abstract, and archive are synchronized; both the working source and the
extracted archive build without warnings and produce identical PDF text. See
`paper-arxiv/submission-check-v2.1.md` for the final checks and `README.md` for
reproduction commands. Earlier page counts below are historical.

## Completed work

- September 14: explicitly truncated Student-t price emissions at ±10 fitted
  scale units, complete reruns at ±10 and ±20, and three sensitivity tables.
  All 251 Julia tests, six Python tests, and 136 forecast reconstruction checks
  pass. The 40-page manuscript and 55-file source archive are synchronized.
  Alpaca indicative-feed provenance is documented and no author query remains.

- Five-arm paired dynamic ablation on GS and LLY: frozen IV, direct surface,
  deterministic mean reversion, uncoupled stochastic factors, and the full
  return-coupled factors. Three seeds, 3,000 paths per ticker, four interim
  endpoints, exact entry/expiry agreement, and paired Monte Carlo summaries.
- Independent pilot normalization for the ablation, held fixed across all
  evaluation paths and seeds. The older fitted illustrations retain their
  explicitly documented ensemble normalization for continuity.
- Necessary American strike-condition audits at two lattice depths; positive
  findings and violations are both included in the manuscript.
- Fixed an additional stock-path indexing defect: JumpHMM's reconstruction
  includes the initial spot, so indexing `prices[1:n]` duplicated the first
  spot and omitted the final return. The corrected scenario uses the full
  `n+1` vector, with a regression check and cache-version increment.
- Replaced the low-volatility European payoff safeguard with an American
  CRR safeguard for the paper's zero-dividend scenarios. Regenerated the
  fitted scenario figures and table bodies using the corrected path clock.
- Corrected per-share monetary labels and the shared-shock algorithm.
- Reframed the novelty around the modular construction and added relevant
  direct IV-surface-dynamics literature. Kept absolute fitting errors visible.
- Recorded the corpus's distinct snapshot labels, capture timestamps and
  underlying-session dates, and generated a table of their mapping.
- Added reproduction instructions, saved result tables, input hashes, and
  exact computational source snapshots for the ablation.

## Interpretation of the new evidence

The ablation establishes how IV assumptions affect interim marks and
hypothetical short-position P&L at fixed horizons on shared stock paths. It
cannot select the best model of future observed IV or establish an exit rule.
Paired Monte Carlo standard errors describe mean P&L differences; paired
bootstrap standard errors now also describe the coupling change in expected
shortfall. These quantify simulation precision conditional on the fitted models
and pilot constants, not uncertainty across market dates.

The GS direct surface and some factor variants violate necessary strike
conditions on the diagnostic grid. This is an empirical limitation, not
fixed by relabeling or by increasing the lattice depth. An arbitrage-constrained
surface or projection remains a separate modeling extension. The audit does
not cover calendar consistency or dynamic no-arbitrage.

## Confirmed data provenance

The author has repeatedly confirmed daily option-chain downloads through a free
Alpaca account. This provenance is settled and requires no further author query.
Alpaca identifies the unsubscribed snapshot feed as indicative and describes its
quotes as modified derivatives of OPRA. Methods and quote-comparison captions
now identify the indicative feed and its interpretation. Retrieval timestamps
remain distinct from exchange quote timestamps. The 2014–2024 JumpHMM training
window and 2,767 training days are also confirmed.

## Truncated price emissions

On September 14 the author authorized explicitly truncated Student-t emissions
for price simulation. The protocol is recorded in
`code/results/truncated_emissions/PROTOCOL.md`: residual bounds of ±10 fitted
scale units, with ±20 as a wider sensitivity case, specified before rerunning
forecast scores. No fitted neural surface is changed by this correction.

## Further work for prospective validation

Use a frozen chronological training cutoff, then evaluate direct-surface,
last-observed-surface, and dynamic-factor forecasts on subsequent observed
contract marks across multiple origins. Report date-level variation and
event/non-event results on matched rows. The existing in-sample comparisons
and two limited holdouts should not be described as broad temporal validation.

## Scope

The publication edits target `paper-arxiv/`. The shared scenario-engine fix
also affects future runs of other scenario drivers; their older results need
regeneration before reuse. ArXiv submission remains a separate step.
