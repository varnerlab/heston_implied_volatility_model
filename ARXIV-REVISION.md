# ArXiv revision: changes and remaining evidence

## Completed work

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

The ablation can establish how IV assumptions affect interim marks and
liquidation risk on fixed stock paths. It cannot select the best model of
future observed IV. Expected shortfall changes are descriptive; paired
standard errors in the paper apply to mean P&L differences only. Simulation
uncertainty excludes fitted-model and pilot uncertainty.

The GS direct surface and some factor variants violate necessary strike
conditions on the diagnostic grid. This is an empirical limitation, not
fixed by relabeling or by increasing the lattice depth. An arbitrage-constrained
surface or projection remains a separate modeling extension. The audit does
not cover calendar consistency or dynamic no-arbitrage.

## Confirmed data provenance

The author identified Alpaca Markets, accessed through a free-tier account,
as the option-data provider and specified the standard 2014–2024 JumpHMM
training window. The saved model records 2,767 training days.

Alpaca's [Market Data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq),
checked September 5, 2026, documents Black–Scholes calculations and a
Vega-based iterative IV solver. The paper cites that documentation and states
that the reported IV field was fitted directly. Reconstructing the vendor's
complete internal implementation is not a prerequisite for this study.

The previous statements treating vendor-method details and the training
window as unresolved submission requirements have been removed.

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
