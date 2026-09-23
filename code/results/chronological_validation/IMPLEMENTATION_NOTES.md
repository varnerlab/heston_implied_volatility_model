# Implementation checks

- 2026-09-07: independently reconstructed every saved legacy fold train/test row
  count; all 53 matched. All legacy accepted rows had finite noncrossed asks and
  positive displayed sizes. The new analysis uses its own stricter capture audit.
- Corrected an initial prose reference to a July 2 early close after checking the
  NYSE official 2026 calendar. July 2 is a regular session. No calendar code,
  capture eligibility, training rows, or endpoint changed; the implemented rule
  already used 20:00 UTC. Source: https://www.nyse.com/trade/hours-calendars
- Added calendar, CRPS, origin-anchoring, and quote-eligibility checks. A 32-path
  smoke run at the first GS and LLY origin completed. It is stored separately
  and is excluded from every reported forecast score.
- Completed all six July-cutoff surface fits and both 1,000-path forecast cohorts.
  The full Julia suite passed 219 checks, the Python quote/calendar suite passed
  two tests, and saved-path reconstruction passed 136 checks. Reconstruction
  covered both tickers and both cohorts and reproduced forecast means, quantiles,
  and CRPS without refitting or generating new paths.
- The final 32-page manuscript was compiled and the affected pages visually
  inspected. All 35 figure/table labels were cited in the Results. There were no
  unresolved references, overfull boxes, or LaTeX errors. The existing author-email
  font substitution warning remains. `git diff --check` passed.
- The 42-file arXiv source package was extracted and compiled separately with
  halt-on-error enabled. Its extracted PDF text exactly matched the reviewed
  manuscript. `arxiv_source_manifest.json` records all packaged file hashes.
  The package has been prepared locally; no submission was uploaded.
