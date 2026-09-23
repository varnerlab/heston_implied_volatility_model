Resolution update, September 14, 2026: findings 1 and 2 have been addressed in
the local revision. The paper explicitly identifies the free Alpaca indicative
feed. Price simulations use truncated Student-t emissions, with completed primary
and wider-bound reruns and sensitivity tables. The PDF, plain abstract, and source
archive have also been rebuilt and independently checked, resolving finding 5.
The source archive contains 55 files and the manuscript is 40 pages. See
`code/results/truncated_emissions/FINDINGS.md` and `verification.json`.
The author's daily downloads through a free Alpaca account are confirmed; no
further provenance question is pending. The original review below is preserved
as a record of the pre-correction state. Other findings remain separate tasks.

---

Review of the arXiv manuscript and associated code, September 14, 2026.

I would address findings 1 and 2 before submitting the replacement. The current framing of the work as a controlled comparison is supported by the saved experiments, but the data description and interpretation of price-distribution statistics need correction. Findings 3–8 concern code reuse and release preparation. This review leaves the manuscript, numerical results, and source code unchanged.

1. **High priority: identify the observations as Alpaca indicative-feed data.**

   The author confirmed during this review that collection used only the free indicative feed. [Methods](paper-arxiv/sections/methods.tex) says “free-tier account,” while the Results, figure captions, and forecast evaluation refer to market mids and bid–ask spreads. The collector in the sibling Alpaca SDK calls `/options/snapshots` without a `feed` parameter. [Alpaca's snapshot documentation](https://docs.alpaca.markets/us/reference/optionsnapshots) states that the unsubscribed default is indicative and that its quotes are modified. Its [data-source documentation](https://docs.alpaca.markets/us/docs/historical-option-data) distinguishes these derivatives from actual OPRA quotes.

   The reported errors remain valid comparisons with the captured vendor data. However, inside-spread rates and the endpoint-IV repricing diagnostic do not establish agreement with exchange-observed spreads. Naming the provider and recording retrieval times do not resolve this distinction. Identify the feed explicitly in Methods, qualify the market-comparison claims and captions, and add the limitation to Discussion. Preserve the paired IV ablation as a controlled simulation experiment. Direct validation against exchange quotes would require a separately identified OPRA dataset. Future collection should request and record the feed explicitly and retain quote timestamps.

2. **High priority: unbounded Student-t log returns do not support finite population stock-price means or short-call expected losses.**

   [The forecast runner](code/scripts/run_chronological_forecasts.jl), lines 27–31 and 77–79, draws JumpHMM emissions and constructs stock prices by exponentiating cumulative log returns. Inspection of the installed, manifest-pinned JumpHMM sampler found `mu + sigma * rand(TDist(nu))`, with no truncation. Both saved GS and LLY marginals have `nu = 5`, positive emission scales, and positive stationary probability for all 50 states. The later [stock comparison](code/scripts/run_small_stock_comparison.jl) uses the same construction.

   For a nondegenerate Student-t random variable X and any b > 0, E[exp(b X)] is infinite: exponential growth dominates the polynomial density tail. Scaling by 1/252 or shifting the mean of log growth does not change that fact. Consequently, already at the first forecast step the modeled stock price has an infinite mean. An American call mark is at least max(S − K, 0), so its mean also diverges; the associated short-call population mean P&L and lower-tail expected shortfall are negative infinity. Stock and call price variances are not finite. The full parent distribution also has infinite price-space CRPS, although the CRPS of each finite empirical ensemble is calculable. This is a tail property of the physical scenario distribution and does not require identifying it with the lattice's pricing measure. See [Cassidy, Hamp, and Ouyed](https://arxiv.org/pdf/0906.4092), especially their discussion of truncation and divergent integrals.

   The saved numbers are finite sample calculations; this finding does not imply an arithmetic error in those files. It does mean that a finite mean, standard error of that mean, or short-call expected shortfall cannot be interpreted as a convergent estimate of the corresponding finite population quantity. Relevant outputs include [distribution_scores](code/src/ForecastValidation.jl), lines 17–25, and the call columns in Supplementary Tables S11 and S12. Quantiles and event probabilities remain meaningful. Paired differences between IV variants require their own moment analysis; divergence of the individual call means alone does not prove divergence of the paired differences.

   The strongest remedy is a documented return-tail specification with finite exponential moments, followed by regeneration and sensitivity checks for affected results. An alternative is to explicitly restrict the claims to the fixed finite ensembles and remove population-moment/convergence interpretations, with robust quantile or log-return diagnostics where appropriate. Merely increasing the path count does not fix the mathematical issue. Do not silently clip outcomes after inspecting forecast scores.

3. **Medium priority: the exported scenario API still implements the older model and has clock errors.**

   `HestonIV.run_single_asset_scenario` remains exported in [HestonIV.jl](code/src/HestonIV.jl). In [Pipeline.jl](code/src/Pipeline.jl), line 224, it keeps `prices_i[1:n_sim_steps]` although reconstruction returns the initial spot plus all simulated transitions. At line 253 it prices with the original contract DTE even at later evaluation steps. The multi-asset entry point similarly retains the original maturity. [HestonVariance.jl](code/src/HestonVariance.jl), lines 64–66, uses independent Gaussian innovations and an absolute-value reflection, whereas the manuscript specifies shared return-coupled innovations and a hard floor.

   A direct probe requesting two returns produced only two stored prices, rather than an initial price plus two returns. For a one-day call evaluated after the first transition, one path had spot 99.36755 and a positive mark of 0.17823 instead of the zero expiration payoff. This reproduces a functional error outside the manuscript's corrected drivers.

   The paper's `ScenarioTemplate` and `DynamicAblation` implementations use the corrected construction; the published saved forecasts were checked through those paths. Route the exported API through the maintained engine, or clearly deprecate and identify the old API so readers do not mistake it for the implementation used in this revision.

4. **Medium priority: CRR discounts at r − q when q is nonzero.**

   In [CRRTree.jl](code/src/CRRTree.jl), both pricers set `R = exp((r-q)*dt)` and then use `disc = 1/R` (lines 51 and 96). The transition drift should use r − q, but discounting should use r. In a direct European-call comparison with S = K = 100, volatility 0.20, r = 0.05, q = 0.03, T = 1, and 1,000 steps, CRR returned 8.91406 versus Black–Scholes–Merton 8.65253, approximately 3.02% too high. Refining the tree does not remove the incorrect discount factor.

   The manuscript sets q = 0, so this does not change its reported prices. Correct both discount factors and verify dividend-paying European parity and an American dividend case before offering this functionality for reuse.

5. **Medium priority: the source, PDF, and upload derivatives are out of sync.**

   [abstract.tex](paper-arxiv/sections/abstract.tex) is modified relative to HEAD. It is also the only packaged dependency whose current content differs from the recorded 51-file archive manifest. The current PDF, [plain abstract](paper-arxiv/abstract-v2.1.txt), and [source archive](paper-arxiv/arxiv-source-v2.1.tar.gz) still contain the earlier abstract with explicit RMSE and MAE values. The existing PDF was built September 7.

   The current reachable source compiled successfully into a separate 36-page review PDF. After the substantive edits, rebuild the deliverable PDF and run `python3 code/scripts/package_arxiv_source.py` so all three submission files agree. Update the arXiv title and abstract fields as well as the uploaded source. The packaging helper correctly refuses sources newer than the PDF; retaining an old archive bypasses that protection at upload time.

6. **Medium priority: the paper's repository link lands on an older default branch.**

   [Supplement](paper-arxiv/sections/supplement.tex), line 12, links to the repository root. The reviewed revision is on `corpus-extension-58-dates`, at commit `cc8201d21ddc671cb64d9e2316d1f3f7d129a375` plus the uncommitted abstract edit. The inspected public default-branch page and local `origin/main` have the old two-line README, rather than the revision's reproduction instructions. Publish a stable tag or commit containing the final code and link to it explicitly. This matters even if the development branch is already pushed.

7. **Lower priority: the documented reproduction sequence requires an author-local sibling checkout.**

   The first chronological-reproduction command in [README.md](README.md) runs `sync_ladder_extended.jl`. That script defaults to `/Users/jdv27/Desktop/julia_work/alpaca-markets-sdk/data` and errors when that directory does not exist ([source](code/scripts/sync_ladder_extended.jl), lines 25–26 and 65). The frozen capture files are already retained in this repository. Mark the synchronization command as optional for extending collection, and start reproduction from the committed snapshots. Also align SETUP.md with the current Python dependencies; it still describes Python as needed only for the earnings fetcher.

8. **Lower priority: the Makefile ignores compiler failures.**

   Every build recipe line in [Makefile](paper-arxiv/Makefile), lines 11–14, begins with `-`, which tells Make to ignore a failing command. The compiler also lacks `-halt-on-error`. A successful `make` exit therefore is not sufficient evidence of a valid new manuscript. Use a build that propagates failures and rejects TeX errors before packaging. The isolated review build used `latexmk -pdf -interaction=nonstopmode -halt-on-error` and passed.

Verification completed: 227 Julia tests, six Python tests, and 136 saved-forecast reconstruction checks passed. Twenty-one selected source/input/generated-table hashes matched their recorded values. Independently reaggregating the shorter-maturity forecast rows reproduced the main frozen/coupled MAEs: GS 4.2206/4.2002 and LLY 11.6207/11.5877. The current source compiled to 36 pages without undefined references, duplicate labels, or overfull/underfull warnings; only the existing font-substitution warning remained. Rendered overviews of all pages showed no obvious clipping or overlap. The existing strike-arbitrage limitations, dependent origins, small option sample, retrospective benchmark selection, and observed-stock diagnostic are disclosed in the manuscript. I did not rerun the full neural fits or every simulation experiment, and this was not a fresh-machine dependency installation test.

The [current public arXiv record](https://arxiv.org/abs/2605.13998) still shows v1 under the earlier title. Describe this replacement as a substantive revision. Suggested Comments text, adjusted to the final changes: “Major revision: corrected simulation timing; added paired IV ablation, chronological evaluation, and stock benchmarks; clarified data provenance and revised interpretation; updated title.” [arXiv's replacement guidance](https://info.arxiv.org/help/replace.html) asks authors to describe the changes in the Comments field; previous versions remain available.
