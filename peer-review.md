# Simulated Peer Review — Heston-IV Paper (JCF submission)

**Manuscript:** *Synthetic American Option Pricing via Jump-HMM-Driven Heston Implied Volatility* (paper-jcf)
**Target venue:** Journal of Computational Finance
**Domain:** Computational finance / stochastic volatility / option pricing

> The peer-review skill's persona descriptions reference metabolic
> engineering and FBA; those were adapted to the actual paper domain
> (Heston/SABR calibration, neural IV surrogates, American option
> pricing) while preserving the moderate / hard / very-hard severity
> gradient.

---

## Reviewer 1 — Senior researcher in computational finance, moderate

### Summary
The paper proposes a synthetic American-option pricing pipeline in
which implied volatility is produced as an output of a structural
return model (JumpHMM → modified Heston → CRR/Leisen-Reimer lattice)
rather than taken as an input. The central methodological move is
making the Heston mean-reversion target $\theta$ depend on regime
state, days-to-expiration, moneyness, and an aggregate market-mood
indicator, with equilibrium initialisation $v_0 = \theta(t{=}0)$ so the
smile/skew/term-structure emerge from the parameterisation of $\theta$.
A neural surrogate $\psi_{\mathrm{NN}}$ replaces the parametric shape
function and is calibrated on a 31-ticker, 15-date, six-sector ladder.
Overall, the construction is sensible, the empirical work is
substantial, and the framework is well-suited to JCF; the main issues
are clarification and additional baselines rather than fundamental
gaps.

### Strengths
1. **The equilibrium-initialisation idea is genuinely useful.** Setting
   $v_0 = \theta(t{=}0)$ (eq.~\ref{eq:v0}, §3.2) eliminates an
   awkward free parameter in standard Heston use and makes the smile
   geometry inheritable from the $\theta$-parameterisation directly.
   The structural framing in the discussion (§6) connecting this to
   the conditional-distribution view of IV is well argued.
2. **The hierarchy of $\psi$ representations (parametric → global NN →
   sector NN → per-ticker NN; Table~\ref{tab:three_way},
   §4.1–4.2) is a clean experimental design.** Isolating the
   capacity axis (parametric vs.\ neural) from the sharing-granularity
   axis (global vs.\ sector) is the right way to decompose where the
   improvement comes from.
3. **The temporal-holdout analysis with the three-way A/B/C contrast
   (Table~\ref{tab:earnings_holdout}, §4.3) is unusually honest.**
   Showing that the $+4.99\%$ generalisation gap collapses to
   $-0.06\%$ when earnings-window rows are excluded is the kind of
   ablation that publication-grade calibration work usually elides.
4. **The scenarios section (§5) connects calibration to a
   risk-management deliverable.** Path-conditional Greeks via central
   finite differences on the LR lattice (Algorithm~\ref{alg:greeks})
   plus terminal P\&L distributions on real-strike contracts is the
   right level of integration to demonstrate the pipeline.
5. **The Leisen-Reimer-for-Greeks justification (§3 closing
   paragraph) is technically correct and well stated.** The
   pinned-strike property is the right reason to prefer LR for FD
   Greeks, and the contrast with CRR aliasing is well drawn.

### Weaknesses
1. **The "IV-as-output, not input" framing is overstated relative to
   the calibration mechanics.** Equation~\eqref{eq:nn_loss} (§3.3)
   trains $\psi_{\mathrm{NN}}$ and $\{\ln\theta_{\text{ticker}}\}$ by
   minimising MSE against $\text{IV}_{\text{market}}$. So at
   calibration time, market IV is the supervision target — the same
   information used to fit SABR/SVI. The "output not input" claim is
   accurate for *forward simulation* (where $\theta(t)$ and $v_t$
   evolve without further calibration) but the abstract and
   introduction read as if no market IV is ever used. *Fix:* qualify
   the claim in the abstract and intro — "IV emerges as an output of
   the forward simulator after one-time calibration of $\psi$ on a
   pooled ladder" — so reviewers do not misread it.
2. **No-arbitrage properties of $\psi_{\mathrm{NN}}$ are not
   discussed.** SABR and SVI deliberately respect calendar-arbitrage
   and butterfly-arbitrage constraints. A neural surrogate has no such
   guarantees by default, and the calibrated surface could exhibit
   negative butterfly densities at the wings or local violations of
   calendar monotonicity. *Fix:* either (a) verify empirically that
   the calibrated $\psi_{\mathrm{NN}}$ surfaces are arbitrage-free
   (compute butterfly $\partial^2 C/\partial K^2 \geq 0$ on a fine
   strike grid and calendar $\partial C/\partial T \geq 0$ on a fine
   maturity grid for each sector network) or (b) discuss the absence
   of guarantees as a known limitation. Either way the issue should
   not be silent.
3. **The 10\% absolute IV RMSE benchmark is never contextualised
   against per-day single-ticker SABR/SVI fits.** A practitioner
   reading Table~\ref{tab:three_way} will ask: a same-day SABR fit on
   SPY typically achieves sub-1\% RMSE per maturity; how does
   pooling-across-dates with a sector network compare to a
   day-by-day, ticker-by-ticker SABR baseline? *Fix:* add one row to
   Table~\ref{tab:three_way} or to Supp.\ Table~\ref{tab:per_sector}
   with a daily-SABR baseline fit ticker-by-ticker, pooled to a single
   RMSE — this would frame whether the 10.24\% reflects genuine
   model error or the cost of cross-date pooling.
4. **The leverage coupling $\rho = -0.6$ and the variance parameters
   $(\kappa = 2.0, \sigma_v = 0.5)$ in §5 are stated without
   justification.** These are standard textbook values, but they are
   not calibrated to the GS or LLY return histories. *Fix:* either
   calibrate $(\kappa, \sigma_v, \rho)$ to the ticker's historical
   return-variance covariance (or to its short-maturity ATM IV term
   structure) or note these are reference values and run a small
   sensitivity sweep.
5. **The cross-ticker validation is only two tickers.** The framework's
   generalisability is claimed in the abstract and conclusion but
   demonstrated on GS and LLY. *Fix:* run the same scenarios on three
   to five additional tickers spanning sectors (e.g., SPY for an ETF,
   XOM for energy, AAPL for liquid tech) and report the entry-edge
   and premium-kept-rate summary statistics in a supplementary table.

### Questions for Authors
1. The peer-distance feature $e^{\text{peer}}$ collapses to zero on
   any day a same-sector equity prints. For tickers in deep sectors
   (Technology with ten names), peer-distance will hit zero on a
   substantial fraction of days during earnings season. Does the
   feature degrade to a coarse seasonality flag? Have you measured
   its mutual information with day-of-week or quarter-week?
2. The CRR pricer uses 200 steps for the full-ladder repricing and
   LR uses 201 steps for the scenarios. Was the choice 200 vs.\ 201
   driven by the LR construction's requirement for odd $N$ in the
   inverse CDF, or is the parity coincidental?
3. In Algorithm~\ref{alg:forward_sim}, line 19 says the JumpHMM
   provides the price-innovation $Z_S$ and an independent $Z_\perp$
   is drawn to construct the variance innovation $Z_v$. Is the
   JumpHMM-implied return innovation Gaussian conditional on the
   state, or are you mapping a non-Gaussian return innovation to a
   standard normal via a probability-integral transform before
   coupling to $Z_v$?
4. The earnings calendar is sourced from Yahoo Finance. Does the
   pipeline handle pre-market vs.\ post-market prints, and is the
   peer-distance feature using calendar days or trading days?

### Requested Experiments/Analyses
1. **No-arbitrage diagnostic.** For each of the six sector
   $\psi_{\mathrm{NN}}$ surfaces, evaluate the implied call price
   $C(K, T)$ on a fine grid and check $\partial^2 C/\partial K^2 \geq
   0$ and $\partial C/\partial T \geq 0$ pointwise. Report the
   fraction of grid points violating each condition in a
   supplementary table.
2. **Per-day SABR baseline.** Fit standard SABR (one set of
   parameters per ticker per capture date, per maturity) on the
   15-date corpus and report the pooled RMSE alongside the three
   model tiers in Table~\ref{tab:three_way}. The point is to show
   the cost of pooling-across-dates relative to the gold-standard
   daily fit.
3. **Heston parameter sensitivity in the scenarios.** Rerun the GS
   scenarios with $\rho \in \{-0.3, -0.6, -0.9\}$ and $\sigma_v \in
   \{0.3, 0.5, 0.7\}$ and report how the terminal P\&L moments and
   premium-kept rates change. Two extra tables would suffice.
4. **Expanded cross-ticker scenarios.** Re-run §5 on three to five
   additional tickers from different sectors and tabulate entry-edge,
   median P\&L, worst-case loss, and premium-kept rate.

### Minor Comments
1. The phrase "the framework is released as an open-source Julia
   package" appears in the abstract and the conclusion but the main
   text never names the package or links to it; the URL appears once
   in the conclusion paragraph. Please add a "Code and data
   availability" section near the end of methods with the package
   name, version, and a one-line invocation.
2. Algorithm~\ref{alg:forward_sim} line 12 reads "equilibrium
   initialisation, eq.~\eqref{eq:v0}" but only line 12 uses $v_0$;
   lines 13–14 enter the loop without re-initialising per path.
   Either move line 12 inside the outer loop (one $v_0$ per path,
   identical because the path index does not affect $v_0$) or note
   that $v_0$ is path-independent.
3. The body word count appears to be ~10,450 (above the JCF 10,000
   guideline). §5 is the longest section at ~2,200 words and could
   absorb a 200-word trim without losing substance.
4. Several figures (Fig.~\ref{fig:gs_paths},
   Fig.~\ref{fig:gs_pnl}) use log-y for the P\&L distributions but
   the caption does not state the binning. Please add the bin count
   or width in the captions.

### Recommendation
**Minor revision.** The contribution is well-motivated and the
empirical work is solid; the issues are clarification, additional
baselines, and one experiment (no-arbitrage check) that should be
either passed or acknowledged as a limitation.

---

## Reviewer 2 — Expert in stochastic volatility / lattice methods, hard

### Summary
The paper offers a structural pipeline that couples a regime-switching
return generator to a modified Heston variance process with a
state-and-contract-dependent mean-reversion target, with $\theta$'s
shape function calibrated as a neural surrogate. The construction is
internally coherent but the paper conflates two different uses of the
framework (calibration to market IV vs.\ forward simulation under the
calibrated model) and the calibration evidence has known weaknesses
that the paper does not fully confront. I cannot recommend acceptance
without substantial additional work on the no-arbitrage question, the
choice of baselines, and the temporal-generalisation evidence.

### Strengths
1. **The decomposition $\sigma^2 = \theta_i \cdot \psi$ is clean and
   the three-way ablation (Table~\ref{tab:three_way}) cleanly
   separates capacity from granularity.** That is the right
   experimental design and the result interpretation is honest.
2. **The temporal-holdout three-config contrast (§4.3,
   Table~\ref{tab:earnings_holdout}) is genuinely informative.** The
   Configuration-B result — $-0.06\%$ generalisation gap once
   earnings windows are excised — is a meaningful diagnostic, and
   the per-ticker pivot in
   Supp.~Table~\ref{tab:tech_pivot} pinpoints the
   peer-coupling mechanism well.
3. **The LR-vs-CRR choice in §3 closing paragraph and §5 is
   technically sound** and the asymmetry between the calibration and
   scenario pricers is justified.

### Weaknesses
1. **The "IV-as-output, not calibration-input" framing is not
   defensible as stated.** The loss in
   eq.~\eqref{eq:nn_loss} explicitly supervises on
   $\text{IV}_{\text{market}}$. The framework is a structural
   *forward simulator* that uses a calibrated shape, but the abstract
   ("smile, skew, and term structure emerged as outputs rather than
   calibration inputs") and the intro ("methodological gap: no
   existing computational pricer generates a self-consistent IV
   surface as an emergent output of a structural return model,
   without requiring observed option data as a calibration target
   during simulation") read as if no market IV is involved. *Fix:*
   re-write these statements to distinguish (a) one-time
   calibration-time supervision of the shape function from (b)
   forward-simulation-time emergence. Otherwise the reader will
   discover the supervision in eq.~\eqref{eq:nn_loss} and
   distrust the framing.
2. **No analysis of no-arbitrage properties of the calibrated surface.**
   This is the single most consequential issue in stochastic
   volatility calibration. SABR-with-arbitrage-fix and SVI
   (Gatheral-Jacquier eSSVI in particular) have explicit
   no-arbitrage parameter regions. The neural $\psi_{\mathrm{NN}}$
   has zero structural guarantees. The paper cannot claim
   "self-consistent IV surfaces" (intro, abstract) without verifying
   absence of static arbitrage. *Fix:* either (a) demonstrate
   empirically on each sector and per-ticker surface that
   $\partial^2 C/\partial K^2 \geq 0$ pointwise and $\partial C
   /\partial T \geq 0$ pointwise (this is one batch of CRR/LR pricings
   on a strike-maturity grid), or (b) explicitly soften the
   "self-consistent" claim to "smile-shaped and term-structured"
   without claiming arbitrage-freeness. The current silence is
   untenable.
3. **No comparison against the standard daily-SABR/SVI baseline.** The
   paper compares parametric vs.\ neural vs.\ sector-neural — these
   are all *novel* representations of the same shape function — but
   never against the actual baselines a practitioner would use
   (per-day SABR per ticker, per-day SVI per ticker). A 10.24\%
   pooled IV RMSE is not interpretable without a same-day fit
   reference. *Fix:* add a baseline column to
   Table~\ref{tab:three_way} that reports per-day SABR fits per
   ticker per maturity, pooled to a single RMSE.
4. **The leverage-effect claim "produced as an emergent property of
   the return dynamics" (§3.2) is asserted, not measured.** The
   paper says large negative returns drive the HMM into low-numbered
   tail states with $p_{\text{neg}} \approx 0.52$, but does not
   report the empirical leverage correlation
   $\text{corr}(r_t, \Delta v_{t+1})$ produced by the simulator
   against the historical leverage correlation on each ticker.
   *Fix:* measure and report the simulated leverage correlation
   under the GS and LLY scenarios in a supplementary table.
5. **The Feller-condition violation handling (§3.2 closing) is
   under-discussed.** Reflecting the Euler scheme at zero
   (eq.~\eqref{eq:euler}) is a known bias source and discretisation
   choice; the paper notes the issue but does not quantify how often
   the reflection is triggered or what the bias is. *Fix:* report
   the fraction of timesteps where the Euler update would have
   produced $v < 0$ in the GS and LLY scenarios, and compare a
   reflection scheme to a more refined alternative (e.g., the
   Andersen 2008 QE scheme already cited) on a held-out subset.
6. **The single-date temporal holdout (one 6-day train, one 2-day
   test, plus one leave-one-date-out) is insufficient evidence for
   the generalisation claim.** *Fix:* report a walk-forward
   validation across the 15-date corpus, training on dates
   $1\ldots k$ and testing on date $k+1$ for $k = 5, \ldots, 14$,
   and tabulate the per-fold gap.
7. **The peer-coupling feature $e^{\text{peer}}$ has a degenerate
   structure during earnings season.** In a sector with ten tickers
   reporting over a six-week window, $e^{\text{peer}}$ hits zero on
   most days; the network learns this as a quarter-weekly seasonal
   signal more than a sector-coupling signal. *Fix:* run an ablation
   with $e^{\text{peer}}$ replaced by a uniformly-drawn placebo
   feature and report the test-RMSE delta. If the placebo recovers
   substantial Configuration-C improvement, the peer feature is
   acting as a seasonal indicator rather than a structural coupling.

### Questions for Authors
1. eq.~\eqref{eq:theta_calibration} sets $\gamma = 0$ for
   calibration and reserves the mood term for forward simulation. In
   §5 the scenarios call $\theta_i$ without the mood term active.
   When is the full mechanism in eq.~\eqref{eq:theta} actually used
   in this paper? It appears the mood term is defined but never
   exercised, only described.
2. eq.~\eqref{eq:psi_nn} writes $\psi_{\text{NN}}$ as a function of
   four inputs but the methods text says the sector network is
   $4 \to 16 \to 16 \to 1$ for groups with ${\geq}2000$ observations.
   With only two inputs in the parametric form
   (eq.~\eqref{eq:psi}: $\tau$, $m$), how do you reconcile model
   capacity comparisons in Table~\ref{tab:three_way}?
3. Why use a per-ticker $\psi_{\mathrm{NN}}$ in §5 for GS and LLY
   when the sector network was the calibration operating point of
   §4? The choice is consequential — per-ticker fits use 1\,000s of
   observations from a single name and will overfit single-day IV
   regimes (as the LLY entry-edge result demonstrates).
4. The mood signal $M_t$ (eq.~\eqref{eq:mood}) is a fraction in
   $[0,1]$, but its scale is highly skewed (most days near 0, rare
   spikes during stress). What distribution does the calibrated
   $\gamma$ recover, and is the linear coupling $1 + \gamma M$
   appropriate given the skew?
5. The 30-day forward simulations assume no earnings event during
   the window (§5). How would the paper rate the practical utility
   of the scenarios for windows that *do* span an earnings print —
   which, for the 31-ticker universe, is the modal use case?

### Requested Experiments/Analyses
1. **Static-arbitrage check on each calibrated $\psi$ surface.**
   Evaluate $\partial^2 C/\partial K^2$ and $\partial C/\partial T$
   on a $20 \times 20$ moneyness-maturity grid for each of the six
   sector surfaces and report the fraction of grid points violating
   each condition. If any sector surface is arbitrage-violating,
   discuss the implications for the scenario results.
2. **Walk-forward temporal validation across the 15-date corpus.**
   Train on dates $1{:}k$, test on date $k{+}1$, for $k = 5,\ldots,14$.
   Report a 10-fold table analogous to
   Table~\ref{tab:earnings_holdout} and quantify the variance of the
   generalisation gap across folds.
3. **Daily per-ticker SABR/SVI baseline on the 15-date corpus.** Fit
   per-day SABR per ticker per maturity, report pooled RMSE, and add
   it as a row to Table~\ref{tab:three_way}. If the pooled SABR
   RMSE is materially below 10\%, the cross-date pooling cost
   should be reported as a known limitation.
4. **Discretisation-bias quantification.** Run a 100\,000-path
   subset of the GS scenario at $\Delta t = 1/252$ (current) and
   $\Delta t = 1/(252 \cdot 10)$ (sub-daily); compare terminal
   variance distributions, premium-kept rates, and worst-case
   losses. The full Andersen-2008 QE scheme as a third arm would
   close the gap.
5. **Placebo ablation on $e^{\text{peer}}$.** Replace the peer
   feature with a uniform random feature in $[-30, 30]$, retrain
   Configuration~C, and report the test RMSE. If the placebo
   recovers substantial improvement over Configuration~A, the
   peer-coupling claim weakens substantially.
6. **Empirical leverage-correlation check.** Compute
   $\text{corr}(r_t, \Delta v_{t+1})$ on the 1\,000 GS simulated
   paths against the same correlation on the GS return history; the
   claim that leverage "emerges" from the regime dynamics needs a
   numerical anchor.

### Minor Comments
1. eq.~\eqref{eq:psi_nn} writes the surrogate as $\psi_{\text{NN}}(\ln\tau,\ln m,e,e^{\text{peer}})$ but the parametric $\psi$ in eq.~\eqref{eq:psi} is only $\psi(\tau,m)$. The dimensionality mismatch is buried in the prose; please flag it explicitly when the calibration table compares the two.
2. The per-ticker network in §4.2 uses a $2 \to 16 \to 16 \to 1$ architecture, but the sector network in §3.3 uses $4 \to 16 \to 16 \to 1$. The per-ticker fit drops $e, e^{\text{peer}}$ silently; please state this and justify.
3. The "1.50× the worst put loss" framing in §5 and the conclusion is repeated several times. The asymmetry is real but the multiplier is per-contract, not per-dollar-of-premium-collected. Stating it as a ratio of risk-to-premium (worst-case loss / premium received) would be more conservative.
4. Table~\ref{tab:per_sector} reports ETF dropping from $10.54\% \to 6.10\%$ across the parametric → sector NN columns, an $-4.44\%$ absolute improvement. The text rounds to "the largest absolute gains in ETF and financials" but Financials only moves $9.05 \to 6.70 = -2.35\%$, so ETF dominates the gain by ~2× — worth being more precise.
5. The market-mood mechanism $M_t$ is referenced as "novel" in spirit but resembles the structural-stress indicators used in factor-volatility models (e.g., the cross-sectional dispersion in equity returns); a one-line acknowledgement of conceptual lineage would be appropriate.

### Recommendation
**Major revision.** The framework is interesting and the empirical
work is substantial, but the no-arbitrage question must be addressed
and the SABR/SVI baseline must be added before the methodological
claims are publishable. The "IV-as-output" framing also needs to be
qualified throughout.

---

## Reviewer 3 — Skeptic with competing work in deep IV calibration, very hard

### Summary
The paper offers a five-step pipeline (JumpHMM, modified Heston, CRR
lattice for ladder, LR lattice for scenarios, neural shape function)
and claims its main novelty is producing "self-consistent IV surfaces
as an emergent output of a structural return model." Several layers
of that claim do not survive close reading: the shape function is
supervised on market IV during calibration (eq.~\eqref{eq:nn_loss}),
the comparison set excludes the actual baselines a practitioner
would use, and the cross-ticker validation is two contracts from a
single capture day. The pipeline is competently executed but the
claimed methodological novelty over the existing
deep-calibration / neural-surrogate literature
(Horvath et al.\ 2021; Bayer et al.\ 2019, both cited but not
benchmarked) is unconvincing. I recommend major revision with
substantial additions, and would not be surprised if some of those
additions reveal the framework's central claims to be overstated.

### Strengths
1. **The temporal-holdout three-configuration analysis (§4.3) is the
   strongest section of the paper.** The Configuration-B result is
   genuinely useful as a debugging artefact and the per-ticker
   pivot in Supp.~Table~\ref{tab:tech_pivot} is exemplary.
2. **The choice to release a Julia package is appropriate for JCF and
   raises the bar for reproducibility.**
3. **The LR-pinned-strike justification for FD Greeks is correct and
   the asymmetric pricer choice (CRR for ladder, LR for scenarios)
   is well argued.**

### Weaknesses
1. **The novelty over Horvath et al.\ 2021 is unestablished.**
   Horvath et al.'s deep-calibration framework provides a neural
   surrogate for IV under rough-Bergomi-like models and validates it
   on a real options corpus. The paper cites this line of work
   (§2.4) and explicitly says "We used the neural surrogate in the
   structural sense rather than the deep-calibration sense" — but
   the operational distinction is not measured. *Fix:* fit a direct
   neural IV surface (Horvath-style) on the same 15-date corpus,
   without the $\theta$-decomposition, and compare RMSE. If the
   direct fit is competitive, the entire $\theta \cdot \psi$
   decomposition needs further justification beyond
   interpretability.
2. **The "IV-as-output" claim is the headline novelty but the
   calibration loss
   (eq.~\eqref{eq:nn_loss}) directly supervises on market IV.** No
   reader of §3.3 can come away believing that market IV is not
   used. The novelty is at *forward-simulation time*, where the
   calibrated $\psi$ is held fixed and IV emerges from path
   dynamics — that is real but it is not what the abstract claims.
   *Fix:* qualify the abstract, intro, and discussion to say "after
   one-time calibration on a multi-date corpus, IV is propagated
   forward as an output of the structural model" and stop framing
   the contribution as eliminating market IV from the pipeline.
3. **Comparison to actual baselines is missing.** A reader expects to
   see (a) per-day SABR per-ticker, (b) per-day SVI per-ticker, (c)
   a direct neural IV surface (Horvath-style) fit on the same
   pooled corpus, and (d) at minimum a Heston with constant $\theta$
   calibrated on the same pooled data. The paper compares three
   *novel* representations of the same shape function (parametric
   $\psi$, global NN, sector NN, per-ticker NN) but none of these
   are the baselines a practitioner would benchmark against. *Fix:*
   add all four baselines to Table~\ref{tab:three_way} or to a
   single comparison table. If the framework does not dominate
   these baselines, the contribution narrative needs revision.
4. **No-arbitrage of $\psi_{\mathrm{NN}}$ is not checked.** This is
   the standard objection to neural IV surrogates and the paper does
   not address it. A "self-consistent IV surface" (abstract) is a
   strong claim that requires the surface to be free of static
   arbitrage. *Fix:* prove or disprove empirically on a strike-maturity
   grid for each of the six sector surfaces and the per-ticker
   surfaces actually used in §5 (GS, LLY).
5. **The cross-ticker validation in §5 is two contracts.** GS and
   LLY are both data-rich high-cap names; the framework's claim is
   universal, but the evidence is from contracts that are easy to
   price. *Fix:* add scenarios for a low-liquidity name (e.g., a
   small-cap technology ticker from the universe), a contract with
   wider bid-ask, and a contract that spans an earnings event.
6. **The leverage parameter $\rho = -0.6$ is hardcoded.** Standard
   Heston practice calibrates $\rho$ from short-maturity ATM skew.
   The paper uses a textbook value without justification. *Fix:*
   either calibrate per-ticker from the implied ATM skew or run a
   sensitivity sweep.
7. **The "endogenous market mood" eliminating VIX (§3.2) is a
   solution in search of a problem.** External market-stress
   indicators are widely available, well-understood, and free of
   the JumpHMM's parameter-dependence. The claim that an endogenous
   mood signal is preferable would need evidence that the
   JumpHMM-derived $M_t$ predicts realised IV spikes better than the
   VIX on a held-out subset. *Fix:* compute $\text{corr}(M_t,
   \text{VIX}_t)$ and the marginal information content of $M_t$
   beyond VIX on the calibration corpus; if $M_t$ does not dominate
   VIX, the "endogenous" framing should be softened.
8. **The earnings-aware feature recovers only 21\% of the
   generalisation gap** ($4.99 \to 4.36$ improvement of $1.05\%$
   IV; Table~\ref{tab:earnings_holdout}), and the paper reframes
   $7.96\%$ as "the right reference" once earnings observations are
   excluded. This is a goalpost shift: the practical use case of
   the framework is precisely to price options over
   earnings-bearing windows, and on that use case the framework
   covers ${\sim}20\%$ of the structural error. *Fix:* report the
   $4.36\%$ residual gap in the abstract or conclusion as the
   honest performance number; do not bury it under the non-event
   reframing.
9. **LLY's entry edge of $-\$1.35$ (put) and $-\$1.80$ (call) on a
   real trade-date contract is a model-vs-market disagreement of
   roughly $7\%$ of premium, not an "as-expected" outcome.** The
   paper explains the gap as an IV-regime mismatch from time-averaged
   $\psi_{\mathrm{NN}}$, which is honest, but the explanation is
   also an admission that the framework cannot price the very
   contracts it is calibrated against on the day they trade. *Fix:*
   either calibrate per-day $\theta_i$ alongside the static $\psi$
   (a "level update on a frozen shape"), or explicitly note that
   the framework is intended for forward simulation, not same-day
   pricing.

### Questions for Authors
1. Why is the JumpHMM cited as a black box (\cite{jumphmm2025},
   §2.1) with the marginal calibration left implicit? The entire
   forward simulation rests on the JumpHMM marginal being a faithful
   return generator on the per-ticker history; this is the rate-limiting
   assumption and deserves more than three lines of background.
2. Eq.~\eqref{eq:theta_calibration} eliminates regime conditioning
   ($\theta_{i,s_t} \to \theta_i$) for calibration because the
   corpus is observation-day surfaces. So during calibration, the
   regime-state dependence — which is the methodological hook of
   the paper — is unused. How can the calibration validate a
   framework whose core mechanism is not active during fitting?
3. The CRR step count of 200 (§3.2) is justified by oscillation
   amplitude $<\$0.05$. On a near-the-money option, $0.05$ is a
   substantial fraction of theta. What is the convergence-error
   ceiling on the ladder RMSE attributable to the CRR finite step?
4. Algorithm~\ref{alg:forward_sim} updates $v_{t+1}$ using the
   reflected Euler scheme but the manuscript Methods says
   $\theta(t) = \theta_i \cdot \psi$ in calibration and $\theta(t) =
   \theta_{i,s_t}(1 + \gamma M_t)\psi$ in the full mechanism. Which
   $\theta_t$ does Algorithm~\ref{alg:forward_sim} use? The
   pseudocode is ambiguous on whether the mood multiplier is active
   in the GS/LLY scenarios.
5. The premium-kept-rate sanity check against $1 - |\Delta|$
   recovers within $3\%$ on GS and $7\%$ on LLY. The LLY gap is
   reported but the asymmetric *direction* (model is more bullish
   than market on both legs simultaneously) is not addressed —
   what causes that?

### Requested Experiments/Analyses
1. **Head-to-head with Horvath-style direct neural IV surrogate.**
   Fit a single neural network that takes $(\ln\tau, \ln m,
   \text{ticker-embedding}, e, e^{\text{peer}})$ and returns IV
   directly, trained on the same 15-date pooled corpus with the
   same MSE-IV loss. Report pooled RMSE and per-sector RMSE
   alongside the existing four model tiers. This is the right
   baseline against which to measure the value of the $\theta_i
   \cdot \psi$ decomposition.
2. **Static-arbitrage check on every surface actually used in §5
   (GS per-ticker, LLY per-ticker, six sector surfaces).** Same
   framing as Reviewer 1's request, but extended to the per-ticker
   surfaces.
3. **Earnings-window scenarios.** Re-run §5 with the simulation
   window deliberately spanning a known earnings print for one of
   the 31 tickers and report whether the framework's terminal P\&L
   distribution captures the realised event-day move. This is the
   modal use case the framework was sold against, and it has been
   sidestepped.
4. **Endogenous-mood validity check against VIX.** Compute the
   JumpHMM-derived $M_t$ on the 15-date corpus and compare its
   correlation with VIX, its in-sample predictive power for realised
   ATM IV moves, and its incremental information content beyond
   VIX in a regression. If $M_t$ is dominated by VIX, the
   endogenous-mood claim should be retracted.
5. **Per-day Heston with constant $\theta$ baseline.** Fit standard
   Heston (constant $\theta$, no $\psi$, no neural surrogate) on the
   same 15-date corpus and report the pooled RMSE. This isolates
   the value of the $\theta$-decomposition itself, separate from
   the neural shape function.
6. **Sensitivity of forward-scenario P\&L to JumpHMM
   parametrisation.** Report how the GS premium-kept rate and
   worst-case loss change under a JumpHMM fit with $N=5$ vs.\ $N=7$
   states. If the worst-case loss varies by a factor of two, the
   framework's robustness claim weakens.

### Minor Comments
1. The abstract calls the framework an "open-source Julia package"
   and the conclusion gives a URL, but the main text never names
   the package. A package-level reference (with version and DOI)
   should appear in the methods.
2. "Negligible linear autocorrelation" (§2.1) is the standard
   stylized-fact list but the paper does not quantify the
   correlation produced by the simulated paths. Add a line.
3. eq.~\eqref{eq:euler} reflects the Euler step at zero; this is
   not a discretisation of CIR but a numerical hack. The Andersen
   2008 QE scheme is cited but not used. Justify or use.
4. The "delta-as-probability" check is presented as a sanity check
   but the Black-Scholes derivation in §5 closing paragraph
   assumes near-zero drift, while the JumpHMM marginal carries a
   long-run drift anchor (per the project memory, GS calibration
   window had non-trivial drift). The sanity check's residual is
   thus partly attributable to the drift, partly to the fat tail —
   please decompose.
5. The bibliography mixes preprints (jumphmm2025) with peer-reviewed
   sources without distinguishing maturity. For a JCF submission,
   either submit jumphmm2025 in parallel or note its preprint status
   explicitly in the citation.

### Recommendation
**Major revision.** The framework is competently implemented but the
methodological novelty over the existing deep-calibration literature
is not established, the comparison baselines are wrong, the
no-arbitrage question is unresolved, and the central
"IV-as-output" framing is not what the calibration mechanics
support. With substantial baseline additions, an arbitrage check, a
walk-forward validation, an earnings-window scenario, and a
qualification of the abstract claims, this could be a publishable
JCF paper.

---

## Summary of Actionable Items

### Required experiments (high priority — appear in multiple reviews)
1. **Static-arbitrage check on each calibrated $\psi$ surface**
   (R1, R2, R3): evaluate $\partial^2 C/\partial K^2 \geq 0$ and
   $\partial C/\partial T \geq 0$ on a strike-maturity grid for the
   six sector $\psi_{\mathrm{NN}}$ surfaces *and* the per-ticker
   surfaces actually used in §5 (GS, LLY). Report violation rates
   in a supplementary table.
2. **Per-day SABR/SVI baseline on the 15-date corpus** (R1, R2,
   R3): add a row to Table~\ref{tab:three_way} or an adjacent
   table reporting pooled RMSE from per-day, per-ticker,
   per-maturity SABR (or SVI) fits.
3. **Direct neural IV-surface baseline (Horvath-style)** (R3): fit
   a single network from $(\ln\tau, \ln m, \text{embedding}, e,
   e^{\text{peer}})$ to IV without the $\theta \cdot \psi$
   decomposition; compare RMSE.
4. **Walk-forward temporal validation** (R2): roll the train/test
   split across the 15-date corpus and report the per-fold gap.
5. **Expanded cross-ticker scenarios** (R1, R3): add scenarios for
   three to five additional tickers (low-liquidity, wider bid-ask,
   one with earnings inside the window).
6. **Earnings-window scenario** (R3): re-run §5 with the simulation
   spanning a known earnings print to test the framework on its
   modal use case.

### Required clarifications (high priority)
1. **Qualify "IV-as-output" framing** (R1, R2, R3): explicitly
   distinguish one-time calibration-time supervision (where market
   IV is the target via eq.~\eqref{eq:nn_loss}) from
   forward-simulation-time emergence. Affects abstract, intro,
   conclusion.
2. **Reconcile abstract claim with the residual generalisation gap**
   (R3): the $4.36\%$ event-aware residual gap should appear in the
   abstract or conclusion alongside the $7.96\%$ non-event number.
3. **Disambiguate which $\theta(t)$ algorithm 1 uses** (R3):
   clarify whether the GS/LLY scenarios activate the
   regime-state-and-mood mechanism or operate at the
   calibration-collapsed $\theta_i$.

### Additional validations (medium priority)
1. **Placebo ablation on $e^{\text{peer}}$** (R2): replace with
   uniform random feature and report Configuration-C test RMSE
   delta.
2. **Endogenous-mood vs.\ VIX** (R3): test $M_t$ against VIX on
   correlation, predictive power for ATM IV moves, and incremental
   information.
3. **Discretisation-bias quantification** (R2): compare reflected
   Euler at $\Delta t = 1/252$ vs.\ sub-daily and vs.\ Andersen
   QE on a 100k-path subset.
4. **Empirical leverage-correlation check** (R2): measure
   $\text{corr}(r_t, \Delta v_{t+1})$ on simulated vs.\ realised
   for GS and LLY.
5. **Heston parameter sensitivity** (R1): rerun GS scenarios with
   $\rho \in \{-0.3, -0.6, -0.9\}$ and $\sigma_v \in \{0.3, 0.5,
   0.7\}$.
6. **Per-day Heston constant-$\theta$ baseline** (R3): isolate
   value of $\theta$-decomposition itself from neural surrogate.
7. **JumpHMM-parameter sensitivity in §5** (R3): rerun GS
   scenarios under different JumpHMM state counts.

### Minor revisions (low priority)
1. Trim §5 by ~200 words to land under the JCF 10\,000-word cap
   (R1).
2. Add a "Code and data availability" section naming the Julia
   package, version, and DOI (R1, R3).
3. Fix Algorithm 1 line 12 placement / $v_0$ per-path scoping (R1).
4. Add bin count/width to log-y P\&L figure captions (R1).
5. Flag $\psi$ vs.\ $\psi_{\mathrm{NN}}$ input dimensionality
   mismatch explicitly when comparing model tiers (R2).
6. Replace "1.50× the worst put loss" framing with worst-case loss
   per dollar of premium collected (R2).
7. Address per-ticker vs.\ sector NN architecture asymmetry
   ($2\to16\to16\to1$ vs.\ $4\to16\to16\to1$, with $e,
   e^{\text{peer}}$ silently dropped) (R2).
8. Quantify "negligible linear autocorrelation" with a numerical
   value (R3).
9. Note jumphmm2025 preprint status explicitly in citation (R3).
10. Justify or replace reflected-Euler with the cited Andersen QE
    scheme (R3).

### Editor's likely overall judgment
With the moderate reviewer requesting minor revision and both hard
reviewers requesting major revision, the most likely editorial
verdict is **major revision** with an invitation to resubmit. The
no-arbitrage check, the SABR/SVI and direct-NN baselines, and the
"IV-as-output" qualification are the three non-negotiable items.
