# Submission Audit Results — Revision

## Executive assessment

The paper is not submission-ready in its current form. The pooled IV-surface calibration result may be salvageable, but the forward-scenario results should be rerun after correcting time-scale and model-definition problems. The manuscript also overstates what the reported experiments validate.

No source code or manuscript files were changed during the audit. The existing test suite passes 129/129 tests, but it does not cover the scenario engine, Leisen–Reimer implementation, SABR baseline, IV inversion, cache invalidation, or end-to-end reproduction of manuscript results.

## Critical findings

### 1. The construction is not a Heston model in the standard sense

Heston has one latent instantaneous variance process per underlying; its option smile emerges from risk-neutral pricing over that stochastic variance. Here, every `(ticker, strike, DTE)` receives a separate variance trajectory, and `sqrt(v)` is declared to be Black–Scholes implied volatility before being inserted into a constant-volatility lattice.

Relevant locations:

- [`paper-jcf/sections/methods.tex`, line 38](paper-jcf/sections/methods.tex#L38)
- [`code/src/HestonVariance.jl`, line 38](code/src/HestonVariance.jl#L38)

This is better described as a dynamic IV-surface model with contract-specific CIR-like factors. Calling it “Heston implied volatility” invites a fundamental reviewer objection: instantaneous Heston volatility is not equal to Black–Scholes implied volatility.

### 2. The advertised JumpHMM state/mood innovation is disabled in the reported scenarios

The paper presents state-dependent `theta[i,s_t]` and aggregate mood as the central innovation, but the headline experiments explicitly set `s_t = s_0` and `gamma = 0`. The scenarios use ticker marginals, not the multi-asset mood mechanism.

Relevant locations:

- [`paper-jcf/sections/methods.tex`, line 23](paper-jcf/sections/methods.tex#L23)
- [`paper-jcf/sections/scenarios.tex`, line 5](paper-jcf/sections/scenarios.tex#L5)
- [`paper-jcf/sections/discussion.tex`, line 3](paper-jcf/sections/discussion.tex#L3)
- [`paper-jcf/sections/conclusion.tex`, line 3](paper-jcf/sections/conclusion.tex#L3)

The discussion and conclusion therefore cannot claim that the reported results validate regime/mood-coupled dynamics. This is a central claim–evidence mismatch.

### 3. Calendar days and trading days are mixed, invalidating the reported P&L distributions

The GS option has 31 calendar days between April 28 and May 29, 2026. That interval contains 23 weekdays and approximately 22 trading sessions after the Memorial Day holiday. The engine nevertheless simulates 31 JumpHMM trading returns, while discounting and evolving variance with `1/365`.

Relevant locations:

- [`code/src/ScenarioTemplate.jl`, line 238](code/src/ScenarioTemplate.jl#L238)
- [`code/src/ScenarioTemplate.jl`, line 274](code/src/ScenarioTemplate.jl#L274)
- [`code/src/ScenarioTemplate.jl`, line 347](code/src/ScenarioTemplate.jl#L347)
- [`paper-jcf/sections/scenarios.tex`, line 10](paper-jcf/sections/scenarios.tex#L10), which instead specifies `Delta t = 1/252`

This overextends the return horizon and changes strike-breach rates, worst losses, premium-kept rates, leverage shocks, Greeks, and terminal P&L statistics. The GS, LLY, expanded-ticker, and INTC scenario results need to be regenerated after adopting one consistent clock.

### 4. The reported variance parameters are fixed rather than calibrated

The scenarios use `(kappa, sigma_v, rho) = (15, 0.5, -0.6)` without an empirical calibration or adequate sensitivity analysis.

Relevant location:

- [`code/src/ScenarioTemplate.jl`, line 66](code/src/ScenarioTemplate.jl#L66)

The package's `calibrate` function advertises calibration of `kappa` and `sigma_v`, but its objective never uses either parameter:

- [`code/src/Calibration.jl`, line 148](code/src/Calibration.jl#L148)

They are therefore unidentified flat directions. Nelder–Mead can return arbitrary values without affecting the IV loss. A diagnostic run with different starting values returned radically different `kappa` and `sigma_v` estimates for the same calibration problem.

### 5. The manuscript's discretization does not match the scenario code

The paper uses an absolute-value reflecting boundary. The actual headline engine clamps variance to a hard `0.5%`-IV-equivalent floor. These operations have different transition distributions and stationary behavior.

Relevant locations:

- [`paper-jcf/sections/methods.tex`, line 51](paper-jcf/sections/methods.tex#L51)
- [`code/src/ScenarioTemplate.jl`, line 261](code/src/ScenarioTemplate.jl#L261)
- [`code/src/ScenarioTemplate.jl`, line 304](code/src/ScenarioTemplate.jl#L304)

The package's separate generic simulator uses `abs`, but it is not the engine that generated the paper's GS/LLY results.

### 6. The LoRA evidence is entirely in-sample

Each adapter is trained and evaluated on the same day-slice. Therefore, “20 out of 20 improved” cannot establish that the adapter “never overfit.” It establishes only that the adapter optimized its training error.

Relevant locations:

- [`paper-jcf/sections/adaptive_recalibration.tex`, line 9](paper-jcf/sections/adaptive_recalibration.tex#L9)
- [`code/examples/lora_sweep.jl`, line 183](code/examples/lora_sweep.jl#L183)
- [`code/examples/lora_sweep.jl`, line 233](code/examples/lora_sweep.jl#L233)

The trigger evaluation is also nearly tautological: its input is today's full-slice base RMSE, while “ground truth” is that same RMSE exceeding 12%.

- [`code/examples/lora_trigger_roc.jl`, line 143](code/examples/lora_trigger_roc.jl#L143)
- [`code/examples/lora_trigger_roc.jl`, line 217](code/examples/lora_trigger_roc.jl#L217)

This is a thresholding policy, not a predictive ROC experiment. A proper evaluation should hold out strikes within a day, use the next capture date, or predict the need for refitting before consuming the full current surface.

### 7. Caches can silently serve stale results

Cache validation checks strikes and three variance parameters but does not check the neural-model/cache hash, `theta`, horizon, path count, seed, risk-free rate, dividend yield, or LR depth.

Relevant location:

- [`code/src/ScenarioTemplate.jl`, line 469](code/src/ScenarioTemplate.jl#L469)

A changed model or scenario configuration can consequently reuse incompatible simulations while appearing reproducible.

## Important statistical and numerical issues

### Temporal earnings attribution is confounded

Configuration B removes most observations and changes the ticker, event, and contract composition. Its smaller gap does not establish that earnings explain the “entire” generalization gap.

- [`paper-jcf/sections/calibration.tex`, line 50](paper-jcf/sections/calibration.tex#L50)

The pooled model should first be evaluated on the exact non-event rows used by Configuration B. Event and non-event errors can then be compared on matched ticker, maturity, and moneyness strata.

### The direct-network comparison is under-supported

The claimed structural advantage is based on a single seed and a `0.38%`-IV difference between nonidentical architectures. Repeated fits, dispersion across seeds, and a genuinely parameter-matched baseline are needed before attributing the difference to the level/shape factorization.

### The neural level/shape decomposition is not identified

For the neural models, `log(theta_i) + log(psi)` is invariant to adding a constant to all ticker levels and subtracting it from the network output bias. Absolute `theta_i` levels are therefore not identified without a normalization such as mean `log(psi) = 0` over a reference grid or `psi(ATM, reference DTE) = 1`.

This weakens the interpretation of the reported absolute “baseline IV levels,” although relative predictions can still be valid.

### The static-arbitrage diagnostic may count lattice noise as arbitrage

The audit uses a 200-step CRR European tree before taking numerical second differences. CRR strike-node oscillation can itself create negative second differences.

- [`code/examples/static_arbitrage_check.jl`, line 138](code/examples/static_arbitrage_check.jl#L138)

Repeat the diagnostic using analytic Black–Scholes–Merton prices and report convergence across grid density and tolerances. Violations should use a scale-aware numerical tolerance rather than a strict `< 0` test.

### The SABR comparison is not truly per-expiry and contains a formula bug

The SABR script:

- combines expiries into seven-day maturity buckets;
- uses one mean maturity for rows having different DTEs;
- uses spot as the forward;
- assumes `q = 0`;
- evaluates on the same rows used to fit each slice.

Relevant location:

- [`code/examples/calibrate_sabr_per_day.jl`, line 83](code/examples/calibrate_sabr_per_day.jl#L83)

The exact-ATM SABR expression also contains an extra time contribution: `term` begins at `1.0` and is then inserted into `1 + term*T`.

- [`code/src/SABR.jl`, line 28](code/src/SABR.jl#L28)

The reported `9.05%` comparison should be regenerated using exact expiries, a proper forward, corrected ATM formula, and explicit convergence/failure reporting.

### Core and experiment time conventions disagree

The package's generic pricer converts DTE using 252 days, whereas the scenario and static-arbitrage scripts use 365:

- [`code/src/CRRTree.jl`, line 113](code/src/CRRTree.jl#L113)

The API should distinguish calendar DTE from trading-step count rather than representing both as an unqualified integer.

### The CRR complexity claim is false

The manuscript calls pricing cost linear in the number of tree steps, but the implemented backward induction contains nested loops and is `O(N^2)` per option.

- [`paper-jcf/sections/background.tex`, line 21](paper-jcf/sections/background.tex#L21)
- [`code/src/CRRTree.jl`, line 52](code/src/CRRTree.jl#L52)

### Monte Carlo uncertainty is not reported

The headline scenario uses one seed and 1,000 paths. Premium-kept rates around 70% have Monte Carlo standard errors around 1–1.5 percentage points even before model uncertainty. The “worst-case” values are unstable order statistics, so ratios of worst losses across tickers have little inferential meaning.

Use repeated seeds or a much larger simulation, confidence intervals for retention and mean P&L, and stable tail measures such as expected shortfall or fixed quantiles.

### Dividends and early assignment are ignored

The scenarios use `q = 0` for dividend-paying equities and ETFs. The low-volatility LR fallback also returns a European deterministic value rather than an American value.

- [`code/src/ScenarioTemplate.jl`, line 322](code/src/ScenarioTemplate.jl#L322)

Terminal P&L assumes every short position survives to expiry, omitting early assignment, transaction costs, margin, and discrete dividends. These omissions should either be modeled or clearly bound the claims.

### The leverage innovation is not a calibrated Brownian innovation

The scenario engine recovers `Z_S` by dividing the JumpHMM log return by `sqrt(theta_bar)` and a `1/365` scale, even though JumpHMM observations use their own annualized-return convention and trading-day time step. `Z_S` is Student-t/jump distributed and need not have unit variance. Consequently,

`Z_v = rho*Z_S + sqrt(1-rho^2)*Z_independent`

does not generally have unit variance or correlation exactly equal to `rho`.

- [`code/src/ScenarioTemplate.jl`, line 277](code/src/ScenarioTemplate.jl#L277)

The paper should either describe this as an empirical leverage shock or transform the return innovation to a calibrated zero-mean, unit-variance variable.

## Core-package implementation problems

### The package does not implement the paper's full model

The core package uses four analytic beta terms and has no earnings inputs, while the paper specifies five terms plus earnings features.

- [`code/src/ThetaFunction.jl`, line 19](code/src/ThetaFunction.jl#L19)
- [`paper-jcf/sections/methods.tex`, line 32](paper-jcf/sections/methods.tex#L32)

Most paper functionality instead resides in large one-off scripts and `ScenarioTemplate.jl`. The claim that the work is released as a reusable package is therefore stronger than the current package boundary supports.

### Theta initialization does not compute the documented mean

The calibration code says it initializes each state using mean squared IV, but the loop overwrites the state value for every observation, leaving the last observation's IV squared.

- [`code/src/Calibration.jl`, line 128](code/src/Calibration.jl#L128)

### State alignment appears one step ahead

Because return states are one element shorter than the price series, an option observation at price index `t` normally corresponds to the return ending at `t`, i.e. state index `t-1`. The preparation code uses `state_sequence[t]`, capped at the end.

- [`code/src/Calibration.jl`, line 75](code/src/Calibration.jl#L75)

This needs an explicit convention and a unit test because it can leak the following day's return state into calibration.

### Mood semantics differ between the paper and package

The paper defines mood as the fraction of tickers in tail states. Single-asset calibration instead assigns a binary tail indicator for the current ticker.

- [`code/src/Calibration.jl`, line 79](code/src/Calibration.jl#L79)

### Constraint enforcement is incomplete

The manuscript specifies `gamma >= 0`, but the package optimizes `gamma` unconstrained. Values below `-1` can make the mood multiplier negative and force later numerical clipping rather than a valid positive variance target.

### Missing tests cover the highest-risk components

The current test entry point includes types, theta, variance, CRR, basic calibration, and corpus utilities:

- [`code/test/runtests.jl`](code/test/runtests.jl)

It does not test:

- `LRTree.jl`;
- `IVInversion.jl`;
- `SABR.jl`;
- `ScenarioTemplate.jl`;
- `Pipeline.jl` end-to-end maturity handling;
- cache invalidation;
- paper-table reproduction;
- calendar/trading-day alignment;
- identification or recovery of dynamic variance parameters.

## Manuscript inconsistencies

### Corpus coverage is described incorrectly

The sentence “The same 31 tickers appeared on every capture day” is false. The `options-04-14-2026` directory contains 23 tickers; the other frozen dates contain 31. The reported 457 cells are exactly `31*15 - 8`, reflecting the missing first-day tickers rather than cells removed solely by the `N_obs >= 10` filter.

### The LLY sweep description contradicts the table

The text says LLY 04-28 appears in the 20-cell sweep, but the displayed sweep table contains LLY 04-22 instead.

- [`paper-jcf/sections/adaptive_recalibration.tex`, line 11](paper-jcf/sections/adaptive_recalibration.tex#L11)
- [`paper-jcf/sections/supplement.tex`, line 313](paper-jcf/sections/supplement.tex#L313)

### “Near-zero” control perturbations are not near zero

Some GS control-cell adapter norms exceed those of problem cells. The paper's assertion that control perturbations are near zero, and the analogy to transformer-LoRA middle-layer behavior, are not supported by the displayed table. Frobenius norms from differently sized layers are also not directly comparable without normalization.

### The base-network parameter count is wrong

The two-input `2 -> 16 -> 16 -> 1` network contains 337 parameters including biases, not 369. A four-input network has 369 parameters. The adapter count of 135 is consistent.

### The moneyness explanation is mathematically wrong

The paper states that shrinking DTE moves standardized moneyness further into the wings. Standardized moneyness depends on `K/S`, not DTE. Shorter DTE moves the contract along the term-structure input; any change in wing location comes from `S_t`.

- [`paper-jcf/sections/scenarios.tex`, line 39](paper-jcf/sections/scenarios.tex#L39)

### The discussion is stale

The discussion says validation remains limited to GS and LLY and proposes an earnings-spanning scenario as future work, although the manuscript already reports five additional tickers and an INTC earnings-window scenario.

- [`paper-jcf/sections/discussion.tex`, line 9](paper-jcf/sections/discussion.tex#L9)

### Several financial interpretations are overstated

- Dollar P&L is described as a “return” without a capital or margin denominator.
- A positive simulated mean is described as earning a “fair return,” although the physical drift and variance parameters are imposed rather than calibrated jointly.
- Market delta is treated as an approximate physical expiry probability. Delta is a risk-neutral sensitivity, so agreement with the simulated physical frequency is not a strong model validation.
- The paper attributes call-tail behavior to a “symmetric Black–Scholes premium,” although the entry IV comes from the fitted skewed surface and the option is priced with an American lattice.

## Narrative arc and flow

The manuscript currently contains several competing papers:

1. a pooled IV-surface calibration paper;
2. a JumpHMM/state-dependent variance paper;
3. a short-premium scenario paper;
4. an earnings-feature paper;
5. a daily LoRA-adapter paper.

The calibration hierarchy is the strongest empirical thread. The claimed state/mood mechanism is the nominal conceptual center but is not active in the headline experiments. The LoRA material then arrives as another large contribution and pulls the manuscript away from the original forward-generation problem.

Recommended narrative structure:

1. Define the contribution precisely as a pooled dynamic IV-surface generator with contract-specific square-root variance factors, unless the implementation is redesigned to be a genuine shared Heston variance process.
2. Present data provenance, surface definition, identifiability constraint, and calibration hierarchy.
3. Lead the empirical section with prospective walk-forward validation, including the newer extended corpus if it is frozen and reproducible.
4. Calibrate the dynamic variance parameters and present explicit ablations: static surface, contract-specific dynamics, leverage on/off, state coupling on/off, and aggregate mood on/off.
5. Present corrected scenario results with uncertainty and a consistent calendar.
6. Move LoRA into an optional online-marking section or a separate paper. Evaluate it out of sample.
7. Move most ticker-specific scenario prose, large tables, and adapter diagnostics into the supplement.
8. Remove claims that earnings explain the entire gap or that the disabled state/mood mechanism has been empirically validated.

The abstract is especially dense and tries to carry all five stories. It should state one primary contribution, one main validation design, and two or three quantitative results that survive the reruns.

## Repository and reproducibility cruft

The workspace is approximately 2.1 GB and contains substantial duplication and generated material:

- both `paper-arxiv` and `paper-jcf` with copied figures and build products;
- checked-in PDFs, PNGs, caches, logs, auxiliary LaTeX files, and `_unused` figures;
- `.DS_Store` files;
- large raw and serialized datasets;
- multiple long experiment scripts that duplicate loaders, sector maps, filters, feature standardization, and model restoration;
- a two-line README that does not explain the actual research workflow;
- setup documentation referring to a nonexistent `paper/` directory.

Relevant locations:

- [`README.md`](README.md)
- [`SETUP.md`, line 66](SETUP.md#L66)

Recommended cleanup:

1. Establish one canonical data loader and one immutable corpus manifest containing file hashes and observation counts.
2. Move experiment configuration into versioned TOML/YAML files.
3. Generate manuscript tables directly from result files rather than copying values into LaTeX.
4. Store a provenance record with every cache: git commit, input hashes, model hash, configuration, package manifest, Julia version, and seed.
5. Separate source data, derived data, caches, paper-ready figures, and scratch artifacts.
6. Remove tracked build products and `.DS_Store`; decide whether large binary data belongs in Git LFS or a versioned archive.
7. Consolidate shared code from `examples/` into tested library modules.

## PDF and presentation quality

Visual inspection of the compiled 52-page PDF found no missing figures, but the supplement has weak float placement and readability:

- page 50 is mostly blank around two small tables;
- page 51 contains a small four-panel figure surrounded by excessive whitespace;
- page 52 compresses two multi-panel figures until labels are difficult to read;
- the LaTeX log reports two overfull boxes and multiple PDF-string/font warnings.

The supplement should use more deliberate float placement, larger figure panels, and fewer panels per page. The current density and page count reinforce the feeling that several papers have been combined.

## Recommended remediation order

1. **Choose the model identity.** Either rename/reframe it as a dynamic IV-surface model or redesign it around one shared underlying Heston variance state and genuine stochastic-volatility option pricing.
2. **Fix time handling.** Separate calendar DTE, trading-step count, and annualization conventions throughout the types and APIs.
3. **Calibrate or justify dynamic parameters.** Remove the nonidentifiable `kappa`/`sigma_v` calibration path and add a defensible time-series or option-dynamics calibration.
4. **Activate and ablate the claimed innovation.** Rerun with state and mood coupling enabled and compare against disabled controls.
5. **Harden scenario numerics.** Correct leverage normalization, variance boundary handling, dividends, American low-vol fallbacks, and cache validation.
6. **Rerun all scenario results.** Report uncertainty and stable tail metrics.
7. **Repair validation methodology.** Use matched event/non-event comparisons, repeated neural fits, proper out-of-sample LoRA evaluation, and analytic static-arbitrage diagnostics.
8. **Align the package with the paper.** Unify beta terms, earnings inputs, NN restoration, and scenario logic behind tested public APIs.
9. **Rewrite the manuscript around surviving evidence.** Update the abstract, discussion, conclusion, and stale cross-references.
10. **Clean and document the repository.** Add an end-to-end reproduction command and machine-generated manuscript tables.

## Overall recommendation

Treat the current version as a substantial internal draft rather than a polishing-stage submission. The next revision should first establish a coherent model definition and a consistent simulation clock. After those corrections, regenerate the forward scenarios and determine which empirical conclusions survive. Only then should the narrative and repository be streamlined around the validated contribution.
