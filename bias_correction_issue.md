# Bias Correction Issue: Monte Carlo Evaluation of the S2 Rebalancing Engine

**Date:** 2026-04-23
**Notebook affected:** `lectures/session-2/eCornell-AI-Finance-S2-Example-Core-MonteCarloEvaluation-May-2026.ipynb`
**Commit:** `6463280`

## TL;DR

The S2 hybrid-SIM Monte Carlo generator inflates the rebalancing engine's
absolute returns by ~22 percentage points per year of CCGR relative to what
the engine actually delivers in real markets. The inflation is structural to
the SIM family (iid Gaussian residuals plus daily rebalancing produces a free
"diversification return" of ~28pp/yr; Booth-Fama 1992) and is *not* fixable
by parameter drift, path filtering, or generator perturbations within the
SIM framework.

We applied a uniform location-shift correction that anchors the corrected
synthetic median CCGR to a long-run prior (`ENGINE_PRIOR_CCGR_PCT = 8.0%/yr`,
risk-free + ~3.5pp active-premium net of frictions). The correction preserves
per-path rank order, drawdown shape, and volatility envelope; only the
distribution's level shifts. Real-2025 (engine CCGR = 11.5%/yr) then lands
at ~P63 of the corrected ensemble, which is honest: 2025 was a good year,
not a typical year, and the corrected MC characterizes percentile / tail
structure around a credible "typical year" center.

## The Problem

### What we observed

Running the engine across the existing synthetic and real datasets produced
a glaring discrepancy:

| | Synthetic ensemble | Real 2025 |
|---|---|---|
| Market CCGR | ~10.8%/yr | 15.4%/yr (SPY) |
| Engine CCGR | **~30%/yr** | **11.5%/yr** |
| Engine vol | 12.5%/yr | 8.2%/yr |
| Engine Sharpe | **~2.0** | **0.86** |
| Engine max DD | 5–8% (P50–P95) | 3.27% |
| Engine / Market multiplier | 2.78× | 0.69× |

The synthetic engine made nearly 2.8× the market growth on synthetic paths,
while the real engine made 0.69× the market on real-2025. A long-only
20-stock portfolio with maximum β = 1.21 (AAPL) **cannot structurally
achieve 2.78× the market** even at full concentration. Something was
inflating synthetic returns by ~18-22 percentage points per year of CCGR.

### Why this mattered

The course audience is senior finance professionals (JPM, Fed Dallas, AQR,
Blackrock per the audience memory). Presenting a synthetic median engine
Sharpe of 2.0 as a forward prediction would have been instantly recognized
by anyone in that room as an artifact of simulation. Showing the comparison
without addressing the gap would have been worse: the audience would
silently mark the speaker down for not knowing about the rebalancing-bonus
literature. We needed both an honest correction *and* a defensible framing.

## Diagnostic Process: What We Tried and Ruled Out

### Hypothesis 1: Parameter drift between calibration and deployment

The first instinct was that the calibrated (α, β) point estimates from
2014–2024 had drifted by 2025, so the allocator was reading stale
parameters relative to the per-path truth. We built
`generate_drifted_hybrid_scenario` in `code/src/Compute.jl` with two
independent drift channels:

1. **Per-asset (α, β) drift**: per (path, ticker) draws of
   `αᵢ_true = α̂ᵢ + drift_scale · SE_α · z_α`, similar for β. Tunable scale.
2. **Per-path market-growth drift**: per-path multiplicative scaler on
   `G_market` so different paths have different overall market regimes.

**Result: drift does not close the gap.**

We tried:
- Drift at bootstrap-SE scale (drift_scale ∈ {0, 1, 2, 3, 5}): essentially
  zero effect on engine median Sharpe (2.02 → 2.03 across the sweep).
- Drift at cross-sectional-SD scale (much larger): still essentially zero
  effect (2.02 → 2.01 at 1.5× cross-section SD).
- Market-growth drift up to σ = 1.5 (sign-flips half the paths): tails
  widen but median barely shifts because Gaussian dispersion is symmetric.

**Why the SE-based drift fails:** β is estimated very tightly. `|β̂|/SE_β`
ranges from 36 to 168 across the universe; cross-sectional spread is 20×
the within-window SE. Even drift at 5× SE moves β by ~5%, invisible
relative to cross-sectional spread.

**Why the symmetric drift fails on the median:** Gaussian dispersion is
symmetric around the prior, so the median is preserved by construction.
To move the median we'd need a *systematic asymmetric* shift — which is
data peeking ("we just *know* 2025 will have lower returns") and not
defensible.

The drifted generator was retained for the S3 walk-forward narrative
("here's what happens when the structure shifts mid-deployment") but it's
not on the critical path for this notebook.

### Hypothesis 2: The HMM market generator's central tendency is too rosy

If the synthetic market is unrealistically bullish, the engine's apparent
performance would inherit that bullishness. **Diagnosed: not the case.**

| Source | Median market CCGR |
|---|---|
| Synthetic HMM ensemble | 10.83%/yr |
| Real SPY 2014–2024 | ~11–12%/yr |
| Real SPY 2025 (the test year) | 15.4%/yr |

The synthetic market has a slightly *lower* central tendency than the
calibration data and lower than real-2025. The synthetic market
generator is honest. The gap was not in the market layer.

### Hypothesis 3: The λ sentiment signal drives the inflation

The EMA-crossover sentiment signal might be procyclically tilting the
allocator toward high-β names in synthetic-bullish regimes, harvesting
extra growth that doesn't exist in real markets. **Diagnosed: not the
case.**

| Engine variant | Median CCGR |
|---|---|
| Engine, dynamic λ (current S2 default) | 29.83% |
| Engine, λ = 0 frozen (no sentiment) | 29.69% |

λ contributes +0.13pp out of 30%. Sentiment is not the source of the
inflation. (This is itself a useful finding: most of S2's "engine alpha"
narrative implicitly attributed value to the sentiment signal; the data
shows that on the median path, the signal contributes essentially
nothing.)

### Root cause: Daily rebalancing on iid SIM residuals

The decomposition that finally exposed the artifact:

| Strategy | Median CCGR | Δ vs prior row |
|---|---|---|
| Equal-weight buy-and-hold | 7.68% | — |
| Static Cobb-Douglas (λ=0, no rebalance) | 1.97% | −5.71pp |
| Engine (λ=0, daily rebalance) | 29.69% | **+27.73pp** |
| Engine (dynamic λ) | 29.83% | +0.13pp |

The +27.7pp jump from "static CD" to "engine with daily rebalancing on
the same allocation rule" is the entire artifact. **Daily rebalancing
extracts ~28 percentage points per year of free alpha from this generator.**
This is the *diversification return* (Booth & Fama 1992; Bouchey,
Gunthorp & Sutherland 2012; Markowitz 1976 noted the same effect):

In the SIM,
```
r_i,t = α_i + β_i · r_m,t + ε_i,t,    ε_i,t ~ iid Gaussian
```

The residuals are independent across days. Consider a two-asset portfolio
that rebalances daily. If asset A goes up today, the rebalance sells some
A at the high price. Tomorrow A is a coin flip (no memory). On half of
tomorrows it goes back down → we bought low. On the other half it goes
up → we already sold at the high. **Both branches have positive expected
P&L** because the next move has mean zero. The textbook formula gives
bonus ≈ ½(Σᵢ wᵢ σᵢ² − σ_p²), which for our concentration on high-σ
names and 20-asset diversification works out to single-digit pp/yr in
basic theory but compounds to the observed +27pp through:

- **Daily rebalancing frequency** (more harvest opportunities than weekly)
- **Concentration on high-σ names** (bonus scales with σ²)
- **The gm_t signal feeding back into γ** (allocator effectively times
  its own SIM-generated regime persistence)

Real markets exhibit *positive autocorrelation* in residuals at daily
horizons (ρ ≈ +0.05 to +0.10). After AAPL goes up today it's slightly
more likely to keep going up tomorrow than to mean-revert. Real
rebalancing therefore *partially pays for itself* instead of being free
money. The realized 2025 engine extracted +9.5pp above the SIM-static
baseline (the engine's actual allocation alpha), not +28pp (which is
mostly the SIM giving the rebalancer a free lunch).

## Why Other Fixes Don't Work

### Filtering paths by performance bound

Tempting: drop synthetic paths where the engine made >25%/yr CCGR (or
some threshold). The retained ensemble would have a lower median.

**Two problems:**

1. Whatever criterion we filter on becomes a target the audience will
   probe ("why 25 not 20?"). Defensible filters exist (long-run Sharpe
   priors, empirical manager-distribution bounds), but each is an
   argument we have to win on the slide.
2. Filtering doesn't change the structural problem. Even on retained
   paths, the SIM's iid residuals still hand the rebalancer a free
   premium; we'd just show a subset where the engine made "merely 22%"
   instead of 30%. The audience still sees engine 2× real.

### Inflating transaction costs

Currently 5 bps. To reduce engine CCGR by 22pp/yr through costs alone
would require ~70 bps per trade (well above realistic institutional
costs of 1–10 bps). Pretending costs are 70 bps is just transferring
the lie from "engine is great" to "costs are huge."

### Adding AR(1) to SIM residuals

Mathematically right: ε_i,t = ρ·ε_i,t-1 + η_i,t with ρ ≈ 0.05–0.10
matches empirical daily autocorrelation. **But:**

- The current generator's copula reordering step (`sorted_eps[copula_ranks]`)
  destroys all temporal structure by construction. AR(1) would have to
  go in *before* the reordering, which means redesigning the copula
  sampling for time-ordered blocks. Estimated 10–15 hours.
- Even after the fix, plug ρ = 0.075 into the diversification-bonus
  algebra and the artifact compresses from ~28pp to ~8–15pp. Better but
  still inflated. The remaining inflation comes from amplifiers we
  *can't* fix without restructuring (concentration, daily frequency,
  gm_t feedback).
- We can't *prove* we've fixed it; sample size on real data is 1.

### Block bootstrap from real residuals

Compute empirical residuals from 2014–2024 data; bootstrap 21-day
blocks across all assets jointly; stitch. Preserves both temporal and
cross-asset structure by construction. Cleanest theoretically.
**Replaces** the JumpHMM marginal pipeline, losing the parametric story
S1 builds out. ~6 hours. We did not pursue this for time reasons.

### Re-fit the HMM on a longer window (2007–2024)

Right answer in principle. Multi-day work, breaks downstream caches
across all sessions. Out of scope for the May 2026 deadline.

## Our Fix: Option A — Location-Shift Bias Correction

### The construction

```julia
function apply_bias_correction(r::MyBacktestResult, bias_pct_per_yr::Float64,
        Δt::Float64)::MyBacktestResult
    n_t, n_p = size(r.wealth_paths)
    drag = exp.(-bias_pct_per_yr / 100.0 .* (0:(n_t-1)) .* Δt)
    Wc = r.wealth_paths .* drag
    # ...recompute final_wealth, max_drawdowns, sharpe_ratios on Wc
end
```

For every wealth path, multiply the time series by `exp(-bias · t · Δt)`.
At `t = 0` no drag; at `t = T` (end of horizon) the path is rotated
downward by `exp(-bias · T)`. Equivalent in CCGR space to subtracting
`bias` from each path's annualized log return.

### Choosing the anchor

The corrected synthetic median CCGR is set to a **documented long-run
prior**, not the realized 2025 value. We use:

```
ENGINE_PRIOR_CCGR_PCT = 8.0   # %/yr
                            = risk-free (~4.5%/yr) + ~3.5pp active-premium net of frictions
```

Decomposition:
- Risk-free CC growth ≈ 4.5%/yr (current 5y Treasury basis)
- Long-run equity premium over rf: 4–6pp (Damodaran annual ERP estimates;
  Ibbotson SBBI long-run series)
- Net of fees / turnover / market impact for an active multi-asset
  strategy: drop ~1–2pp from the passive premium
- Active-premium estimate: ~3.5pp net

8%/yr is on the conservative side; 9%/yr would be more bullish. We chose
8 because it pulls real-2025 (11.5%) into a clear above-median good-year
landing region (~P63), matching the empirical fact that 2025 was an
above-typical year. Anchoring at 9 would have put real-2025 closer to
P55; anchoring at 7 would have put it closer to P75. 8 was the user's call.

### Why a documented prior, not realized 2025

Earlier draft anchored to `REAL_2025_CCGR = 11.54`. This put real-2025 at
the synthetic median by construction, which:

1. **Was circular.** A senior quant would ask "if you anchor to real-2025,
   what does the MC add over a single backtest?" and we'd have nothing
   sharp to say.
2. **Misrepresented 2025.** Real-2025 was empirically a *good* year (SPY
   +16.6%, engine DD only 3.27%); centering the distribution there says
   "this is what a typical year looks like," which it isn't.

A documented long-run prior is not circular (the prior is independent
of the realized year) and is honest about 2025 being above-typical.

### Why option A and not option C (multiplicative haircut)

We prototyped both. Option C was

```
W_corr[p, t] = W_static[p, t] · (W_engine[p, t] / W_static[p, t])^k
```

with `k` calibrated so the corrected median matches the anchor (around
k = 0.27 at the realized-2025 anchor; would be similar at the prior
anchor). Theoretically more sophisticated: it scales each path's
deviation from the static-CD baseline by a constant factor `k` (the
"fraction of SIM rebalancing alpha that's real").

**Empirically rejected because of two issues:**

| Metric | Option A (location shift) | Option C (haircut k=0.274) |
|---|---|---|
| Median CCGR | 11.54% (anchor) | 11.54% (anchor) |
| Median Sharpe | 0.56 | 0.97 |
| P95 Sharpe | 1.73 | 2.46 |
| Median DD | 8.0% | 9.0% |
| P95 DD | 13.5% | 19.0% |
| Per-path CCGR rank correlation w/ raw engine | **1.000** | 0.748 |

1. **C reorders paths.** Rank correlation 0.75 means the haircut
   substantially scrambles which paths are "good" or "bad." Concretely:
   a path where the engine made +50% (because it caught the SIM
   rebalancing alpha well) gets pulled toward the static-CD outcome on
   that path; a path where static-CD happened to do well organically
   gets *promoted*. The corrected ensemble stops being an evaluation of
   *the engine* and becomes ~"static-CD with engine flavor."
2. **C distorts the drawdown distribution.** Static CD has wider DDs
   than the engine (median DD 11.7% vs 5%); under C the corrected wealth
   inherits static-CD's drawdown character. The engine's actual
   risk-management discipline gets washed out.

Option A, by contrast:
- Per-path rank correlation = 1.0 (pure monotone transform; "the worst
  raw-engine path is still the worst corrected-engine path")
- DD distribution stays close to engine's character
- Median Sharpe of 0.56 is realistic for an active strategy at the
  median; P95 of 1.73 is "top-decile year" — credible

A loses theoretical sophistication (no per-path proportional story) but
gains: honest per-path engine evaluation, better-calibrated risk profile,
quant-credible Sharpe distribution, simpler to defend on a slide.

## Audience Framing

### What to say on the slide

```
"On simulated SIM paths the engine extracts ~28pp/yr of rebalancing
alpha, consistent with the diversification-return literature
(Booth-Fama 1992) on iid residual structures. In real markets — where
residual autocorrelation is positive at daily horizons — that
mechanism doesn't deliver this much. We anchor the corrected
synthetic median to a long-run prior (8.0%/yr, risk-free plus a
modest active premium net of frictions) so the ensemble characterizes
percentile and tail structure around a credible center rather than a
SIM-inflated point. Real-2025 lands as an above-median good-year draw
(~P63) inside the corrected cone. The MC's value is the percentile
and tail structure, plus relative strategy comparison; the absolute
Sharpe levels are not a return prediction."
```

### Why this lands well

- It demonstrates awareness of a known SIM artifact, which the audience
  will recognize. *Not* mentioning it would mark the speaker as someone
  who doesn't know their tools.
- It cites the literature, which signals serious engagement.
- It explicitly separates two questions: "where's the central tendency?"
  (anchored to documented prior, not realized year) and "what's the
  dispersion around that center?" (from the MC ensemble). This is the
  same structure as Black-Litterman: prior on the mean, posterior on the
  shape. Quant audiences recognize this framing.
- Real-2025 becomes an honest landing point inside the cone, not a
  calibration target. It tells us 2025 was above-typical without
  pretending it was a typical year.

## Implementation Details

### MC notebook structure (32 cells)

- **Cell 4 (constants):** `ENGINE_PRIOR_CCGR_PCT = 8.0` (anchor),
  `COST_BPS = 5.0`, `SCENARIO_SEED = 2026`, etc.
- **Cell 6 (helpers):** `backtest_engine_frozen`, `backtest_static_cd_frozen`,
  `apply_bias_correction`, `tail_metrics`, `cones`, `panel`. The
  bias-correction helper lives in the notebook (editorial choice), not
  in `Compute.jl` (library function).
- **Cell 10 (run cell):** loads/generates scenario, runs engine raw and
  static-CD baselines, computes `bias_pct_per_yr = median(eng_raw_CCGR) -
  ENGINE_PRIOR_CCGR_PCT`, applies correction so `result_eng` is
  corrected. `result_eng_raw` retained for the decomposition.
- **Cell 11 (NEW markdown):** decomposition table + SIM rebalancing-alpha
  framing + Booth-Fama citation + bias-correction methodology.
- **Cells 13–15:** scorecard table (corrected engine row), markdown intro
  to convergence check, convergence DataFrame.
- **Cell 22 (save cell):** dual-writes `eng_*` (raw, downstream-notebook
  compatibility for `TurnoverAttributionDiagnostics`,
  `RegimeAwareSentiment`, Maya intro) and `eng_corr_*` (corrected,
  opt-in) plus `bias_pct_per_yr`. Same scenario, same other strategies
  as before.
- **Cell 21 (real overlay):** real-2025 engine now passes
  `cost_bps = COST_BPS` (was running cost-free against cost-included
  synthetic — pre-existing bug). Print replaced with CCGR-percentile
  diagnostic (W/W₀ comparison was apples-to-oranges due to 166-vs-252
  active-day mismatch).

### Compute.jl additions

`generate_drifted_hybrid_scenario` (lines ~1289–1500). Two drift
channels (per-asset α/β, per-path market-growth). Default arguments
reproduce `generate_hybrid_scenario` bit-for-bit. **Not consumed by
this notebook** — kept for the S3 walk-forward narrative where parameter
drift is the correct framing (online estimation responding to actual
structural changes).

### Smoke-test validation (200 paths, smaller-than-production)

```
Engine raw median CCGR:  29.78%/yr
Bias correction drag:    21.78pp/yr
Corrected synthetic median CCGR: 8.0%/yr (matches anchor)

Real 2025 CCGR vs corrected synthetic ensemble:
  Engine     : real 11.54%/yr  vs synthetic median 8.0%/yr   → P63.5
  S1 Min-Var : real 14.95%/yr  vs synthetic median 8.71%/yr  → P71.0
```

Both real strategies land in P60–P75 range, matching the empirical fact
that 2025 was a good year. Distribution at the anchor:

- P5 CCGR ≈ −9% (bad year, plausible)
- P50 CCGR = 8% (typical year)
- P95 CCGR ≈ +25% (great year, plausible)

## What This Means for Future Work

### Session 3 (online learning)

S3's EWLS narrative becomes cleaner: the *cross-sectional structure*
genuinely does drift between calibration windows, and EWLS adapts to
that drift online. The MC's residual gap (after bias correction) that
real-2025 sits in tells us how much percentile-uncertainty an active
strategy should expect; S3 shows how to reduce that uncertainty by
tracking the structure live. The drifted generator
(`generate_drifted_hybrid_scenario`) is available for an S3 example
that explicitly demonstrates "deploy with stale calibration vs deploy
with EWLS-tracked calibration on the same drifted paths."

### General lesson for backtest evaluation in this codebase

Any future MC notebook that evaluates a *dynamically rebalanced*
strategy on the SIM-family generator should:

1. Always include a static (no-rebalance) baseline of the same
   allocation rule for the rebalancing-alpha decomposition.
2. Apply the bias correction (or equivalent) to the rebalanced
   strategy if the audience is going to see absolute return numbers.
3. Use a documented prior anchor, not the realized outcome.
4. Frame the absolute Sharpe levels as "this generator + this
   strategy" not "what the strategy will do in production."

The `feedback_sim_rebalancing_artifact.md` memory captures this for
future sessions.

## References

- Booth, D. G. & Fama, E. F. (1992). "Diversification Returns and Asset
  Contributions." *Financial Analysts Journal*, 48(3), 26–32.
- Bouchey, P., Gunthorp, D., & Sutherland, M. (2012). "Volatility
  Harvesting in Theory and Practice." *Journal of Wealth Management*,
  15(2), 89–100.
- Damodaran, A. — annual equity risk premium estimates (NYU Stern
  working papers, updated yearly).
- Markowitz, H. (1976). "Investment for the Long Run: New Evidence for
  an Old Rule." *Journal of Finance*, 31(5), 1273–1286.
- Lo, A. W. & MacKinlay, A. C. (1990). "When are Contrarian Profits
  Due to Stock Market Overreaction?" *Review of Financial Studies*,
  3(2), 175–205.

## Files Touched

- `code/src/Compute.jl` — added `generate_drifted_hybrid_scenario`
- `code/src/eCornellAIFinance.jl` — exported new function
- `lectures/session-2/eCornell-AI-Finance-S2-Example-Core-MonteCarloEvaluation-May-2026.ipynb` — new notebook
- Deleted: `eCornell-AI-Finance-S2-Example-StaticVsAdaptiveComparison-May-2026.ipynb`
- Deleted: `eCornell-AI-Finance-S2-Example-StressTestRebalancingEngine-May-2026.ipynb`

## Memory Files Created or Updated

- `memory/feedback_sim_rebalancing_artifact.md` (new)
- `memory/project_mc_bias_correction.md` (new)
- `memory/feedback_trigger_whipsaw.md` (updated: SIM-artifact context)
- `memory/project_session2_pipeline.md` (updated: MC consolidation status)
- `memory/MEMORY.md` (index entries added/retitled)
