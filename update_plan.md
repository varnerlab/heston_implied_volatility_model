# Plan: Integrate HybridSingleIndexModel into Pipeline

## Context

The JumpHMM.jl package now supports `HybridSingleIndexModel` (HybridSIM), a variance-corrected single-index model that decomposes multi-asset returns into a market factor (SPY) + idiosyncratic noise with copula dependence. We are switching from the pure StudentTCopula-based portfolio simulation to HybridSIM to get economically grounded cross-asset dynamics.

**Key design decisions:**
- SPY is the market ticker
- Market HMM states drive θ_{s_t} for all tickers (shared systematic regime)
- Mood is binary: M_t = 1 if SPY HMM state is in tail, 0 otherwise
- Price data: 2014-2024 from `code/data/equity/SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2`

## Known Constraints

**HybridSIM simulation result structure:**
- `result.results[ticker].paths[i].observations` — valid growth rates (float)
- `result.results[ticker].paths[i].states` — all zeros (not populated for SIM-generated paths)
- `result.results[ticker].paths[i].jumps` — all false
- SPY is NOT in `result.results` (only non-market tickers)
- Market model accessible via `portfolio.dependence.market_model`

**Synchronization approach:** Simulate market model separately to get SPY HMM states. Pair with portfolio simulation paths by index. States and returns are from independent random draws but are statistically consistent (both from the same calibrated models). This is a known approximation; perfect synchronization would require exposing the internal market path from HybridSIM's `sample_dependence`.

## File Changes

### 1. `code/src/Pipeline.jl` — Modify `run_scenario`

**Current flow (lines 47-109):**
1. `JumpHMM.simulate(portfolio, n_sim_steps; ...)` → sim_result
2. Extract target ticker paths: `sim_result.results[ticker]`
3. Get N_states, N_tail from `portfolio.marginals[ticker]`
4. For each path: extract `.states`, `.observations`, convert to prices
5. Mood: loop all tickers, count tail states, mood = fraction

**New flow:**
1. `JumpHMM.simulate(portfolio, n_sim_steps; ...)` → sim_result (ticker returns)
2. Detect if portfolio uses HybridSIM: check `hasproperty(portfolio, :dependence) && portfolio.dependence isa JumpHMM.HybridSingleIndexModel`
3. **If HybridSIM:**
   - Get market model: `market_model = portfolio.dependence.market_model`
   - Simulate market: `market_result = JumpHMM.simulate(market_model, n_sim_steps; n_paths, seed=seed !== nothing ? seed + 1 : nothing)`
   - Get N_states, N_tail from `market_model.partition.N`, `market_model.jump.N_tail`
   - Get rf, dt from `market_model.rf`, `market_model.dt`
   - For each path i: use `market_result.paths[i].states` for HMM states
   - Mood: binary from market state (`s <= N_tail || s > N_states - N_tail`)
4. **If legacy (StudentTCopula):**
   - Keep existing behavior unchanged
5. Extract target ticker observations from sim_result, convert to prices
6. Rest of pipeline unchanged (simulate_variance, CRR pricing)

**Key coupling:** `ThetaHybrid.θ_states` must have length = `market_model.partition.N` (SPY states, not per-ticker states). This is a user responsibility when constructing the ThetaHybrid.

**Functions affected:**
- `run_scenario` (lines 34-152): multi-asset, takes portfolio
- `run_single_asset_scenario` (lines 159-228): single-asset, takes JumpHiddenMarkovModel — keep as-is for backward compatibility

### 2. `code/examples/multi_asset_scenario.jl` — Rewrite for HybridSIM

**New flow:**
1. Load price data from JLD2 (need DataFrames)
2. Build price matrix: [SPY, NVDA, AMD, MU, INTC] close prices
3. Fit: `portfolio = fit(PortfolioModel, tickers, prices; dependence=HybridSingleIndexModel, market="SPY", N=50, rf=0.05)`
4. Tune: `portfolio = tune(portfolio, prices; n_paths=200)`
5. Set up HestonParameters (κ, σ_v)
6. Set up ThetaHybrid with θ_states of length = market N_states (50)
   - Map SPY HMM states to variance levels
   - Tail states get elevated θ
7. Define option contracts
8. Call `run_scenario(portfolio, heston_params, θ_func, contracts, "NVDA", S0; ...)`
9. Analyze results

### 3. `code/examples/single_ticker_scenario.jl` — Minor update

Keep using `run_single_asset_scenario` with a single JumpHiddenMarkovModel. No HybridSIM here. But update the data loading to use the JLD2 file instead of hardcoded prices, if desired.

### 4. `code/examples/calibrate_and_validate.jl` — No changes needed

The calibration fits β parameters to IV smile data (single-date snapshots). It doesn't use the portfolio simulation. The calibrated β values are independent of whether we use StudentTCopula or HybridSIM for simulation.

### 5. `code/Project.toml` — Add DataFrames dependency

Needed to read the JLD2 files which contain DataFrames.

### 6. `code/src/Calibration.jl` — No changes needed now

The `prepare_calibration_data` and `calibrate` functions are for future time-series calibration (stage 2). They take a single JumpHiddenMarkovModel. When stage 2 is implemented, they should accept the market model (SPY HMM) instead, but that's a separate task.

## Critical Checks — VERIFIED

1. **`portfolio.marginals[ticker]` works with HybridSIM.** PortfolioModel fields: `(:tickers, :marginals, :dependence, :tickers_map, :market_ticker)`. Both `.marginals` and `.dependence` are accessible.

2. **`prices_from_growth_rates` works on HybridSIM observations.** Confirmed.

3. **Path count alignment confirmed.** Both `simulate(portfolio, ...)` and `simulate(market_model, ...)` respect `n_paths`.

4. **Observation length mismatch:** HybridSIM returns n-1 observations for n-step simulation (first trimmed internally). Market simulation returns n. **Resolution: trim market paths to match** (drop first timestep of market states/observations).

## Final Design Decisions

- **Mood:** Binary from market state. M_t = 1 if s_t ≤ N_tail or s_t > N - N_tail, else 0.
- **θ_states auto-calibration:** Empirical variance per SPY HMM state. No smoothing. Decode SPY historical returns → states, compute mean(G²) per state.
- **Single-asset pipeline:** Unchanged. HybridSIM is multi-asset only.
- **Market ticker detection:** Check `hasproperty(portfolio, :market_ticker)` to distinguish HybridSIM from legacy portfolios.

## Verification

1. Run `multi_asset_scenario.jl` end-to-end — confirm it produces ScenarioResult with valid price_paths, iv_paths, mood_paths
2. Check that mood_paths are binary (0.0 or 1.0)
3. Check that θ_{s_t} correctly indexes into market states
4. Compare IV path statistics with pre-HybridSIM results (should be qualitatively similar)
5. Run `single_ticker_scenario.jl` to confirm backward compatibility
