# Dynamic IV ablation: fixed analysis design

This design was written before inspecting the ablation outputs. It is a
sensitivity experiment on fitted scenarios, not a forecast or trading backtest.

- GS and LLY; the April 28 to May 29, 2026 contracts already used in the paper.
- Three evaluation seeds: 20260429, 20260430, 20260431; 1,000 paths per seed.
- An independent 10,000-path, 22-transition pilot (seed 20260905) supplies
  the drift shift and log-return mean/standard deviation. These constants stay
  fixed for all evaluation seeds and IV variants. The prior is 10% annual
  excess log growth (the saved marginal risk-free rate is zero).
- Same stock paths, strikes, entry premiums, fitted entry IV, clocks, rates,
  and pricing depths in every variant. Independent normal innovations are
  shared across contracts and stochastic variants.
- Five variants: frozen entry IV; direct fitted surface; mean reversion with
  sigma_v=0; stochastic factor with rho=0; full factor with rho=-0.6.
  Other parameters remain kappa=15, sigma_v=0.5, variance floor=0.005^2.
- Primary endpoint: early-liquidation P&L after 10 trading transitions.
  Secondary endpoints: 5, 15, and 20 transitions, plus expiry as an invariant.
  P&L is entry market mid minus model liquidation mark, per share, with no
  interest on premium, execution cost, dividend, or prior assignment.
- Report mean, standard deviation, 5% quantile and 5% expected shortfall;
  paired mean P&L difference and its Monte Carlo SE; mean and 95th percentile
  absolute mark differences. Reference is the direct-surface variant.
  Report each seed and pooled paths; intervals cover simulation error only,
  conditional on fitted components and the frozen pilot constants.
- Assert identical entry marks and terminal payoff vectors across all arms.
- Check 201 versus 401 LR steps on the first 20 paths of each seed, at the
  primary and 20-transition endpoints. Use an American CRR fallback for
  low-volatility/saturated LR states, with valid transition probabilities.
- Audit bounds, strike monotonicity, vertical-spread bounds and convexity
  using 11 fixed strikes from 0.8 to 1.2 times entry spot, on the first 50 paths
  of the first seed at transitions 0, 10, and 20. Check calls and puts at the
  same expiry with common innovations. Require violations to exceed $0.01
  per share; repeat at 401 steps to distinguish lattice effects. This checks
  necessary strike conditions, not calendar or dynamic no-arbitrage.
- Record variance-floor frequency and the fraction of current strike/spot
  coordinates outside the calibration moneyness range [0.8,1.2].
- Save configuration, source/input SHA-256 hashes, input corpus manifest,
  pilot constants, path-level marks, seed-specific and pooled CSV summaries,
  numerical and strike audits, and publication figures/tables.

The experiment can show which IV components change interim risk and by how
much. Superior predictive accuracy would require comparison against future
observed option quotes under a chronological training design.
