# Data Needed to Improve Sector NN IV Model

Current model: sector-specific neural network psi with per-ticker theta_base.
Trained on 23 tickers, 18,250 observations from a single capture (2026-04-14).
Overall RMSE: 6.38% IV. Worst sectors: Healthcare (9.16%), Tech (9.51%).

## Priority 1: Daily ladder pulls for data-thin tickers (3-5 days)

Pull daily close ladders for the tickers below. Same format and naming
convention as the existing files (drop into code/data/ladders/). The loader
reads everything in the directory automatically.

3 additional days gets every ticker above 700 obs (enough for per-ticker NNs).
5 days gets everyone above 1,100 and enables temporal out-of-sample validation.

### Tickers to pull (current obs count):

**Under 300 obs (most urgent):**
- ABBV (232)
- JNJ (241)
- CVX (271)
- BAC (276)
- OXY (281)
- XOM (290)

**300-500 obs:**
- INTC (309)
- MRNA (310)
- WFC (335)
- UPS (360)
- AMD (370)
- WMT (406)
- JPM (416)
- TGT (428)

**500-750 obs (would benefit but less urgent):**
- NVDA (536)
- AAPL (540)
- MU (587)
- MSFT (722)

**Already sufficient (1000+ obs, pull if convenient):**
- LLY (958)
- GS (1047)
- IWM (1812)
- QQQ (3315)
- SPY (4208)

## Priority 2: More Healthcare tickers (4+)

Healthcare has the worst RMSE (9.16%) and widest intra-sector dispersion:
ABBV at 16% vs LLY at 5.5%. The sector mixes large-cap pharma (JNJ, ABBV)
with high-growth biotech (LLY, MRNA), which have fundamentally different
smile shapes. Adding PFE, UNH, BMY, AMGN (or similar) would either support
splitting into Pharma vs Biotech sub-sectors or give the NN enough variation
to learn the distinction.

## Priority 3: More Tech tickers (4+)

Tech at 9.51% mixes mega-cap stable names (AAPL, MSFT) with semiconductor
cyclicals (INTC, MU, AMD, NVDA). INTC alone (11.2% RMSE) is an outlier.
Adding GOOG, META, AVGO, QCOM would help separate these sub-groups and
improve the sector NN's ability to generalize.

## Priority 4: Denser short-DTE coverage

DTE 2-4 still has 10-11% RMSE even with sector NNs. The smile geometry
changes rapidly near expiration (gamma compression). Capturing 0DTE, 1DTE,
2DTE, 3DTE separately rather than lumping into target_dte=4 would give the
NNs more resolution in the hardest regime.

## Priority 5: Volume/open interest (nice to have)

Would allow weighting the loss by liquidity. A tight-spread ATM option is
far more reliable than a wide-spread deep OTM quote. Currently all
observations are weighted equally.

## Priority 6: Intraday snapshots (nice to have)

Two captures per day (open + close) would double the dataset and reveal
intraday smile dynamics without waiting for additional trading days.
