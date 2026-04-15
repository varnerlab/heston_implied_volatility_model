# Data Needed for Per-Ticker NN IV Models

Current model: sector-specific neural network psi with per-ticker theta_base.
Trained on 31 tickers, 35,961 observations from two captures (2026-04-14,
2026-04-15). Overall RMSE: 6.76% IV. Worst sectors: Tech (9.16%),
Healthcare (8.83%).

The 2026-04-15 pull executed most of the Priority 1-3 items from the
previous version of this file (added AMGN, AVGO, BMY, GOOG, META, PFE, QCOM,
UNH and doubled observations for the original 23). Next focus: enable
**per-ticker NNs** and **out-of-sample validation**.

## Priority 1: One more full-coverage ladder pull (all 31 tickers)

Pull daily close ladders for all 31 current tickers. Drop into
`code/data/ladder/options-MM-DD-YYYY/`; the loader now recurses into any
day-folder under `code/data/ladder/`.

One more day gets ~24 of 31 tickers above 1000 obs (comfortable for a
per-ticker 2->8->8->1 NN, ~89 params) and enables a clean temporal holdout
(train on two days, test on the third). This is the single highest-leverage
pull remaining.

### Current obs counts (04-14 + 04-15 pooled):

**Tickers with only 04-15 data — need a second day most (the 8 additions):**
- PFE (101)  -- still too thin, see Priority 2
- BMY (~160) -- still too thin, see Priority 2
- QCOM (299)
- UNH (304)
- AMGN (~380)
- GOOG (~400)
- AVGO (~780)
- META (~1500)

**Already have 2 days, another pull would break 1000:**
- ABBV (~505), JNJ (~529), CVX (~493), BAC (~363), OXY (~459), XOM (~498)
- INTC (~637), MRNA (~529), WFC (~565), UPS (~608)
- AMD (~897), WMT (~676), JPM (~733), TGT (~705)
- NVDA (~903), AAPL (~1177), MU (~1721), MSFT (~1301)

**Already sufficient for per-ticker NN:**
- LLY (~1313), GS (~1549), IWM (~2353), QQQ (~6386), SPY (~7748)

## Priority 2: Second day for PFE and BMY, or drop to sector-pooled only

PFE (101 obs) and BMY (~160 obs) are too thin for per-ticker NNs even after
several additional pulls. Two options:
1. Pull PFE/BMY specifically for 3-5 more days to reach ~400+ obs
2. Accept they stay sector-pooled (Healthcare sector NN) and skip per-ticker
   models for them

Recommended: option 2 unless they become high-priority underlyings for the
paper. PFE currently has the worst per-ticker RMSE (13.33%) — its thin
sample is also the noisiest.

## Priority 3: Third day enables temporal out-of-sample validation

With three capture days, hold out one day entirely and train on the other
two. This is the cleanest test of whether the NN generalizes across time
(as opposed to just interpolating within a single snapshot). Essential
before any paper-quality claim about model accuracy.

## Priority 4: Split Tech sub-sectors (mega-cap vs semi)

Tech remains the worst sector (9.16%). With 10 Tech tickers now, we can
split into:
- Mega-cap stable: AAPL, MSFT, GOOG, META
- Semiconductor cyclicals: NVDA, AMD, INTC, MU, AVGO, QCOM

Two separate Tech NNs instead of one shared should reduce RMSE meaningfully.
Worth trying before collecting more Tech data.

## Priority 5: Split Healthcare sub-sectors (pharma vs biotech)

Similar story — 8 Healthcare tickers now enable:
- Large-cap pharma: JNJ, ABBV, PFE, BMY, AMGN, UNH
- High-growth biotech: LLY, MRNA

## Priority 6: Denser short-DTE coverage (still open)

DTE 1-4 still has 7-12% RMSE. Capturing 0DTE, 1DTE, 2DTE, 3DTE separately
rather than lumping into target_dte=4 would give the NNs more resolution in
the hardest regime.

## Priority 7: Volume/open interest, intraday snapshots (nice to have)

Unchanged from previous list. Liquidity-weighted loss and intraday
(open + close) captures would each be valuable once the per-ticker NN
baseline is in place.
