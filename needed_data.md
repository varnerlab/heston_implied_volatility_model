# Data Needed for Per-Ticker NN IV Models

Current model: sector-specific neural network psi with per-ticker theta_base.
Corpus spans five capture days (2026-04-14, 04-15, 04-16, 04-17, 04-20), but
**04-20 only covered 23 of the 31 tickers**, so the effective full-coverage
corpus remains the 4-day, 71,586-observation set from 04-14 through 04-17.
Sector NN: 6.91% in-sample RMSE, 7.54% temporal-holdout RMSE. Persistent
worst names on holdout: PFE 18.41%, BMY 15.29%, INTC 13.92%, JNJ 12.98%,
MRNA 12.65%.

## Priority 1: EVERY PULL MUST COVER ALL 31 TICKERS

The 2026-04-20 pull captured only 23 tickers. The 8 missing names are:

- **Tech (4):** AVGO, GOOG, META, QCOM
- **Healthcare (4):** AMGN, BMY, PFE, UNH

This is a regression. The 04-16 and 04-17 pulls both covered all 31, so the
ladder fetcher is capable of full coverage and something changed on 04-20.
**PFE and BMY are the two most data-starved tickers in the whole corpus**
(326 and 455 observations across the prior 4 days) and they are also the
two worst performers on the temporal holdout. Dropping them on a new
capture day is the exact opposite of what the per-ticker NN roadmap needs.

Action items:

1. Investigate why 04-20 dropped those 8 tickers (fetcher error? symbol
   list? rate limit? data-provider outage on those symbols?)
2. Re-pull 04-20 for the missing 8 if the close-of-day snapshot is still
   recoverable, otherwise treat 04-20 as a partial day and flag it in the
   loader
3. Add a preflight check to the ladder-pull script that fails loudly if any
   of the 31 tickers is absent from the output directory

The canonical 31-ticker list (do not deviate):

- Tech (10): AAPL, AMD, AVGO, GOOG, INTC, META, MSFT, MU, NVDA, QCOM
- Healthcare (8): ABBV, AMGN, BMY, JNJ, LLY, MRNA, PFE, UNH
- Financials (4): BAC, GS, JPM, WFC
- Energy (3): CVX, OXY, XOM
- Retail (3): TGT, UPS, WMT
- ETF (3): IWM, QQQ, SPY

## Priority 2: Keep collecting full-coverage days until the Healthcare tail clears ~2000 obs each

Per-ticker 2->8->8->1 NN has ~105 parameters. Rule of thumb: ~20 obs per
parameter before overfitting becomes tolerable, so target ~2000 obs per
ticker. Current starved-tail obs counts (through 04-17):

- PFE 326, BMY 455, JNJ 745, MRNA 954, INTC 978

At the observed capture rate these tickers need roughly another 8-12
full-coverage capture days before per-ticker NNs are viable on the names
that actually need them. ETFs (SPY 14k, QQQ 12k, IWM 6.5k) already have
enough, but fitting per-ticker on the ETFs alone does not move the paper
narrative; the Healthcare/Tech tail is where per-ticker has to win.

## Priority 3: Denser short-DTE coverage

DTE 1-4 remains the hardest regime (7-12% RMSE). Capturing 0DTE, 1DTE,
2DTE, 3DTE separately rather than lumping into `target_dte=4` would give
the NNs more resolution where error concentrates.

## Priority 4: Tech sub-sector split (mega-cap vs semi)

Tech is still the second-worst sector on holdout (9.00%). Ten Tech tickers
is enough to split:

- Mega-cap stable: AAPL, MSFT, GOOG, META
- Semiconductor cyclicals: NVDA, AMD, INTC, MU, AVGO, QCOM

Two separate Tech NNs could reduce RMSE before per-ticker becomes viable.

## Priority 5: Healthcare sub-sector split (pharma vs biotech)

Healthcare is the worst sector on holdout (10.23%). Eight tickers enable:

- Large-cap pharma: JNJ, ABBV, PFE, BMY, AMGN, UNH
- High-growth biotech / specialty: LLY, MRNA

## Priority 6: Volume / open interest, intraday snapshots

Liquidity-weighted loss and intraday (open + close) captures remain
nice-to-haves once the per-ticker NN baseline lands.
