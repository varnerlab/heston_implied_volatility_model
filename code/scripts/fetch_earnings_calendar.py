"""
Fetch earnings dates for the 31-ticker corpus via yfinance.

Output: code/data/earnings/earnings_calendar.csv
Columns: ticker, earnings_date, eps_estimate, eps_actual, surprise_pct

ETFs (IWM, QQQ, SPY) are skipped — no earnings.

Usage:
    python3 code/scripts/fetch_earnings_calendar.py [--years N]

Default history is 2 years. yfinance also returns ~4 upcoming dates
when available.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


SECTORS = {
    "AAPL": "Tech", "AMD": "Tech", "AVGO": "Tech", "GOOG": "Tech",
    "INTC": "Tech", "META": "Tech", "MSFT": "Tech", "MU": "Tech",
    "NVDA": "Tech", "QCOM": "Tech",
    "BAC": "Financials", "GS": "Financials", "JPM": "Financials",
    "WFC": "Financials",
    "CVX": "Energy", "OXY": "Energy", "XOM": "Energy",
    "ABBV": "Healthcare", "AMGN": "Healthcare", "BMY": "Healthcare",
    "JNJ": "Healthcare", "LLY": "Healthcare", "MRNA": "Healthcare",
    "PFE": "Healthcare", "UNH": "Healthcare",
    "TGT": "Retail", "UPS": "Retail", "WMT": "Retail",
    "IWM": "ETF", "QQQ": "ETF", "SPY": "ETF",
}

ETF_TICKERS = {t for t, s in SECTORS.items() if s == "ETF"}
EQUITY_TICKERS = sorted(t for t in SECTORS if t not in ETF_TICKERS)


def fetch_one(ticker: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return a DataFrame with columns ticker, earnings_date, eps_estimate,
    eps_actual, surprise_pct for one ticker, filtered to dates >= cutoff."""
    tk = yf.Ticker(ticker)
    df = tk.earnings_dates
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    date_col = "Earnings Date" if "Earnings Date" in df.columns else df.columns[0]
    df = df.rename(columns={
        date_col: "earnings_date",
        "EPS Estimate": "eps_estimate",
        "Reported EPS": "eps_actual",
        "Surprise(%)": "surprise_pct",
    })
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], utc=True).dt.tz_convert(None).dt.normalize()
    df = df[df["earnings_date"] >= cutoff]
    df["ticker"] = ticker
    cols = ["ticker", "earnings_date", "eps_estimate", "eps_actual", "surprise_pct"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols].sort_values("earnings_date").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2,
                    help="History window in years (default 2)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output CSV path (default code/data/earnings/earnings_calendar.csv)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_path = Path(args.out) if args.out else (
        repo_root / "code" / "data" / "earnings" / "earnings_calendar.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cutoff = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize() - pd.DateOffset(years=args.years)

    rows = []
    failed = []
    for i, ticker in enumerate(EQUITY_TICKERS, 1):
        try:
            df = fetch_one(ticker, cutoff)
            n = len(df)
            print(f"  [{i:2d}/{len(EQUITY_TICKERS)}] {ticker:5s}  {n:3d} prints")
            if n:
                rows.append(df)
        except Exception as e:
            print(f"  [{i:2d}/{len(EQUITY_TICKERS)}] {ticker:5s}  FAILED: {e}")
            failed.append(ticker)
        time.sleep(0.5)  # be polite to Yahoo

    if not rows:
        print("ERROR: no earnings data fetched", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
    combined["earnings_date"] = combined["earnings_date"].dt.strftime("%Y-%m-%d")
    combined.to_csv(out_path, index=False)

    print(f"\nWrote {len(combined)} rows for {combined['ticker'].nunique()} tickers")
    print(f"  -> {out_path}")
    print(f"  date range: {combined['earnings_date'].min()} .. {combined['earnings_date'].max()}")
    if failed:
        print(f"  failed tickers ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
