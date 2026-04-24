# VIX close data

VIX daily close values covering the options-ladder capture days, used as a
volatility-regime signal alongside the per-ticker options pulls in
`data/options-MM-DD-YY/`.

## Source

FRED series `VIXCLS` (CBOE Volatility Index: VIX, daily close). Free, no auth,
authoritative. Alpaca's market-data API is equities/options only and does not
serve indices, so we pull from FRED instead.

Series page: https://fred.stlouisfed.org/series/VIXCLS

## How to pull

```
curl -sS -f \
  "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS&cosd=YYYY-MM-DD&coed=YYYY-MM-DD" \
  -o data/VIX-data/vix_close_<start>_<end>.csv
```

`cosd` = start date (inclusive), `coed` = end date (inclusive). Output is CSV:

```
observation_date,VIXCLS
2026-04-14,18.36
...
```

Non-trading days (weekends, holidays) are omitted by FRED — the file contains
business days only, matching the options-capture calendar.

## Lag caveat

`VIXCLS` publishes with roughly a one-business-day lag. A pull run on the
morning of day T typically contains closes through day T-1. If a capture day
is missing from the file, either re-pull the next business day or fetch that
single day from Yahoo (`^VIX`), which is usually available same-day.

## Yahoo fallback for same-day close

```
P1=$(date -j -f "%Y-%m-%d" "YYYY-MM-DD" "+%s")
P2=$(date -j -f "%Y-%m-%d" "YYYY-MM-DD+1" "+%s")
curl -sS -A "Mozilla/5.0" \
  "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?period1=$P1&period2=$P2&interval=1d"
```

Returns a JSON `chart.result[0]` with `timestamp[]` and
`indicators.quote[0].close[]`. Yahoo's close for VIX matches FRED's to two
decimals in practice, so appending a Yahoo row to a FRED file is safe.

## Current files

- `vix_close_2026-04-14_2026-04-22.csv` — closes for the 04-14…04-22 window.
  Rows 04-14…04-21 from FRED; 04-22 appended from Yahoo (FRED had not yet
  published 04-22 at pull time).
