# Setup

These steps get a fresh student checkout to a working state. Run from the
repo root (the directory containing this file).

## Prerequisites

- **Julia** 1.10+ — main calibration code. Install from <https://julialang.org/downloads/>.
- **Python** 3.10+ — only used for the earnings-calendar fetcher. macOS Homebrew Python, system Python on Linux, or any Anaconda install all work.
- **Git** — to clone the repo.

## Julia environment

The Julia code lives under `code/`. From the repo root:

```bash
cd code
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

That installs every Julia dependency listed in `code/Project.toml`.

To run the test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Earnings calendar (Python venv)

The Heston model's theta-function uses an earnings indicator to handle
pre-earnings IV expansion (see `code/src/EarningsCalendar.jl`). The
calendar is fetched from Yahoo Finance via `yfinance`. We use a project-
local virtual environment so the install is isolated and reproducible —
this avoids the PEP 668 "externally managed environment" error you'll
hit on macOS Homebrew Python.

**One-time setup** (from repo root):

```bash
python3 -m venv code/venv
code/venv/bin/pip install --upgrade pip
code/venv/bin/pip install yfinance pandas lxml
```

**Fetch (or refresh) the calendar:**

```bash
code/venv/bin/python code/scripts/fetch_earnings_calendar.py
```

This writes `code/data/earnings/earnings_calendar.csv` with columns
`ticker, earnings_date, eps_estimate, eps_actual, surprise_pct`.

Default history is 2 years. Override with `--years N` if you want more
or less. ETFs (IWM, QQQ, SPY) are skipped — no earnings.

Re-run any time you want fresh upcoming dates. Yahoo throttles by IP, so
if you hit failures wait a minute and re-run; the script retries one
ticker at a time and reports which ones failed.

The CSV is checked into git so day-to-day work doesn't require a fetch —
you only need to re-run when extending the corpus or when a new earnings
print has happened since the last fetch.

## Repo layout

```
code/
  src/                Julia source (HestonIV.jl is the entry module)
  examples/           runnable calibration scripts
  scripts/            utility scripts (Python earnings fetch lives here)
  data/
    ladder/           per-day option chain captures (CSV)
    ladder_excluded/  partial captures held out of the corpus
    earnings/         earnings_calendar.csv (fetched, see above)
    equity/           equity price data
  test/               Julia test suite
paper-arxiv/          LaTeX sources and figures, arXiv variant
paper-jcf/            LaTeX sources and figures, JCF submission variant
```

## Common problems

- **`pip install` errors with "externally-managed-environment"**: you
  skipped the venv step. Use `code/venv/bin/pip` not the system `pip3`.
- **`Import lxml failed`**: install `lxml` into the venv —
  `code/venv/bin/pip install lxml`. yfinance needs it but doesn't pull
  it in automatically.
- **yfinance returns empty for a ticker**: Yahoo throttling. Wait, retry.
  If a ticker consistently fails, check `yf.Ticker("XXX").earnings_dates`
  in a Python REPL to see the underlying error.
