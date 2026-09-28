"""Scale of reported IV in the fifteen-snapshot calibration corpus.

Applies the calibration filters (positive bid, positive DTE, 0.01 < IV < 2.0,
0.8 <= K/S <= 1.2) and prints the quartiles of reported IV for interpreting the scale of the
in-sample RMSEs.

Run from the repository root:
    code/venv/bin/python code/scripts/summarize_corpus_iv.py
"""
from pathlib import Path

import pandas as pd

LADDER = Path(__file__).resolve().parents[1] / "data/ladder"


def main():
    frames = []
    for f in sorted(LADDER.glob("options-*/*_dte_ladder_*.csv")):
        d = pd.read_csv(f)
        iv = pd.to_numeric(d.implied_vol, errors="coerce")
        m = d.strike / d.und_close.iloc[0]
        keep = iv.gt(0.01) & iv.lt(2.0) & d.bid.gt(0) & m.between(0.8, 1.2) & d.actual_dte.gt(0)
        frames.append(pd.DataFrame({"iv": iv[keep], "m": m[keep]}))
    corpus = pd.concat(frames)
    assert len(corpus) == 234_549, len(corpus)
    q = 100 * corpus.iv.quantile([0.25, 0.5, 0.75])
    atm = 100 * corpus.iv[corpus.m.between(0.98, 1.02)].median()
    print(f"{len(corpus):,} observations")
    print(f"reported IV quartiles (%): {q[0.25]:.1f} / {q[0.5]:.1f} / {q[0.75]:.1f}")
    print(f"median near the money (0.98 <= K/S <= 1.02): {atm:.1f}%")


if __name__ == "__main__":
    main()
