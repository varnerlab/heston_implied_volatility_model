"""Bootstrap standard errors for the coupling shift in ten-day expected shortfall.

The IV ablation reports the 5% expected shortfall (ES) of short-option P&L under
the uncoupled factor (rho = 0) and the full factor (rho = -0.6) on the same
3,000 stock paths. This script resamples paths, keeping each path's two P&L
values together and resampling within each evaluation seed, and reports the
standard error and percentile interval of the paired ES shift.

Run from the repository root:
    code/venv/bin/python code/scripts/bootstrap_ablation_es.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "code/results/dynamic_ablation"
STEP, ALPHA, N_BOOT, SEED = 10, 0.05, 5000, 20260928


def es(pnl):
    # Same definition as DynamicAblation.summarize: mean of P&L at or below the
    # type-7 5% quantile (Julia's and NumPy's default)
    q = np.quantile(pnl, ALPHA, axis=-1, keepdims=True)
    return np.nanmean(np.where(pnl <= q, pnl, np.nan), axis=-1)


def main():
    marks = pd.read_csv(RESULTS / "path_marks.csv")
    marks = marks[marks.step.eq(STEP) & marks["mode"].isin(["uncoupled", "coupled"])]
    marks = marks.assign(pnl=marks.premium - marks.mark)
    summary = pd.read_csv(RESULTS / "summary.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for (ticker, kind), group in marks.groupby(["ticker", "kind"]):
        wide = group.pivot(index=["seed", "path"], columns="mode", values="pnl").sort_index()
        seeds = wide.index.get_level_values("seed").to_numpy()
        pnl_u, pnl_c = wide["uncoupled"].to_numpy(), wide["coupled"].to_numpy()
        es_u, es_c = es(pnl_u), es(pnl_c)
        # Pooled ES must match the value the manuscript table was built from
        for mode, value in [("uncoupled", es_u), ("coupled", es_c)]:
            ref = summary[summary.ticker.eq(ticker) & summary.kind.eq(kind) &
                          summary["mode"].eq(mode) & summary.step.eq(STEP) & summary.seed.eq(0)]
            assert np.isclose(ref.es05_pnl.iloc[0], value), (ticker, kind, mode)
        # Stratified paired bootstrap: resample paths within each seed
        idx = np.concatenate([
            rng.choice(np.flatnonzero(seeds == s), size=(N_BOOT, (seeds == s).sum()))
            for s in np.unique(seeds)], axis=1)
        shift = es(pnl_c[idx]) - es(pnl_u[idx])
        per_seed = [es(pnl_c[seeds == s]) - es(pnl_u[seeds == s]) for s in np.unique(seeds)]
        rows.append(dict(ticker=ticker, kind=kind, es_uncoupled=es_u, es_coupled=es_c,
                         shift=es_c - es_u, se_shift=shift.std(ddof=1),
                         lo95=np.quantile(shift, 0.025), hi95=np.quantile(shift, 0.975),
                         se_es_coupled=es(pnl_c[idx]).std(ddof=1),
                         seed_shift_min=min(per_seed), seed_shift_max=max(per_seed)))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "es_coupling_bootstrap.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
