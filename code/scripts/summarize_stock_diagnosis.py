"""Summarize every fixed stock diagnostic candidate and simulation replicate."""
from pathlib import Path
import os
import tomllib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('SIMULATION_RESULTS_ROOT',ROOT/'code/results'))/'stock_forecast_diagnosis'


def main():
    assert tomllib.loads((OUT/'manifest.toml').read_text())['completed']
    x=pd.read_csv(OUT/'scores.csv')
    keys=['ticker','origin','endpoint','horizon','method','replicate']
    assert not x.duplicated(keys).any()
    reference=None
    for _,group in x.groupby(['method','replicate']):
        cases=set(map(tuple,group[['ticker','origin','endpoint','horizon','observed']].to_numpy()))
        if reference is None:reference=cases
        assert cases==reference
    per_seed=x.groupby(['ticker','horizon','method','replicate']).agg(
        n_dates=('origin','size'),mean_mae=('absolute_error','mean'),
        median_mae=('median_absolute_error','mean'),mse=('squared_error','mean'),
        crps=('crps','mean'),coverage=('covered90','mean'),width=('width90','mean'),
        bias=('error','mean'),mae_pct=('mean_error_pct',lambda a:a.abs().mean()),
        crps_pct=('crps_pct','mean')).reset_index()
    per_seed['rmse']=np.sqrt(per_seed.mse)
    per_seed.to_csv(OUT/'summary_by_replicate.csv',index=False)
    averaged=per_seed.groupby(['ticker','horizon','method']).mean(numeric_only=True).reset_index()
    averaged.to_csv(OUT/'summary.csv',index=False)
    by_origin=x.groupby(['ticker','horizon','method','origin']).mean(numeric_only=True).reset_index()
    by_origin.to_csv(OUT/'scores_by_origin.csv',index=False)
    pairs=by_origin.pivot(index=['ticker','horizon','origin'],columns='method',values='crps')
    paired=[]
    for (ticker,horizon),group in pairs.groupby(level=['ticker','horizon']):
        for method in group.columns:
            change=group[method]-group['legacy_stationary']
            paired.append(dict(ticker=ticker,horizon=horizon,method=method,
                mean_crps_change=change.mean(),median_crps_change=change.median(),
                better_dates=int(change.lt(0).sum()),n_dates=len(change)))
    pd.DataFrame(paired).to_csv(OUT/'paired_crps.csv',index=False)
    print(averaged[averaged.horizon.eq(5)][['ticker','method','n_dates','median_mae',
        'rmse','crps','coverage','width']].round(3).to_string(index=False))


if __name__=='__main__':main()
