"""Report every frozen model and keep the two evaluation periods separate."""
from pathlib import Path
import json,tomllib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/results/small_stock_comparison'


def main():
    assert tomllib.loads((OUT/'run_manifest.toml').read_text())['completed']
    x=pd.read_csv(OUT/'scores.csv')
    identifiers=['period','ticker','origin','endpoint','horizon']
    assert not x.duplicated(identifiers+['method','replicate']).any()
    reference=None
    for _,group in x.groupby(['method','replicate']):
        cases=set(map(tuple,group[identifiers+['origin_spot','observed']].to_numpy()))
        if reference is None:reference=cases
        assert reference==cases
    settings=json.loads((OUT/'frozen_settings.json').read_text())
    if settings['selected_penalty']=='zero':
        a=x[x.method=='Adaptive volatility'].sort_values(identifiers+['replicate'])
        b=x[x.method=='Directional'].sort_values(identifiers+['replicate'])
        for name in ['predicted_mean','predicted_median','crps','covered90','width90']:
            assert np.array_equal(a[name].to_numpy(),b[name].to_numpy())
    seed=x.groupby(['period','ticker','horizon','method','replicate']).agg(
        n_dates=('origin','size'),median_mae=('median_absolute_error','mean'),
        mean_mae=('absolute_error','mean'),mse=('squared_error','mean'),
        crps=('crps','mean'),coverage=('covered90','mean'),width=('width90','mean'),
        bias=('error','mean'),median_mae_pct=('median_absolute_error_pct','mean'),
        crps_pct=('crps_pct','mean'),width_pct=('width_pct','mean')).reset_index()
    seed['rmse']=np.sqrt(seed.mse)
    seed.to_csv(OUT/'summary_by_replicate.csv',index=False)
    summary=seed.groupby(['period','ticker','horizon','method']).mean(numeric_only=True).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    dates=x.groupby(identifiers+['method'])[['crps','median_absolute_error','squared_error']].mean().reset_index()
    dates.to_csv(OUT/'scores_by_date.csv',index=False)
    rows=[]
    for (period,ticker,horizon),group in dates.groupby(['period','ticker','horizon']):
        pivot=group.pivot(index='origin',columns='method',values='crps')
        for method in ['Adaptive volatility','Directional']:
            difference=pivot[method]-pivot.JumpHMM
            rows.append(dict(period=period,ticker=ticker,horizon=horizon,method=method,
                mean_crps_change=difference.mean(),median_crps_change=difference.median(),
                dates_better=int(difference.lt(0).sum()),n_dates=len(difference)))
    pd.DataFrame(rows).to_csv(OUT/'paired_comparison.csv',index=False)
    for period in [2025,2026]:
        print('\nPERIOD',period)
        print(summary[(summary.period==period)&(summary.horizon==5)][['ticker','method','n_dates',
            'median_mae','rmse','crps','coverage','width','median_mae_pct']].round(3).to_string(index=False))
    print('\nONE SESSION')
    print(summary[summary.horizon==1][['period','ticker','method','n_dates','median_mae',
        'rmse','crps','coverage','width']].round(3).to_string(index=False))


if __name__=='__main__':main()
