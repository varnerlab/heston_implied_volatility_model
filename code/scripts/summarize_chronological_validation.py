"""Aggregate the fixed forecast experiment and write reviewable manuscript tables."""
from pathlib import Path
import json
import tomllib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/results/chronological_validation'
GEN=ROOT/'paper-arxiv/sections/generated'
LABELS={'frozen':'Frozen IV','surface':'Direct surface','relaxation':'Mean reversion',
        'uncoupled':'Uncoupled factor','coupled':'Coupled factor'}
MODES=list(LABELS)


def write_table(name,header,rows,align):
    text=[r'\begin{tabular}{'+align+'}',r'\toprule',header+r' \\',r'\midrule']
    text.extend(' & '.join(map(str,row))+r' \\' for row in rows)
    text.append(r'\bottomrule\end{tabular}')
    (GEN/name).write_text('\n'.join(text)+'\n')


def aggregate(scores,groups):
    metrics=['error','absolute_error','squared_error','crps','covered90','width90','inside_spread','mc_se']
    dates=scores.groupby(groups+['origin'],as_index=False)[metrics].mean()
    pooled=dates.groupby(groups,as_index=False)[metrics].mean()
    pooled['rmse']=np.sqrt(pooled.squared_error)
    counts=scores.groupby(groups).agg(n_quotes=('symbol','size'),n_dates=('origin','nunique')).reset_index()
    return pooled.merge(counts,on=groups),dates


def main():
    assert tomllib.loads((OUT/'run_manifest.toml').read_text())['completed']
    parts=[]; audits=[]
    for cohort,directory in [('monthly',OUT),('short',OUT/'short_maturity')]:
        assert tomllib.loads((directory/'run_manifest.toml').read_text())['completed']
        part=pd.read_csv(directory/'forecast_scores.csv');part['cohort']=cohort;parts.append(part)
        audit=pd.read_csv(directory/'endpoint_availability.csv');audit['cohort']=cohort;audits.append(audit)
    f=pd.concat(parts,ignore_index=True)
    a=pd.concat(audits,ignore_index=True)
    keys=['cohort','ticker','origin','endpoint','horizon','symbol','mode']
    conditional_keys=f[f.conditioning.eq('observed_stock')][keys].drop_duplicates()
    joint_common=f[f.conditioning.eq('joint')].merge(conditional_keys,on=keys,how='inner')
    common=pd.concat([joint_common,f[f.conditioning.eq('observed_stock')]],ignore_index=True)
    common_scores,_=aggregate(common,['cohort','conditioning','ticker','horizon','mode'])
    common_scores.to_csv(OUT/'matched_conditioning_summary.csv',index=False)
    scores,dates=aggregate(f,['cohort','conditioning','ticker','horizon','mode'])
    scores.to_csv(OUT/'forecast_summary.csv',index=False)
    dates.to_csv(OUT/'forecast_scores_by_origin.csv',index=False)
    availability=a.groupby(['cohort','ticker','horizon']).agg(selected=('symbol','size'),
        matched=('quote_matched','sum'),origins=('origin','nunique'),
        complete_stock_path=('complete_stock_path','sum')).reset_index()
    availability.to_csv(OUT/'availability_summary.csv',index=False)
    pairs=dates.pivot(index=['cohort','conditioning','ticker','horizon','origin'],columns='mode',values='absolute_error')
    paired=[]
    for keys,g in pairs.groupby(level=['cohort','conditioning','ticker','horizon']):
        if not {'coupled','frozen','surface'}<=set(g.columns):continue
        for baseline in ['frozen','surface']:
            delta=(g.coupled-g[baseline]).dropna()
            if delta.empty:continue
            paired.append(dict(cohort=keys[0],conditioning=keys[1],ticker=keys[2],horizon=keys[3],baseline=baseline,
                mean_mae_change=delta.mean(),median_mae_change=delta.median(),
                min_mae_change=delta.min(),max_mae_change=delta.max(),
                coupled_better_dates=int(delta.lt(0).sum()),n_dates=len(delta)))
    pd.DataFrame(paired).to_csv(OUT/'paired_skill_by_date.csv',index=False)
    for conditioning,horizon,name in [('joint',5,'forecast_main_table.tex'),
        ('joint',1,'forecast_one_session_table.tex'),
        ('observed_stock',5,'forecast_conditional_table.tex')]:
        rows=[]
        for ticker in ['GS','LLY']:
            for mode in MODES:
                q=scores[scores.cohort.eq('monthly' if horizon==1 else 'short')&scores.conditioning.eq(conditioning)&scores.horizon.eq(horizon)&scores.ticker.eq(ticker)&scores['mode'].eq(mode)]
                if q.empty:continue
                r=q.iloc[0]
                rows.append([ticker,LABELS[mode],int(r.n_dates),int(r.n_quotes),f'{r.absolute_error:.2f}',
                    f'{r.crps:.2f}',f'{100*r.covered90:.1f}',f'{r.width90:.2f}'])
        write_table(name,r'Ticker & IV variant & Dates & Quotes & MAE & CRPS & Coverage (\%) & Width',rows,'llrrrrrr')
    rows=[]
    for r in availability.itertuples():
        rows.append([r.cohort,r.ticker,r.horizon,r.origins,r.selected,r.matched,r.selected-r.matched,r.complete_stock_path])
    write_table('forecast_availability_table.tex',r'Cohort & Ticker & Horizon & Origins & Selected & Matched & Missing & Full stock path',rows,'llrrrrrr')
    ss=pd.read_csv(OUT/'stock_scores.csv')
    stock=ss.groupby(['ticker','horizon','mode']).agg(n_dates=('origin','size'),
        mae=('absolute_error','mean'),mse=('squared_error','mean'),crps=('crps','mean'),
        coverage=('covered90','mean'),width=('width90','mean')).reset_index()
    stock.to_csv(OUT/'stock_summary.csv',index=False)
    rows=[[r.ticker,r.horizon,r.mode,r.n_dates,f'{r.mae:.2f}',f'{r.crps:.2f}',f'{100*r.coverage:.1f}'] for r in stock.itertuples()]
    write_table('forecast_stock_table.tex',r'Ticker & Horizon & Stock model & Dates & MAE & CRPS & Coverage (\%)',rows,'lrllrrr')
    surface=pd.read_csv(OUT/'surface_scores_by_date.csv')
    sector_map={}
    for sector,names in {'Tech':'AAPL AMD AVGO GOOG INTC META MSFT MU NVDA QCOM','Financials':'BAC GS JPM WFC',
        'Healthcare':'ABBV AMGN BMY JNJ LLY MRNA PFE UNH','Energy':'CVX OXY XOM','Retail':'TGT UPS WMT','ETF':'IWM QQQ SPY'}.items():
        sector_map.update({t:sector for t in names.split()})
    surface['sector']=surface.ticker.map(sector_map)
    records=[]
    for (split,sector),g in surface.groupby(['split','sector']):
        records.append(dict(split=split,sector=sector,n=int(g.n.sum()),
            rmse_pp=np.sqrt(np.average(g.rmse_pp**2,weights=g.n)),
            bias_pp=np.average(g.bias_pp,weights=g.n)))
    for split,g in surface.groupby('split'):
        records.append(dict(split=split,sector='Pooled',n=int(g.n.sum()),
            rmse_pp=np.sqrt(np.average(g.rmse_pp**2,weights=g.n)),bias_pp=np.average(g.bias_pp,weights=g.n)))
    summary=pd.DataFrame(records);summary.to_csv(OUT/'surface_summary.csv',index=False)
    rows=[]
    for sector in ['ETF','Financials','Energy','Retail','Healthcare','Tech','Pooled']:
        tr=summary[summary.sector.eq(sector)&summary.split.eq('train')].iloc[0]
        te=summary[summary.sector.eq(sector)&summary.split.eq('test')].iloc[0]
        rows.append([sector,f'{int(tr.n):,}',f'{int(te.n):,}',f'{tr.rmse_pp:.2f}',f'{te.rmse_pp:.2f}',f'{te.bias_pp:+.2f}'])
    write_table('chronological_surface_table.tex',r'Sector & Train rows & Test rows & Train RMSE & Test RMSE & Test bias',rows,'lrrrrr')
    wf=pd.read_csv(ROOT/'code/figures/walk_forward_extended_summary.csv')
    wf['month']=pd.to_datetime(wf.test_date).dt.strftime('%B')
    rows=[]
    for month in ['April','May','June','July']:
        g=wf[wf.month.eq(month)]
        rows.append([month,len(g),f'{100*g.test_rmse.median():.2f}',f'{100*g.test_rmse.min():.2f}',f'{100*g.test_rmse.max():.2f}'])
    write_table('walk_forward_extended_table.tex',r'Test month & Folds & Median RMSE & Minimum & Maximum',rows,'lrrrr')
    rows=[]
    for r in scores[scores.conditioning.eq('endpoint_repricing')].itertuples():
        rows.append([r.cohort,r.ticker,r.horizon,r.n_dates,r.n_quotes,f'{r.absolute_error:.2f}',f'{100*r.inside_spread:.1f}'])
    write_table('forecast_repricing_table.tex',r'Cohort & Ticker & Horizon & Dates & Quotes & MAE & Inside spread (\%)',rows,'llrrrrr')
    print('Five-session joint forecasts:')
    print(scores[scores.conditioning.eq('joint')&scores.horizon.eq(5)][['ticker','mode','n_dates','n_quotes','absolute_error','crps','covered90','width90','mc_se']].to_string(index=False))
    print('Availability:');print(availability.to_string(index=False))
    print('Stock scores:');print(stock.to_string(index=False))
    print('Conditional forecasts:')
    print(scores[scores.conditioning.eq('observed_stock')&scores.horizon.eq(5)][['ticker','mode','n_dates','absolute_error','crps']].to_string(index=False))
    print('Numerical max:',pd.read_csv(OUT/'numerical_check.csv').absolute_change.max())

if __name__=='__main__':main()
