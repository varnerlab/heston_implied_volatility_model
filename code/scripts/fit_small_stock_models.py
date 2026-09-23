"""Choose the small comparison's settings using only 2014–2024 observations."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from small_stock_models import DECAYS,PENALTIES,feature_history,fit_ridge,predict_daily_drift

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/results/small_stock_comparison'


def validation_cases(history,closes,ticker,year,horizon):
    use=history[(history.ticker==ticker)&(history.date.dt.year==year)].copy()
    target_index=use['index'].to_numpy()+horizon
    valid=target_index<len(closes)
    use=use.loc[valid].copy();target_index=target_index[valid]
    valid=closes.index[target_index].year==year
    use=use.loc[valid].copy();target_index=target_index[valid]
    y=np.log(closes[ticker].to_numpy()[target_index]/closes[ticker].to_numpy()[use['index']])
    return use,y


def main():
    raw=pd.read_csv(OUT/'historical_closes.csv',parse_dates=['session'])
    training=raw[raw.session<='2024-12-31']
    closes=training.pivot(index='session',columns='ticker',values='spot')[['GS','LLY','SPY']]
    assert closes.index.max()==pd.Timestamp('2024-12-31') and closes.notna().all().all()
    volatility_rows=[]
    for decay in DECAYS:
        history,_=feature_history(closes,decay)
        for ticker in ['GS','LLY']:
            for year in range(2019,2025):
                for h in [1,5]:
                    use,y=validation_cases(history,closes,ticker,year,h)
                    variance=h*use.variance.to_numpy()
                    loss=np.mean(.5*(np.log(variance)+y*y/variance))
                    volatility_rows.append(dict(decay=decay,ticker=ticker,year=year,horizon=h,n=len(y),loss=loss))
    vol=pd.DataFrame(volatility_rows);vol.to_csv(OUT/'volatility_selection.csv',index=False)
    ranked=vol.groupby('decay').loss.mean().reset_index().sort_values(['loss','decay'],ascending=[True,False])
    decay=float(ranked.iloc[0].decay)
    history,_=feature_history(closes,decay)
    regression_rows=[]
    for penalty in PENALTIES:
        for ticker in ['GS','LLY']:
            for year in range(2019,2025):
                model=fit_ridge(history,ticker,f'{year-1}-12-31',penalty)
                for h in [1,5]:
                    use,y=validation_cases(history,closes,ticker,year,h)
                    prediction=h*predict_daily_drift(use,model)
                    loss=np.mean((y-prediction)**2/(h*use.variance.to_numpy()))
                    regression_rows.append(dict(penalty=penalty,ticker=ticker,year=year,horizon=h,n=len(y),loss=loss))
    regression=pd.DataFrame(regression_rows);regression.to_csv(OUT/'direction_selection.csv',index=False)
    ranked_r=regression.groupby('penalty').loss.mean().reset_index().sort_values(['loss','penalty'],ascending=[True,False])
    penalty=float(ranked_r.iloc[0].penalty)
    models={ticker:fit_ridge(history,ticker,'2024-12-31',penalty) for ticker in ['GS','LLY']}
    initial_variances=np.log(closes/closes.shift(1)).var(ddof=1).to_dict()
    source_files=['code/scripts/small_stock_models.py','code/scripts/fit_small_stock_models.py',
                  'code/results/small_stock_comparison/PROTOCOL.md','code/results/small_stock_comparison/historical_closes.csv']
    settings=dict(training_cutoff='2024-12-31',selected_decay=decay,
        selected_penalty='zero' if np.isinf(penalty) else penalty,models=models,
        initial_variances=initial_variances,
        selection_period='2019–2024 expanding annual blocks',
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files})
    (OUT/'frozen_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    print('EWMA selection:');print(ranked.to_string(index=False))
    print('Directional selection:');print(ranked_r.to_string(index=False))
    print('Frozen:',decay,settings['selected_penalty'],models)


if __name__=='__main__':main()
