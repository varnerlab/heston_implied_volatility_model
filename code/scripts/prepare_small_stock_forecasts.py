"""Build test forecasts only after historical model selection has been frozen."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from small_stock_models import feature_history,predict_daily_drift

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/results/small_stock_comparison'


def main():
    settings=json.loads((OUT/'frozen_settings.json').read_text())
    for name,digest in settings['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    raw=pd.read_csv(OUT/'historical_closes.csv',parse_dates=['session'])
    historical=raw.pivot(index='session',columns='ticker',values='spot')[['GS','LLY','SPY']]
    current=pd.read_csv(OUT/'closes_2026.csv',parse_dates=['session'])
    grid=pd.bdate_range(current.session.min(),current.session.max()).difference(
        pd.to_datetime(['2026-05-25','2026-06-19','2026-07-03']))
    current=current.pivot(index='session',columns='ticker',values='spot').reindex(grid)[['GS','LLY','SPY']]
    rows=[]
    for period,prices,start,initial in [('2025',historical,pd.Timestamp('2025-01-01'),None),
        ('2026',current,pd.Timestamp('2026-08-04'),settings['initial_variances'])]:
        features,_=feature_history(prices,settings['selected_decay'],initial)
        for ticker in ['GS','LLY']:
            use=features[(features.ticker==ticker)&(features.date>=start)].copy()
            use['daily_drift']=predict_daily_drift(use,settings['models'][ticker])
            for r in use.itertuples():
                origin_spot=prices[ticker].iloc[r.index]
                if not np.isfinite(origin_spot):continue
                for h in [1,5]:
                    end_index=r.index+h
                    if end_index>=len(prices):continue
                    observed=prices[ticker].iloc[end_index]
                    if not np.isfinite(observed):continue
                    rows.append(dict(period=period,ticker=ticker,origin=r.date.date(),
                        endpoint=prices.index[end_index].date(),horizon=h,origin_spot=origin_spot,
                        observed=observed,daily_variance=r.variance,daily_drift=r.daily_drift,
                        return_count=r.return_count,stock_feature_age=r.stock_feature_age,
                        market_feature_age=r.market_feature_age))
    inputs=pd.DataFrame(rows)
    assert not inputs.duplicated(['period','ticker','origin','horizon']).any()
    prior=pd.read_csv(ROOT/'code/results/chronological_validation/stock_scores.csv')
    prior=prior[prior['mode']=='JumpHMM']
    current_inputs=inputs[inputs.period=='2026'].copy()
    for col in ['origin','endpoint']:current_inputs[col]=current_inputs[col].astype(str)
    compared=current_inputs.merge(prior,on=['ticker','origin','endpoint','horizon'],validate='one_to_one',suffixes=('','_original'))
    assert len(compared)==len(prior)==len(current_inputs)
    assert np.array_equal(compared.observed,compared.observed_original)
    inputs.to_csv(OUT/'forecast_inputs.csv',index=False)
    print(inputs.groupby(['period','ticker','horizon']).size())
    print('2026 outcomes exactly match all original stock forecast cases.')


if __name__=='__main__':main()
