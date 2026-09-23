"""Causal EWMA features and strongly regularized stock-drift regression."""
import numpy as np
import pandas as pd

FEATURES=['stock_1','stock_5','stock_20','market_5']
DECAYS=[.80,.90,.94,.97,.99]
PENALTIES=[.01,.1,1.,10.,100.,1000.,float('inf')]


def feature_history(closes,decay,initial_variances=None):
    """Rows use only adjacent-session returns known by that row's date."""
    prices=closes[['GS','LLY','SPY']].to_numpy(float)
    returns=np.full_like(prices,np.nan)
    returns[1:]=np.log(prices[1:]/prices[:-1])
    variances=np.full_like(prices,np.nan)
    histories=[[],[],[]];last_dates=[None,None,None]
    state=np.full(3,np.nan)
    if initial_variances is not None:
        state=np.array([initial_variances[t] for t in ['GS','LLY','SPY']])
    records=[]
    for i,date in enumerate(closes.index):
        for j in range(3):
            r=returns[i,j]
            if np.isfinite(r):
                histories[j].append(r);last_dates[j]=date
                if np.isfinite(state[j]):
                    state[j]=decay*state[j]+(1-decay)*r*r
                elif len(histories[j])==60:
                    state[j]=np.var(histories[j],ddof=1)
            if np.isfinite(state[j]):state[j]=max(state[j],1e-12)
        variances[i]=state
        for j,ticker in enumerate(['GS','LLY']):
            if len(histories[j])<20 or len(histories[2])<5 or not np.isfinite(state[[j,2]]).all():
                continue
            sd=np.sqrt(state[j]);market_sd=np.sqrt(state[2])
            record=dict(date=date,ticker=ticker,variance=state[j],
                stock_1=histories[j][-1]/sd,
                stock_5=sum(histories[j][-5:])/(np.sqrt(5)*sd),
                stock_20=sum(histories[j][-20:])/(np.sqrt(20)*sd),
                market_5=sum(histories[2][-5:])/(np.sqrt(5)*market_sd),
                return_count=len(histories[j]),
                stock_feature_age=(date-last_dates[j]).days,
                market_feature_age=(date-last_dates[2]).days,
                index=i)
            # The response is separate from the origin features and is never
            # used without checking its endpoint against the training cutoff.
            if i+1<len(closes) and np.isfinite(returns[i+1,j]):
                record['response']=returns[i+1,j]/sd
                record['response_date']=closes.index[i+1]
            else:
                record['response']=np.nan;record['response_date']=pd.NaT
            records.append(record)
    return pd.DataFrame(records),pd.DataFrame(variances,index=closes.index,columns=closes.columns)


def fit_ridge(history,ticker,cutoff,penalty):
    use=history[(history.ticker==ticker)&(history.response_date<=pd.Timestamp(cutoff))&history.response.notna()]
    X=use[FEATURES].to_numpy(float);y=use.response.to_numpy(float)
    assert len(y)>100
    scale=np.maximum(np.sqrt(np.mean(X*X,axis=0)),1e-12)
    Z=X/scale
    if np.isinf(penalty):coef=np.zeros(len(FEATURES))
    else:coef=np.linalg.solve(Z.T@Z/len(y)+penalty*np.eye(len(FEATURES)),Z.T@y/len(y))
    return dict(coefficients=coef.tolist(),feature_scale=scale.tolist(),n_training_rows=len(y),
                last_response_date=str(use.response_date.max().date()))


def predict_daily_drift(history,model):
    normalized=(history[FEATURES].to_numpy()/np.array(model['feature_scale']))@np.array(model['coefficients'])
    return np.sqrt(history.variance.to_numpy())*normalized
