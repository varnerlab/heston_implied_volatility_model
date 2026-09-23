"""Add the availability-driven short-maturity cohort without examining forecast scores."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/results/chronological_validation'
quotes=pd.read_csv(OUT/'market_quotes.csv')
contracts=[]
for (ticker,origin),group in quotes.groupby(['ticker','session']):
    for kind,target in [('put',.95),('call',1.05)]:
        candidates=group[group.type.eq(kind)&group.actual_dte.between(10,21)&
            group.implied_vol.gt(.01)&group.implied_vol.lt(2)&group.moneyness.between(.8,1.2)].copy()
        candidates['expiry_distance']=(candidates.actual_dte-14).abs()
        candidates['strike_distance']=(candidates.moneyness-target).abs()
        candidates=candidates.sort_values(['expiry_distance','expiration','strike_distance','symbol'])
        if candidates.empty:continue
        r=candidates.iloc[0]
        contracts.append(dict(ticker=ticker,origin=origin,kind=kind,symbol=r.symbol,expiry=r.expiration,
            strike=r.strike,spot=r.S,origin_iv=r.implied_vol,origin_mid=r.mid,origin_bid=r.bid,origin_ask=r.ask))
pd.DataFrame(contracts).to_csv(OUT/'origin_contracts_short.csv',index=False)
print('Selected',len(contracts),'short-maturity origin contracts using origin information only.')
