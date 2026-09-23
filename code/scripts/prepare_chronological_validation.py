"""Audit close captures and freeze chronological forecast inputs without scoring them."""
from pathlib import Path
from datetime import timedelta
import hashlib
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'code/results/chronological_validation'
CUTOFF = pd.Timestamp('2026-07-31 23:59:59')
END = pd.Timestamp('2026-09-04')
HOLIDAYS = {pd.Timestamp(d) for d in ['2026-05-25','2026-06-19','2026-07-03']}
SECTORS = {}
for sector, names in {
    'Tech':'AAPL AMD AVGO GOOG INTC META MSFT MU NVDA QCOM',
    'Financials':'BAC GS JPM WFC', 'Healthcare':'ABBV AMGN BMY JNJ LLY MRNA PFE UNH',
    'Energy':'CVX OXY XOM', 'Retail':'TGT UPS WMT', 'ETF':'IWM QQQ SPY',
}.items():
    SECTORS.update({ticker:sector for ticker in names.split()})


def advance_session(day, n=1):
    day = pd.Timestamp(day).normalize()
    for _ in range(n):
        day += timedelta(days=1)
        while day.weekday() >= 5 or day in HOLIDAYS:
            day += timedelta(days=1)
    return day


def valid_quotes(df):
    cols = ['strike','und_close','bid','ask','bid_size','ask_size']
    return (np.isfinite(df[cols]).all(axis=1) & df.strike.gt(0) & df.und_close.gt(0)
            & df.bid.gt(0) & df.ask.ge(df.bid) & df.bid_size.gt(0) & df.ask_size.gt(0))


def fit_rows(df):
    return (valid_quotes(df) & df.implied_vol.gt(.01) & df.implied_vol.lt(2)
            & df.moneyness.between(.8,1.2) & df.actual_dte.gt(0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources=[]
    for root in [ROOT/'code/data/ladder', ROOT/'code/data/ladder_extended']:
        for path in sorted(root.glob('options-*/*_dte_ladder_*.csv')):
            row=pd.read_csv(path,nrows=1).iloc[0]
            session=pd.Timestamp(row.und_session_date)
            capture=pd.Timestamp(row.capture_ts)
            if session > END: continue
            valid = session + pd.Timedelta(hours=20) <= capture < advance_session(session)+pd.Timedelta(hours=13,minutes=30)
            sources.append(dict(path=str(path.relative_to(ROOT)), ticker=str(row.underlying),
                session=str(session.date()), capture=str(capture),
                close_capture=bool(valid), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest=pd.DataFrame(sources)
    selected=(manifest[manifest.close_capture].sort_values(['capture','path'])
              .drop_duplicates(['ticker','session'],keep='last'))
    manifest['selected']=manifest.path.isin(selected.path)
    manifest.to_csv(OUT/'source_manifest.csv',index=False)
    train=[]; test=[]; market=[]; spots=[]; audit=[]; legacy=[]
    # Independently reconstruct the row counts used in the saved 53-fold run.
    legacy_counts={}
    for source in sources:
        path=ROOT/source['path']
        directory=path.parent.name
        label=pd.to_datetime(directory[8:],format='%m-%d-%Y')
        if label > pd.Timestamp('2026-07-17'): continue
        d=pd.read_csv(path)
        m=d.strike/d.und_close.iloc[0]
        keep=d.implied_vol.gt(.01)&d.implied_vol.lt(2)&d.bid.gt(0)&m.between(.8,1.2)&d.actual_dte.gt(0)
        legacy_counts[str(label.date())]=legacy_counts.get(str(label.date()),0)+int(keep.sum())
        legacy.append(dict(path=source['path'], filtered_rows=int(keep.sum()),
            crossed_or_missing_ask=int((keep & (~np.isfinite(d.ask)|d.ask.lt(d.bid))).sum()),
            missing_or_nonpositive_sizes=int((keep & (~d.bid_size.gt(0)|~d.ask_size.gt(0))).sum())))
    saved=pd.read_csv(ROOT/'code/figures/walk_forward_extended_summary.csv')
    for row in saved.itertuples():
        assert legacy_counts[row.test_date]==row.n_test_obs
        assert sum(n for d,n in legacy_counts.items() if d<row.test_date)==row.n_train_obs
    pd.DataFrame(legacy).to_csv(OUT/'legacy_walk_forward_audit.csv',index=False)
    for row in selected.itertuples():
        df=pd.read_csv(ROOT/row.path)
        df=df.drop_duplicates('symbol',keep='last').copy()
        assert df.underlying.eq(row.ticker).all()
        assert df.und_session_date.eq(row.session).all()
        assert df.capture_ts.nunique()==1
        assert df.und_close.nunique()==1
        session=pd.Timestamp(row.session)
        df['ticker']=row.ticker;df['sector']=SECTORS[row.ticker]
        df['session']=row.session;df['S']=df.und_close
        df['moneyness']=df.strike/df.S
        df['actual_dte']=(pd.to_datetime(df.expiration)-session).dt.days
        df['mid']=(df.bid+df.ask)/2
        quote_mask=valid_quotes(df)&df.actual_dte.gt(0)
        fitting=fit_rows(df)
        cols=['ticker','sector','session','capture_ts','symbol','expiration','type','strike','S',
              'actual_dte','moneyness','bid','ask','mid','implied_vol','bid_size','ask_size']
        clean=df.loc[fitting,cols]
        if pd.Timestamp(row.capture)<=CUTOFF and session<=CUTOFF.normalize():train.append(clean)
        elif session>=pd.Timestamp('2026-08-01'):test.append(clean)
        if row.ticker in ['GS','LLY']:
            spots.append(dict(ticker=row.ticker,session=row.session,spot=float(df.S.iloc[0]),capture=row.capture))
            if session>=pd.Timestamp('2026-08-01'):market.append(df.loc[quote_mask,cols])
        audit.append(dict(ticker=row.ticker,session=row.session,raw_rows=len(df),
            valid_quote_rows=int(quote_mask.sum()),fit_rows=int(fitting.sum())))
    train=pd.concat(train,ignore_index=True);test=pd.concat(test,ignore_index=True)
    market=pd.concat(market,ignore_index=True)
    spots=pd.DataFrame(spots).sort_values(['ticker','session'])
    train.to_csv(OUT/'training.csv',index=False)
    test.to_csv(OUT/'surface_test.csv',index=False)
    market.to_csv(OUT/'market_quotes.csv',index=False)
    spots.to_csv(OUT/'stock_sessions.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'quote_audit.csv',index=False)
    contracts=[];availability=[]
    for row in spots[spots.session.ge('2026-08-01')].itertuples():
        origin=market[market.ticker.eq(row.ticker)&market.session.eq(row.session)]
        for kind,target in [('put',.95),('call',1.05)]:
            eligible=origin[origin.type.eq(kind)&origin.actual_dte.between(25,45)&fit_rows(origin.rename(columns={'S':'und_close'}))].copy()
            # Origin-only ranking. Endpoint availability never enters this selection.
            eligible['expiry_distance']=(eligible.actual_dte-31).abs()
            eligible['strike_distance']=(eligible.moneyness-target).abs()
            eligible=eligible.sort_values(['expiry_distance','expiration','strike_distance','symbol'])
            availability.append(dict(ticker=row.ticker,origin=row.session,kind=kind,candidates=len(eligible)))
            if eligible.empty:continue
            chosen=eligible.iloc[0]
            contract=dict(ticker=row.ticker,origin=row.session,kind=kind,symbol=chosen.symbol,
                expiry=chosen.expiration,strike=float(chosen.strike),spot=float(chosen.S),
                origin_iv=float(chosen.implied_vol),origin_mid=float(chosen.mid),
                origin_bid=float(chosen.bid),origin_ask=float(chosen.ask))
            contracts.append(contract)
    pd.DataFrame(contracts).to_csv(OUT/'origin_contracts.csv',index=False)
    pd.DataFrame(availability).to_csv(OUT/'origin_availability.csv',index=False)
    info=dict(training_rows=len(train),training_sessions=train.session.nunique(),
        training_start=train.session.min(),training_end=train.session.max(),
        test_rows=len(test),test_sessions=test.session.nunique(),
        test_start=test.session.min(),test_end=test.session.max(),
        source_files=len(manifest),selected_files=int(manifest.selected.sum()),
        excluded_capture_files=int((~manifest.close_capture).sum()),
        superseded_files=int((manifest.close_capture&~manifest.selected).sum()),
        legacy_folds_verified=len(saved),selected_origin_contracts=len(contracts))
    (OUT/'data_summary.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
