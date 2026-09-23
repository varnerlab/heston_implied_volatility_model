"""Build manuscript tables from completed, unchanged stock-validation results."""
from pathlib import Path
import hashlib,json,tomllib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/results/small_stock_comparison'
GEN=ROOT/'paper-arxiv/sections/generated'
LABELS={'JumpHMM':'JumpHMM','Unchanged':'Unchanged','Adaptive volatility':'Adaptive','Directional':'Directional'}


def write_table(name,header,rows,alignment):
    assert all(len(row)==len(header.split('&')) for row in rows)
    lines=[r'\begin{tabular}{'+alignment+'}',r'\toprule',header+r' \\',r'\midrule']
    lines.extend(' & '.join(map(str,row))+r' \\' for row in rows)
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (GEN/name).write_text('\n'.join(lines)+'\n')


def main():
    settings=json.loads((OUT/'frozen_settings.json').read_text())
    run=tomllib.loads((OUT/'run_manifest.toml').read_text())
    assert run['completed']
    for mapping in [settings['source_sha256'],run['source_sha256']]:
        for file,digest in mapping.items():
            assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest,file
    s=pd.read_csv(OUT/'summary.csv')
    assert settings['selected_decay']==.97 and settings['selected_penalty']=='zero'
    for horizon,name,full in [(5,'stock_validation_main_table.tex',False),
        (5,'stock_validation_five_table.tex',True),(1,'stock_validation_one_table.tex',True)]:
        rows=[]
        methods=['JumpHMM','Unchanged','Adaptive volatility','Directional'] if full else ['JumpHMM','Unchanged','Adaptive volatility']
        for period in [2025,2026]:
            for ticker in ['GS','LLY']:
                for method in methods:
                    r=s[(s.period==period)&(s.ticker==ticker)&(s.horizon==horizon)&(s.method==method)].iloc[0]
                    assert int(r.n_dates)==({2025:{1:249,5:245},2026:{1:21,5:17}}[period][horizon])
                    row=[period,ticker,LABELS[method],f'{r.median_mae:.2f}']
                    if full:row.append(f'{r.rmse:.2f}')
                    row.extend([f'{r.crps:.2f}',f'{100*r.coverage:.1f}' if method!='Unchanged' else '--',
                        f'{r.width:.2f}' if method!='Unchanged' else '--'])
                    rows.append(row)
        header=r'Year & Stock & Model & Median MAE & '+(r'Mean RMSE & ' if full else '')+r'CRPS & Coverage (\%) & Width'
        write_table(name,header,rows,'lll'+('rrrrr' if full else 'rrrr'))
    v=pd.read_csv(OUT/'volatility_selection.csv').groupby('decay').loss.mean()
    d=pd.read_csv(OUT/'direction_selection.csv').groupby('penalty').loss.mean()
    rows=[['EWMA decay',f'{decay:.2f}',f'{loss:.6f}'] for decay,loss in v.items()]
    rows.extend([['Ridge penalty','Zero drift' if np.isinf(penalty) else f'{penalty:g}',f'{loss:.6f}'] for penalty,loss in d.items()])
    write_table('stock_selection_table.tex','Selection & Setting & Validation loss',rows,'llr')
    q=pd.read_csv(ROOT/'code/results/stock_forecast_diagnosis/summary.csv')
    rows=[]
    for ticker in ['GS','LLY']:
        for horizon in [1,5]:
            old=q[(q.ticker==ticker)&(q.horizon==horizon)&(q.method=='legacy_stationary')].iloc[0]
            new=q[(q.ticker==ticker)&(q.horizon==horizon)&(q.method=='filtered_state')].iloc[0]
            rows.append([ticker,horizon,f'{old.median_mae:.2f}',f'{new.median_mae:.2f}',f'{old.crps:.2f}',f'{new.crps:.2f}'])
    write_table('stock_initialization_table.tex','Stock & Horizon & Original MAE & Filtered MAE & Original CRPS & Filtered CRPS',rows,'lrrrrr')
    f=pd.read_csv(ROOT/'code/results/chronological_validation/short_maturity/forecast_scores.csv')
    f=f[(f.horizon==5)&(f.conditioning=='joint')].copy()
    f['median_error']=(f.predicted_median-f.observed).abs()
    dates=f.groupby(['ticker','mode','origin'])[['absolute_error','median_error','squared_error']].mean()
    points=dates.groupby(['ticker','mode']).mean()
    labels={'frozen':'Frozen IV','surface':'Direct surface','relaxation':'Mean reversion','uncoupled':'Uncoupled factor','coupled':'Coupled factor'}
    rows=[]
    for ticker in ['GS','LLY']:
        for mode,label in labels.items():
            r=points.loc[(ticker,mode)]
            rows.append([ticker,label,f'{r.absolute_error:.2f}',f'{r.median_error:.2f}',f'{np.sqrt(r.squared_error):.2f}'])
    write_table('option_point_summary_table.tex','Stock & IV variant & Mean MAE & Median MAE & Mean RMSE',rows,'llrrr')
    generated=['stock_validation_main_table.tex','stock_validation_five_table.tex','stock_validation_one_table.tex',
        'stock_selection_table.tex','stock_initialization_table.tex','option_point_summary_table.tex']
    manifest={name:hashlib.sha256((GEN/name).read_bytes()).hexdigest() for name in generated}
    (OUT/'manuscript_table_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Promoted six tables from verified completed results.')


if __name__=='__main__':main()
