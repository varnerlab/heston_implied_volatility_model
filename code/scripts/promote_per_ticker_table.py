"""Build the per-ticker versus sector-network table from the saved per-ticker NN run log.

The last column comes from `diagnose_per_ticker_regressions.jl`, which refitted
the small (8-unit) per-ticker networks at the sector networks' 16-unit size.
"""
from pathlib import Path
import csv
import re

ROOT=Path(__file__).resolve().parents[2]
LOG=ROOT/'code/logs/per_ticker_nn.log'
REFIT=ROOT/'code/results/per_ticker_regression_diagnosis.csv'
GEN=ROOT/'paper-arxiv/sections/generated'
ROW=re.compile(r'^\s+([A-Z]+)\s+\S+\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)\s*$')
ARCH=re.compile(r'^\s+([A-Z]+)\s+\[.*arch=2->(\d+)->', re.M)


def thousands(n):
    # LaTeX thousands separator used throughout the manuscript
    return f'{n:,}'.replace(',','{,}')


def main():
    lines=LOG.read_text().splitlines()
    start=next(i for i,line in enumerate(lines) if 'Qualified tickers (sorted by improvement' in line)
    rows=[]
    for line in lines[start+3:]:
        m=ROW.match(line)
        if not m:break
        ticker,n,sector,per_ticker,delta=m.groups()
        rows.append((ticker,int(n),sector,per_ticker,float(delta)))
    assert len(rows)==29
    improved=sum(r[4]>0 for r in rows)
    worsened=sum(r[4]<0 for r in rows)
    summary=re.search(r'improved over sector NN: (\d+)\s*\|\s*regressed: (\d+)',LOG.read_text())
    assert (improved,worsened)==tuple(map(int,summary.groups()))
    units=dict(ARCH.findall(LOG.read_text()))
    with REFIT.open() as f:
        refit={r['ticker']:float(r['large_final']) for r in csv.DictReader(f)}
    # every small network, and only those, has a larger-network refit
    assert set(refit)=={t for t,u in units.items() if u=='8'}
    assert all(abs(float(r['small_final'])-float(r['logged_small']))<0.01 for r in csv.DictReader(REFIT.open()))
    refit_better=sum(refit[t]<float(s) for t,_,s,_,_ in rows if t in refit)
    out=[r'\begin{tabular}{lrrrrrr}',r'\toprule',
         r'Ticker & Observations & Units & Sector NN & Per-ticker NN & Reduction & 16-unit refit \\',r'\midrule']
    # typeset negative reductions with a math minus sign
    for t,n,s,p,d in rows:
        reduction=f'${d:.2f}$' if d<0 else f'{d:.2f}'
        refitted=f'{refit[t]:.2f}' if t in refit else '--'
        out.append(f'{t} & {thousands(n)} & {units[t]} & {s} & {p} & {reduction} & {refitted}'+r' \\')
    out.extend([r'\midrule',
        rf'\multicolumn{{6}}{{l}}{{Tickers with lower per-ticker RMSE}} & {improved} \\',
        rf'\multicolumn{{6}}{{l}}{{Tickers with higher per-ticker RMSE}} & {worsened} \\',
        rf'\multicolumn{{6}}{{l}}{{Refitted tickers with lower RMSE than their sector network}} & {refit_better} of {len(refit)} \\',
        r'\bottomrule',r'\end{tabular}'])
    (GEN/'per_ticker_comparison_table.tex').write_text('\n'.join(out)+'\n')
    print(f'Wrote per-ticker table: {len(rows)} tickers, {improved} improved, {worsened} worsened.')


if __name__=='__main__':main()
