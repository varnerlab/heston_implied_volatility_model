"""Build the per-ticker versus sector-network table from the saved per-ticker NN run log."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
LOG=ROOT/'code/logs/per_ticker_nn.log'
GEN=ROOT/'paper-arxiv/sections/generated'
ROW=re.compile(r'^\s+([A-Z]+)\s+\S+\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)\s*$')


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
    out=[r'\begin{tabular}{lrrrr}',r'\toprule',
         r'Ticker & Observations & Sector NN & Per-ticker NN & Reduction \\',r'\midrule']
    # typeset negative reductions with a math minus sign
    out.extend(f'{t} & {thousands(n)} & {s} & {p} & '+(f'${d:.2f}$' if d<0 else f'{d:.2f}')+r' \\' for t,n,s,p,d in rows)
    out.extend([r'\midrule',
        rf'\multicolumn{{4}}{{l}}{{Tickers with lower per-ticker RMSE}} & {improved} \\',
        rf'\multicolumn{{4}}{{l}}{{Tickers with higher per-ticker RMSE}} & {worsened} \\',
        r'\bottomrule',r'\end{tabular}'])
    (GEN/'per_ticker_comparison_table.tex').write_text('\n'.join(out)+'\n')
    print(f'Wrote per-ticker table: {len(rows)} tickers, {improved} improved, {worsened} worsened.')


if __name__=='__main__':main()
