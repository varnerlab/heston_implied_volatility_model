"""Compare completed primary and wider-cutoff runs and generate manuscript tables."""
from pathlib import Path
import hashlib
import json
import tomllib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'code/results'
OUT = RESULTS / 'truncated_emissions'
GEN = ROOT / 'paper-arxiv/sections/generated'


def verify(directory, name, cutoff):
    manifest = tomllib.loads((directory / name).read_text())
    assert manifest['completed'], directory
    assert manifest['emission_spec']['family'] == 'truncated_student_t'
    assert manifest['emission_spec']['standardized_cutoff'] == cutoff
    for file, digest in manifest.get('source_sha256', {}).items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == digest, file
    return manifest


def table(name, header, rows, align):
    lines = [r'\begin{tabular}{' + align + '}', r'\toprule',
             header + r' \\', r'\midrule']
    lines += [' & '.join(map(str, row)) + r' \\' for row in rows]
    lines += [r'\bottomrule\end{tabular}']
    (GEN / name).write_text('\n'.join(lines) + '\n')


def main():
    tails, forecasts = [], []
    for cutoff, base in [(10, RESULTS), (20, OUT / 'wide')]:
        for folder, manifest in [('fitted_scenarios', 'run_manifest.toml'),
                                 ('dynamic_ablation', 'config.toml'),
                                 ('chronological_validation', 'run_manifest.toml'),
                                 ('chronological_validation/short_maturity', 'run_manifest.toml'),
                                 ('small_stock_comparison', 'run_manifest.toml')]:
            verify(base / folder, manifest, cutoff)
        fitted = pd.read_csv(base / 'fitted_scenarios/summary.csv')
        ablation = pd.read_csv(base / 'dynamic_ablation/summary.csv')
        options = pd.read_csv(base / 'chronological_validation/short_maturity/forecast_scores.csv')
        stock = pd.read_csv(base / 'small_stock_comparison/summary.csv')
        for ticker in ['GS', 'LLY']:
            row = fitted[fitted.ticker.eq(ticker)].iloc[0]
            tails.append(dict(experiment='Terminal illustration', ticker=ticker, cutoff=cutoff,
                              mean_pnl=row.mean_pnl_call, q05_pnl=row.q05_pnl_call,
                              es05_pnl=row.es05_pnl_call))
            row = ablation[ablation.ticker.eq(ticker) & ablation.kind.eq('call') &
                           ablation['mode'].eq('coupled') & ablation.step.eq(10) &
                           ablation.seed.eq(0)].iloc[0]
            tails.append(dict(experiment='Ten-session ablation', ticker=ticker, cutoff=cutoff,
                              mean_pnl=row.mean_pnl, q05_pnl=row.q05_pnl, es05_pnl=row.es05_pnl))
            rows = options[options.ticker.eq(ticker) & options.horizon.eq(5) &
                           options.conditioning.eq('joint') & options['mode'].eq('coupled')]
            mae = rows.groupby('origin').absolute_error.mean().mean()
            forecasts.append(dict(experiment='Option MAE, 2026', ticker=ticker, cutoff=cutoff, value=mae))
            for period in [2025, 2026]:
                row = stock[stock.period.eq(period) & stock.ticker.eq(ticker) &
                            stock.horizon.eq(5) & stock.method.eq('JumpHMM')].iloc[0]
                forecasts.append(dict(experiment=f'Stock CRPS, {period}', ticker=ticker,
                                      cutoff=cutoff, value=row.crps))
    tails = pd.DataFrame(tails).sort_values(['experiment', 'ticker', 'cutoff'])
    forecasts = pd.DataFrame(forecasts).sort_values(['experiment', 'ticker', 'cutoff'])
    tails.to_csv(OUT / 'call_tail_sensitivity.csv', index=False)
    forecasts.to_csv(OUT / 'forecast_sensitivity.csv', index=False)
    bounds = pd.read_csv(OUT / 'bounds.csv').sort_values(['ticker', 'cutoff'])
    table('emission_bounds_table.tex', r'Stock & $b$ & Lower return (\%) & Upper return (\%) & Removed mass (\%)',
          [[r.ticker, int(r.cutoff), f'{r.min_daily_return_pct:.2f}', f'{r.max_daily_return_pct:.2f}',
            f'{100*r.original_probability_removed:.4f}'] for r in bounds.itertuples()], 'lrrrr')
    table('tail_sensitivity_table.tex', r'Experiment & Stock & $b$ & Mean P\&L & 5\% quantile & ES$_{5\%}$',
          [[r.experiment, r.ticker, int(r.cutoff), f'{r.mean_pnl:.3f}', f'{r.q05_pnl:.3f}',
            f'{r.es05_pnl:.3f}'] for r in tails.itertuples()], 'llrrrr')
    pivot = forecasts.pivot(index=['experiment', 'ticker'], columns='cutoff', values='value')
    table('tail_forecast_table.tex', r'Quantity & Stock & $b=10$ & $b=20$ & Change',
          [[experiment, ticker, f'{r[10]:.6f}', f'{r[20]:.6f}', f'{r[20]-r[10]:+.6f}']
           for (experiment, ticker), r in pivot.iterrows()], 'llrrr')
    (OUT / 'table_manifest.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [GEN / n for n in ['emission_bounds_table.tex', 'tail_sensitivity_table.tex',
                                   'tail_forecast_table.tex']]}, indent=2) + '\n')
    print(tails.to_string(index=False))
    print(pivot.to_string())


if __name__ == '__main__':
    main()
