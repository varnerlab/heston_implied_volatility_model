"""Render the arXiv figures from frozen, verified simulation and fit outputs.

First run export_manuscript_figure_data.jl. No model fitting or simulation occurs
here. The renderer checks its derived statistics against the saved experiment
summaries before writing the manuscript PDFs and PNG previews.
"""
from pathlib import Path
import subprocess
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'code/results/figure_revision'
GENERATED = ROOT / 'paper-arxiv/sections/generated'


def ablation():
    subprocess.run([
        'julia', f'--project={ROOT / "code"}', '--startup-file=no',
        str(ROOT / 'code/scripts/render_ablation_figure.jl'),
    ], check=True, cwd=ROOT)


def scenario_paths():
    # Reuse the HMM paper's Plots/GR conventions for the scenario figures.
    subprocess.run([
        'julia', f'--project={ROOT / "code"}', '--startup-file=no',
        str(ROOT / 'code/scripts/render_scenario_figures.jl'),
    ], check=True, cwd=ROOT)


def smiles():
    subprocess.run([
        'julia', f'--project={ROOT / "code"}', '--startup-file=no',
        str(ROOT / 'code/scripts/render_smile_figure.jl'),
    ], check=True, cwd=ROOT)
    diag = pd.read_csv(DATA/'smile_date_diagnostics.csv')
    diag = diag[(diag.date=='2026-05-11') & (diag.scope=='near_atm_shown_maturity')].set_index('ticker')
    tickers = ['SPY','NVDA','MSFT','LLY','GS','AVGO']
    # Keep the near-money diagnostics in the supplementary table.
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',
           r'Ticker & DTE & $N$ & Observed IV & Predicted IV & Bias \\',r'\midrule']
    for ticker in tickers:
        r=diag.loc[ticker]
        lines.append(f'{ticker} & {int(r.displayed_dte)} & {int(r.n)} & {r.mean_observed_iv:.2f} & {r.mean_predicted_iv:.2f} & {r.mean_bias:+.2f} '+r'\\')
    lines.extend([r'\bottomrule\end{tabular}'])
    (GENERATED/'smile_date_bias_table.tex').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    ablation()
    scenario_paths()
    smiles()
    print('Rendered manuscript figures and checked their statistics against saved results.')
