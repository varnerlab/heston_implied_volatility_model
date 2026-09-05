# Synthetic American-option scenarios

This repository connects JumpHMM physical stock paths, fitted IV surfaces,
contract-specific square-root factors, and American-option lattices. The
contract factors are scenario IV inputs, not a jointly calibrated Heston
stock/variance model.

The revised manuscript is in [`paper-arxiv/`](paper-arxiv/). The separate
`paper-jcf/` draft has not been synchronized with this arXiv revision.

## Reproduce the arXiv scenarios and dynamic ablation

Run from the repository root with the supplied `code/Project.toml` and local
`code/Manifest.toml`. The experiments were run with Julia 1.12.7.

```sh
julia --project=code -e 'using Pkg; Pkg.instantiate()'
julia --project=code --startup-file=no -e 'include("code/test/runtests.jl")'
julia --project=code --startup-file=no code/examples/reproduce_arxiv_scenarios.jl
julia --project=code --startup-file=no code/examples/dynamic_iv_ablation.jl
python3 code/scripts/promote_arxiv_ablation.py
make -C paper-arxiv all
```

The first experiment regenerates the corrected GS/LLY illustrations and
their table bodies. The second runs five IV variants on shared stock paths,
with three seeds and 1,000 paths per seed for each ticker. A separate pilot
fixes return normalization before evaluation. Its primary endpoint is
liquidation after ten trading transitions. See the
[fixed analysis design](code/experiments/dynamic-ablation-design.md).

For a quick pipeline check, add `--smoke` to the ablation command. To rebuild
its summaries and figures from saved path marks, use `--render-only` instead.
Smoke outputs have a separate directory and are never promoted to the paper.

The ablation writes `code/results/dynamic_ablation/`, including path marks,
per-seed and pooled summaries, adjacent-variant contrasts, lattice-depth
checks, strike audits, pilot constants, input hashes, a per-file corpus
manifest, and a snapshot of the computational sources. The promotion script
copies generated tables and figures into the self-contained arXiv source
tree; the paper does not read files outside `paper-arxiv/` when compiled.

## Required local inputs

- `code/data/ladder/`: the frozen 457-file, 234,549-filtered-row calibration
  corpus, collected from Alpaca Markets using a free-tier account. The runner
  checks the row count and records file hashes. Do not
  substitute the larger `ladder_extended/` corpus without retraining.
- `code/figures/calibrate_ladders_per_ticker_nn_cache.jld2`: the fitted neural
  surfaces. The calibration scripts under `code/examples/calibrate_ladders*`
  generate the static fitting hierarchy.
- `code/data/pretrained-portfolio-surrogate.jld2`: pretrained JumpHMM marginals.

These are research inputs; installation alone does not download them. The
JumpHMM training window was 2014–2024. Saved portfolio metadata records 424
tickers, 50 states, 2,767 training days, `dt=1/252`, and `rf=0`.
Alpaca's [Market Data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq)
documents Black–Scholes calculations with a Vega-based iterative IV solver.
The experiments fit the reported IV field directly.

## Interpretation

The ablation measures sensitivity of interim marks and model liquidation
P&L. It does not establish predictive superiority or trading profitability.
All monetary option values and P&L are per share. Residual GS strike-condition
violations are reported explicitly; passing the finite LLY grid does not
establish an arbitrage-free market. See [SETUP.md](SETUP.md) for the older
environment and earnings-calendar setup instructions.
