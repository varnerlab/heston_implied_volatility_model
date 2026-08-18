"""
Expanded cross-ticker short-premium scenarios (Reviewer 1, 3).

Re-runs the §5 short-premium pipeline on five additional tickers spanning
sector + liquidity profiles:
- SPY  (ETF, broad index)
- XOM  (Energy)
- AMD  (Tech, smaller-cap)
- PFE  (Healthcare, **unqualified** for per-ticker fit → sector NN fallback;
        demonstrates low-liquidity behavior)
- JNJ  (Healthcare large-cap)

Strikes for each ticker are pulled from the 2026-04-28 capture at the
closest ~30Δ, ~30-DTE pair (expiry 2026-05-29). The CIR-like factor settings
are the same κ=15.0, σ_v=0.5, ρ=-0.6 as the GS/LLY scenarios.

Output:
  code/figures/{ticker}_short_premium_simulation_cache.jld2   (sim cache)
  paper-jcf/sections/figures/{ticker}/                        (4 fig sets)
  code/figures/expanded_scenarios_summary.csv                 (aggregate)

Run:
    julia --project=. examples/run_expanded_scenarios.jl
    julia --project=. examples/run_expanded_scenarios.jl --resim
"""

using CSV
using DataFrames
using Dates
using HestonIV
using Printf

include(joinpath(@__DIR__, "..", "src", "ScenarioTemplate.jl"))
using .ScenarioTemplate

const RESIM = "--resim" in ARGS

const LADDER_DIR    = joinpath(@__DIR__, "..", "data", "ladder")
const FIG_CACHE_DIR = joinpath(@__DIR__, "..", "figures")
const PAPER_FIG_BASE = joinpath(@__DIR__, "..", "..", "paper-jcf", "sections", "figures")
const NN_CACHE      = joinpath(FIG_CACHE_DIR, "calibrate_ladders_per_ticker_nn_cache.jld2")
const PORT_PATH     = joinpath(@__DIR__, "..", "data", "pretrained-portfolio-surrogate.jld2")

# Each entry: ScenarioSpec args + sector-fallback metadata
const SPECS = [
    (ticker="SPY", sector="ETF",        per_ticker=true, prior_ccgr_pct=8.0,
     K_put=696.0, K_call=729.0,
     mid_put=6.96,  mid_call=5.28,
     iv_put=0.173,  iv_call=0.134,
     d_put=-0.298,  d_call=+0.303),
    (ticker="XOM", sector="Energy",     per_ticker=true, prior_ccgr_pct=8.0,
     K_put=145.0, K_call=160.0,
     mid_put=3.29,  mid_call=2.20,
     iv_put=0.340,  iv_call=0.308,
     d_put=-0.321,  d_call=+0.276),
    (ticker="AMD", sector="Tech",       per_ticker=true, prior_ccgr_pct=12.0,
     K_put=300.0, K_call=365.0,
     mid_put=13.29, mid_call=10.68,
     iv_put=0.657,  iv_call=0.647,
     d_put=-0.303,  d_call=+0.302),
    (ticker="PFE", sector="Healthcare", per_ticker=false, prior_ccgr_pct=7.0,
     K_put=26.0,  K_call=28.0,
     mid_put=0.67,  mid_call=0.24,
     iv_put=0.308,  iv_call=0.230,
     d_put=-0.381,  d_call=+0.234),
    (ticker="JNJ", sector="Healthcare", per_ticker=true, prior_ccgr_pct=8.0,
     K_put=220.0, K_call=235.0,
     mid_put=3.44,  mid_call=2.77,
     iv_put=0.254,  iv_call=0.212,
     d_put=-0.308,  d_call=+0.316),
]

const EXPIRY      = "2026-05-29"
const EXPIRY_DATE = Date(EXPIRY)
const ANCHOR_DATE = Date("2026-04-28")
const MARKET_HOLIDAYS = [Date("2026-05-25")]

hspec = VarianceSpec()

summary_rows = NamedTuple[]
for (i, s) in enumerate(SPECS)
    println("\n" * "="^78)
    @printf("  TICKER %d/%d:  %s  (sector=%s  per_ticker=%s)\n",
            i, length(SPECS), s.ticker, s.sector, s.per_ticker)
    println("="^78)

    spec = ScenarioSpec(
        ticker=s.ticker, anchor_date=ANCHOR_DATE,
        expiry_date=EXPIRY_DATE,
        K_put=s.K_put, K_call=s.K_call,
        market_premium_put=s.mid_put,   market_premium_call=s.mid_call,
        market_iv_put=s.iv_put,         market_iv_call=s.iv_call,
        market_delta_put=s.d_put,       market_delta_call=s.d_call,
        expiry_label=EXPIRY,
        ticker_prior_ccgr_pct=s.prior_ccgr_pct,
        market_holidays=MARKET_HOLIDAYS,
        n_paths=1000, seed=20260429,
        fig_subdir=lowercase(s.ticker),
    )

    sim_cache = joinpath(FIG_CACHE_DIR, "$(lowercase(s.ticker))_short_premium_simulation_cache.jld2")
    plot_dir  = joinpath(PAPER_FIG_BASE, lowercase(s.ticker))

    result = run_short_scenario(spec, hspec;
                                nn_cache_path=NN_CACHE, port_path=PORT_PATH,
                                ladder_dir=LADDER_DIR, sim_cache_path=sim_cache,
                                resim=RESIM,
                                use_per_ticker=s.per_ticker,
                                sector=s.sector,
                                compute_greeks=true)
    render_scenario_figures(result, spec; plot_dir=plot_dir)
    print_summary(result)

    push!(summary_rows, merge(result.summary_row,
        (sector=s.sector, psi_source=String(result.nn_source),
         earnings_window=false)))
end

# Persist aggregate summary
df = DataFrame(summary_rows)
out = joinpath(FIG_CACHE_DIR, "expanded_scenarios_summary.csv")
CSV.write(out, df)
@printf("\n[csv] wrote -> %s\n", out)

# Console table
println("\n" * "="^120)
println("  EXPANDED CROSS-TICKER SCENARIOS — entry-edge, P&L, premium-kept summary")
println("="^120)
println("  ticker  sector       ψ-src     K_put    K_call    entry_edge_put   entry_edge_call   worst_loss_put   worst_loss_call   kept_put(%)  kept_call(%)")
println("  " * "-"^120)
for r in summary_rows
    @printf("  %-5s   %-11s  %-9s %7.2f  %7.2f       %+8.3f          %+8.3f       %+10.2f       %+10.2f         %5.1f         %5.1f\n",
            r.ticker, r.sector, r.psi_source,
            r.K_put, r.K_call,
            r.entry_edge_put, r.entry_edge_call,
            r.worst_loss_put, r.worst_loss_call,
            r.pct_premium_kept_put, r.pct_premium_kept_call)
end
println("\nDone.")
