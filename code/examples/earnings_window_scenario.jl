"""
Earnings-window short-premium scenario (Reviewer 3).

Anchors INTC's scenario at the 2026-04-14 capture and runs a 32-day window
that deliberately spans INTC's 04-22 earnings print (day 8). This is the
*modal use case* the framework was sold against; the prior cross-ticker
scenarios all sat in event-free windows by construction.

The scenario uses the per-ticker ψ_NN INTC surface (calibrated across all
15 dates with no earnings-feature awareness), the calibration-collapsed
\bar{θ}_INTC, and the same Heston backbone as GS/LLY (κ=15.0, σ_v=0.5, ρ=-0.6).
Per the §5 Algorithm 1 note, the regime-state and mood multipliers are
held inactive, so the simulator has *no* mechanism to anticipate the
04-22 print — and the honest finding is that path-conditional pricing
across the print is dominated by the modal-non-event paths.

Inputs (INTC 2026-04-14 capture, 30Δ pair on 2026-05-15 expiry):
  S_0 = \$65.17
  put : K=\$60   Δ=-0.299  mid=\$3.28   IV=78.2%
  call: K=\$75   Δ=+0.316  mid=\$2.62   IV=73.6%

Output:
  code/figures/intc_earnings_window_simulation_cache.jld2
  paper-jcf/sections/figures/intc_earnings/
  Aggregate row appended (after run_expanded_scenarios) is recorded
  separately via the standard print_summary path.

Run:
    julia --project=. examples/earnings_window_scenario.jl
    julia --project=. examples/earnings_window_scenario.jl --resim
"""

using Dates
using HestonIV
using Printf
using CSV
using DataFrames
include(joinpath(@__DIR__, "..", "src", "ScenarioTemplate.jl"))
using .ScenarioTemplate

const RESIM = "--resim" in ARGS

const LADDER_DIR    = joinpath(@__DIR__, "..", "data", "ladder")
const FIG_CACHE_DIR = joinpath(@__DIR__, "..", "figures")
const PLOT_DIR      = joinpath(@__DIR__, "..", "..", "paper-jcf", "sections", "figures", "intc_earnings")
const NN_CACHE      = joinpath(FIG_CACHE_DIR, "calibrate_ladders_per_ticker_nn_cache.jld2")
const SIM_CACHE     = joinpath(FIG_CACHE_DIR, "intc_earnings_window_simulation_cache.jld2")
const PORT_PATH     = joinpath(@__DIR__, "..", "data", "pretrained-portfolio-surrogate.jld2")

spec = ScenarioSpec(
    ticker="INTC",
    anchor_date=Date("2026-04-13"),   # captures in the 04-14 dir are 04-13 trading session
    T_days=32,                         # spans 04-22 print at ~day 7 / 32 from 04-13
    K_put=60.0, K_call=75.0,
    market_premium_put=3.28,   market_premium_call=2.62,
    market_iv_put=0.782,       market_iv_call=0.736,
    market_delta_put=-0.299,   market_delta_call=+0.316,
    expiry_label="2026-05-15",
    ticker_prior_ccgr_pct=8.0,
    n_paths=1000, seed=20260414,
    fig_subdir="intc_earnings",
)

hspec = HestonSpec()  # same as GS/LLY for fair comparison

result = run_short_scenario(spec, hspec;
                            nn_cache_path=NN_CACHE,
                            port_path=PORT_PATH,
                            ladder_dir=LADDER_DIR,
                            sim_cache_path=SIM_CACHE,
                            resim=RESIM,
                            use_per_ticker=true,
                            sector="Tech",
                            compute_greeks=true)

render_scenario_figures(result, spec; plot_dir=PLOT_DIR,
                        prefix="intc_earnings")
print_summary(result)

# Append a row to the expanded_scenarios_summary if present (so the
# earnings-window result lives alongside the event-free scenarios).
agg_path = joinpath(FIG_CACHE_DIR, "expanded_scenarios_summary.csv")
row = merge(result.summary_row,
            (sector="Tech", psi_source=String(result.nn_source), earnings_window=true))
df_row = DataFrame([row])
if isfile(agg_path)
    existing = CSV.read(agg_path, DataFrame)
    # Drop a stale INTC earnings-window row if it's already there
    if hasproperty(existing, :earnings_window)
        existing = existing[.!((existing.ticker .== "INTC") .& (existing.earnings_window .== true)), :]
    end
    combined = vcat(existing, df_row; cols=:union)
    CSV.write(agg_path, combined)
else
    CSV.write(agg_path, df_row)
end
@printf("\n[csv] appended INTC earnings-window row to %s\n", agg_path)
