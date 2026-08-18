"""
Short-premium scenario study for GS.

Sells a 31-calendar-day 30Δ put and call. Forward-simulates 1,000 GS price
paths across 22 exchange sessions from the pretrained JumpHMM marginal,
evolves a CIR-like contract variance with return-shock coupling along each
path, and prices both contracts via Leisen-Reimer American using the
calibrated ψ_NN surface.

The shared ScenarioTemplate keeps the calendar and trading clocks explicit.

Real GS contracts pulled from the 04-28-2026 capture (closest to 30 DTE / 30Δ):
  put : GS 2026-05-29 K=\$890   Δ = -0.295   bid/ask = 14.54/18.48   mid = \$16.51   IV = 31.3%
  call: GS 2026-05-29 K=\$970   Δ = +0.328   bid/ask = 13.70/18.47   mid = \$16.09   IV = 28.9%

Run:
    julia --project=. examples/gs_short_premium_simulation.jl
    julia --project=. examples/gs_short_premium_simulation.jl --resim
"""

using Dates
using HestonIV
include(joinpath(@__DIR__, "..", "src", "ScenarioTemplate.jl"))
using .ScenarioTemplate

const RESIM = "--resim" in ARGS

const LADDER_DIR    = joinpath(@__DIR__, "..", "data", "ladder")
const FIG_CACHE_DIR = joinpath(@__DIR__, "..", "figures")
const PLOT_DIR      = joinpath(@__DIR__, "..", "..", "paper-arxiv", "sections", "figures", "gs")
const NN_CACHE      = joinpath(FIG_CACHE_DIR, "calibrate_ladders_per_ticker_nn_cache.jld2")
const SIM_CACHE     = joinpath(FIG_CACHE_DIR, "gs_short_premium_simulation_cache.jld2")
const PORT_PATH     = joinpath(@__DIR__, "..", "data", "pretrained-portfolio-surrogate.jld2")

spec = ScenarioSpec(
    ticker="GS",
    anchor_date=Date("2026-04-28"),
    expiry_date=Date("2026-05-29"),
    K_put=890.0, K_call=970.0,
    market_premium_put=16.51,   market_premium_call=16.085,
    market_iv_put=0.3125,       market_iv_call=0.2893,
    market_delta_put=-0.2951,   market_delta_call=+0.3278,
    expiry_label="2026-05-29",
    ticker_prior_ccgr_pct=10.0,
    market_holidays=[Date("2026-05-25")],
    n_paths=1000, seed=20260429,
    fig_subdir="gs",
)

hspec = VarianceSpec()  # κ=15.0, σ_v=0.5, ρ=-0.6, r=0.0425, q=0, N_LR=201

result = run_short_scenario(spec, hspec;
                            nn_cache_path=NN_CACHE,
                            port_path=PORT_PATH,
                            ladder_dir=LADDER_DIR,
                            sim_cache_path=SIM_CACHE,
                            resim=RESIM)

render_scenario_figures(result, spec; plot_dir=PLOT_DIR)
print_summary(result)
