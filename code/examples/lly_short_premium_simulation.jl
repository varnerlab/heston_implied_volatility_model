"""
Short-premium scenario study for LLY.

Sells a 30-day 30Δ put and a 30-day 30Δ call today. Forward-simulates 1,000
LLY price paths from the pretrained JumpHMM marginal, evolves a Heston-style
CIR variance with leverage coupling along each path, and prices both
contracts via Leisen–Reimer American at every daily step using the
calibrated per-ticker ψ_NN surface.

Refactored 2026-05-14 to use ScenarioTemplate. The numerics are unchanged;
the previous monolithic body lives in git history.

Real LLY contracts pulled from the 04-28-2026 capture (closest to 30 DTE / 30Δ):
  put : LLY 2026-05-29 K=\$825   Δ = -0.303   bid/ask = 18.95/27.65   mid = \$23.30   IV = 44.4%
  call: LLY 2026-05-29 K=\$940   Δ = +0.309   bid/ask = 17.88/23.64   mid = \$20.76   IV = 44.0%

The +24.7%/yr unconditional CCGR baked into the trained LLY JumpHMM model
(from the 2014–2024 Mounjaro/Zepbound regime) is anchored down to a 10%/yr
healthcare long-run prior via a uniform per-step CCGR shift.

Run:
    julia --project=. examples/lly_short_premium_simulation.jl
    julia --project=. examples/lly_short_premium_simulation.jl --resim
"""

using Dates
using HestonIV
include(joinpath(@__DIR__, "..", "src", "ScenarioTemplate.jl"))
using .ScenarioTemplate

const RESIM = "--resim" in ARGS

const LADDER_DIR    = joinpath(@__DIR__, "..", "data", "ladder")
const FIG_CACHE_DIR = joinpath(@__DIR__, "..", "figures")
const PLOT_DIR      = joinpath(@__DIR__, "..", "..", "paper-jcf", "sections", "figures", "lly")
const NN_CACHE      = joinpath(FIG_CACHE_DIR, "calibrate_ladders_per_ticker_nn_cache.jld2")
const SIM_CACHE     = joinpath(FIG_CACHE_DIR, "lly_short_premium_simulation_cache.jld2")
const PORT_PATH     = joinpath(@__DIR__, "..", "data", "pretrained-portfolio-surrogate.jld2")

spec = ScenarioSpec(
    ticker="LLY",
    anchor_date=Date("2026-04-28"),
    T_days=31,
    K_put=825.0, K_call=940.0,
    market_premium_put=23.30,   market_premium_call=20.76,
    market_iv_put=0.444,        market_iv_call=0.440,
    market_delta_put=-0.303,    market_delta_call=+0.309,
    expiry_label="2026-05-29",
    ticker_prior_ccgr_pct=10.0,
    n_paths=1000, seed=20260429,
    fig_subdir="lly",
)

hspec = HestonSpec()

result = run_short_scenario(spec, hspec;
                            nn_cache_path=NN_CACHE,
                            port_path=PORT_PATH,
                            ladder_dir=LADDER_DIR,
                            sim_cache_path=SIM_CACHE,
                            resim=RESIM)

render_scenario_figures(result, spec; plot_dir=PLOT_DIR)
print_summary(result)
