"""
Diagnostic: why is the median LLY share-price path drifting up so fast?

Loads the pretrained portfolio's LLY JumpHMM marginal and prints:

  1. Per-state emission means and dispersions (in %/yr CCGR units).
  2. Stationary distribution → unconditional drift implied by the model.
  3. Jump kernel parameters (ε, λ, p_neg, N_tail) and the unconditional
     state-occupation tilt induced by jumps.
  4. The empirical annualized drift of obs and of the reconstructed price
     paths over a 30-day forward simulation, compared to (2).
  5. The empirical drift of the calibration ladder LLY history (closes).

If (2) is ≫ realistic LLY drift (~6-10%/yr), the bug is upstream in the
calibration / state-mean estimates. If (2) is fine but the empirical
forward drift on the simulator is much higher, the issue is in the jump
kernel or the simulator path. If both are reasonable but the median-path
geometric drift is still ~36%/yr (which is what the figure shows), the
bug is in `prices_from_growth_rates` or in the reconstruction branch
inside lly_short_premium_simulation.jl.

Run:
    julia --project=. examples/diagnose_lly_drift.jl
"""

using CSV
using DataFrames
using JLD2
using JumpHMM
using Printf
using Random
using Statistics

const TICKER    = "LLY"
const T_DAYS    = 31
const N_PATHS   = 1000
const SEED      = 20260429
const PORT_PATH = joinpath(@__DIR__, "..", "data", "pretrained-portfolio-surrogate.jld2")
const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladder")

isfile(PORT_PATH) || error("Pretrained portfolio model missing at $(PORT_PATH).")

println("="^78)
println("  LLY DRIFT DIAGNOSTIC")
println("="^78)

# ----------------------------------------------------------------------------
# 1. Load the LLY marginal
# ----------------------------------------------------------------------------
println("\n[1] Loading LLY JumpHMM marginal from $(basename(PORT_PATH)) ...")
portfolio = JLD2.load(PORT_PATH)
lly_model = portfolio["marginals"][TICKER]

@printf("    rf  = %.4f /yr   (the constant subtracted in excess_growth_rates)\n", lly_model.rf)
@printf("    dt  = %.6f      (= 1/%.0f)\n", lly_model.dt, 1/lly_model.dt)
@printf("    N   = %d states\n", lly_model.partition.N)

# ----------------------------------------------------------------------------
# 2. Per-state emission means + stationary distribution → unconditional drift
# ----------------------------------------------------------------------------
println("\n[2] Per-state emission summary (G is annualized excess CCGR):")
println("    state    π_stat       mean_G(%/yr)   std_G(%/yr)")
state_means = Float64[]
state_stds  = Float64[]
for s in 1:lly_model.partition.N
    em = lly_model.emissions[s]
    # Try mean(em); fallback to sampling if not defined
    μ_s = try
        mean(em)
    catch
        # sample it
        rng = MersenneTwister(123 + s)
        xs = [JumpHMM.sample_emission(em) for _ in 1:50_000]
        mean(xs)
    end
    σ_s = try
        std(em)
    catch
        rng = MersenneTwister(456 + s)
        xs = [JumpHMM.sample_emission(em) for _ in 1:50_000]
        std(xs)
    end
    push!(state_means, μ_s)
    push!(state_stds, σ_s)
    @printf("    %5d   %8.4f    %+12.2f    %12.2f\n",
            s, lly_model.stationary[s], μ_s*100, σ_s*100)
end

uncond_mean_G = sum(lly_model.stationary .* state_means)
@printf("\n    Unconditional E[G]  = %+.4f /yr   (= %.2f%%/yr CCGR, *excess* of rf)\n",
        uncond_mean_G, uncond_mean_G*100)
@printf("    Implied total drift = %+.2f%%/yr  (excess + rf)\n",
        (uncond_mean_G + lly_model.rf)*100)

# ----------------------------------------------------------------------------
# 3. Jump kernel
# ----------------------------------------------------------------------------
println("\n[3] Jump kernel:")
@printf("    ϵ      = %.4f   (probability per step of *triggering* a jump epoch)\n",
        lly_model.jump.ϵ)
@printf("    λ      = %.4f   (Poisson mean cluster size when triggered)\n",
        lly_model.jump.λ)
@printf("    p_neg  = %.4f   (within a jump, fraction routed to BOTTOM tail)\n",
        lly_model.jump.p_neg)
@printf("    N_tail = %d        (number of bottom/top states forming each tail)\n",
        lly_model.jump.N_tail)

# Mean state-mean during a jump step:
N = lly_model.partition.N
N_tail = lly_model.jump.N_tail
bottom_states = 1:min(N_tail, N)
top_states    = max(1, N - N_tail + 1):N
mean_bottom = mean(state_means[bottom_states])
mean_top    = mean(state_means[top_states])
mean_jump   = lly_model.jump.p_neg * mean_bottom + (1 - lly_model.jump.p_neg) * mean_top
@printf("    Mean G inside jump epoch = %+.2f%%/yr   (bottom %.2f%%, top %.2f%%)\n",
        mean_jump*100, mean_bottom*100, mean_top*100)

# Crude unconditional drift accounting for jumps. Inside an "epoch" the
# expected number of jump-steps is λ (Poisson(λ)) — but if K=0 the simulator
# falls back to a normal transition. So expected steps per trigger = λ, and
# per-trigger contribution to drift is λ·mean_jump. Triggers happen with rate
# ϵ per step. Probability that a given step came from a jump is ≈ ϵ·λ
# (approximation; the simulator advances `t` by K when K>0).
p_jump_step_approx = lly_model.jump.ϵ * lly_model.jump.λ
uncond_with_jumps  = (1 - p_jump_step_approx) * uncond_mean_G + p_jump_step_approx * mean_jump
@printf("    Crude drift incl. jumps ≈ %+.2f%%/yr   (p_jump_step ≈ %.4f)\n",
        uncond_with_jumps*100, p_jump_step_approx)

# ----------------------------------------------------------------------------
# 4. Empirical: simulate, compute mean obs and median terminal price drift
# ----------------------------------------------------------------------------
println("\n[4] Empirical forward simulation ($N_PATHS paths × $T_DAYS days):")
sim = JumpHMM.simulate(lly_model, T_DAYS; n_paths=N_PATHS, seed=SEED)

all_obs = vcat([sim.paths[p].observations for p in 1:N_PATHS]...)
emp_mean_G = mean(all_obs)
@printf("    Empirical E[G] across all (path, step):  %+.2f%%/yr  (model says %+.2f%%/yr)\n",
        emp_mean_G*100, uncond_mean_G*100)
@printf("    Empirical std(G):                         %.2f%%/yr\n", std(all_obs)*100)

# Reconstruct prices and compute median drift in CCGR units
S_0 = 873.83  # from the figure caption; not load-bearing for the drift calc
S_paths = Array{Float64}(undef, T_DAYS + 1, N_PATHS)
S_paths[1, :] .= S_0
for p in 1:N_PATHS
    prices = JumpHMM.prices_from_growth_rates(sim.paths[p].observations, S_0;
                                               rf=lly_model.rf, dt=lly_model.dt)
    S_paths[2:end, p] .= prices[1:T_DAYS]
end

S_T = S_paths[end, :]
S_T_median = median(S_T)
S_T_mean   = mean(S_T)
horizon_yrs = T_DAYS * lly_model.dt
median_drift_per_yr = log(S_T_median / S_0) / horizon_yrs
mean_drift_per_yr   = log(S_T_mean   / S_0) / horizon_yrs

@printf("    Median S_T = \$%.2f   →  median CCGR drift = %+.2f%%/yr\n",
        S_T_median, median_drift_per_yr*100)
@printf("    Mean   S_T = \$%.2f   →    mean CCGR drift = %+.2f%%/yr\n",
        S_T_mean, mean_drift_per_yr*100)
@printf("    Itô gap (mean − median) = %+.2f%%/yr   (≈ ½σ² where σ ≈ %.1f%%/yr)\n",
        (mean_drift_per_yr - median_drift_per_yr)*100, std(all_obs)*100)

# ----------------------------------------------------------------------------
# 5. Empirical drift of LLY in the OHLC equity calibration files
# ----------------------------------------------------------------------------
println("\n[5] Empirical drift of LLY in the OHLC calibration files:")
const EQUITY_2014 = joinpath(@__DIR__, "..", "data", "equity",
                              "SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2")
const EQUITY_2025 = joinpath(@__DIR__, "..", "data", "equity",
                              "SP500-Daily-OHLC-1-2-2025-to-12-31-2025.jld2")

function lly_log_rets(path::String)
    isfile(path) || return Float64[], "(missing)"
    d = JLD2.load(path)
    ds = d["dataset"]
    haskey(ds, TICKER) || return Float64[], "(no LLY)"
    df = ds[TICKER]
    closes = Float64.(df.close)
    return diff(log.(closes)), @sprintf("n=%d closes", length(closes))
end

for (label, path) in (("2014–2024", EQUITY_2014), ("2025", EQUITY_2025))
    rets, info = lly_log_rets(path)
    if isempty(rets)
        @printf("    %-10s  %s\n", label, info)
    else
        @printf("    %-10s  %s   CCGR = %+.2f%%/yr   realized σ = %.2f%%/yr\n",
                label, info,
                mean(rets) / lly_model.dt * 100,
                std(rets)  / sqrt(lly_model.dt) * 100)
    end
end

println("\n" * "="^78)
println("  INTERPRETATION KEY")
println("="^78)
println("""
  • If [2] unconditional drift ≫ [5] empirical drift → calibration-window
    bias baked into the per-state means.
  • If [2] is fine but [3] mean_jump tilts strongly positive AND p_jump_step
    is non-negligible → jump kernel is the inflation mechanism.
  • If [2] and [3] both look reasonable but [4] median_drift_per_yr is much
    larger than [2] → bug is in the reconstruction (Itô gap too large /
    units mismatch / dt mismatch in lly_short_premium_simulation.jl).
  • The Itô gap printed in [4] should equal ≈ ½·σ²; if larger, dispersion
    is being double-counted somewhere.
""")
