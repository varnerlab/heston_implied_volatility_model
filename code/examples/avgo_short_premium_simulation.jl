"""
Short-premium scenario study for AVGO.

Mirror of `lly_short_premium_simulation.jl` retargeted to AVGO. Same
generator (JumpHMM marginal → drift-anchored CCGR → Heston CIR variance
with leverage), same Leisen–Reimer American pricer, same five-figure
output set. Only the underlying-specific constants (ticker, contract
strikes, market mids/IVs, drift prior) and output filenames differ.

Run:
    julia --project=. examples/avgo_short_premium_simulation.jl
    julia --project=. examples/avgo_short_premium_simulation.jl --resim
"""

using CSV
using Dates
using DataFrames
using Distributions
using Flux
using JLD2
using JumpHMM
using Plots
using Plots.PlotMeasures
using Printf
using Random
using Statistics

# ============================================================================
# Leisen–Reimer American option pricer (Peizer–Pratt method 2 inversion).
#
# Why not CRR? The recombining CRR lattice has discrete node positions, so
# bumping σ or S in finite-difference Greek calculations causes the strike
# K to snap across node boundaries. The resulting V(σ) and V(S) are
# piecewise-bilinear with kinks, which produces the staircase/diamond-grid
# aliasing visible in CRR-FD Vega. LR constructs the up/down probabilities
# from a Peizer–Pratt inversion of the Black–Scholes d_1, d_2 so that K
# falls at a fixed lattice position by design at every (S, σ) configuration.
# Greeks are smooth via FD at modest N (LR converges as O(1/N²) vs CRR's
# oscillatory O(1/√N), so N≈201 here matches CRR accuracy at N≈1500).
# Requires N odd; we bump even N up by one.
# ============================================================================
function _peizer_pratt(z::Float64, n::Int)
    arg = (z / (n + 1/3 + 0.1/(n+1)))^2 * (n + 1/6)
    return 0.5 + sign(z) * sqrt(0.25 - 0.25 * exp(-arg))
end

function lr_american_price(S::Float64, K::Float64, σ::Float64,
                            r::Float64, T::Float64, N::Int,
                            otype::Symbol; q::Float64=0.0)
    N % 2 == 0 && (N += 1)
    Δt    = T / N
    disc  = exp(-r * Δt)
    σsqrtT = σ * sqrt(T)
    d1 = (log(S/K) + (r - q + 0.5*σ^2)*T) / σsqrtT
    d2 = d1 - σsqrtT
    p_bar = _peizer_pratt(d2, N)     # martingale (lattice) probability
    p     = _peizer_pratt(d1, N)
    R_over_Q = exp((r - q) * Δt)
    u = R_over_Q * p / p_bar
    d = R_over_Q * (1 - p) / (1 - p_bar)

    V = Vector{Float64}(undef, N + 1)
    @inbounds for j in 0:N
        S_T = S * u^j * d^(N-j)
        V[j+1] = otype === :call ? max(S_T - K, 0.0) : max(K - S_T, 0.0)
    end
    @inbounds for n in (N-1):-1:0
        for j in 0:n
            S_n  = S * u^j * d^(n-j)
            cont = disc * (p_bar * V[j+2] + (1 - p_bar) * V[j+1])
            intrinsic = otype === :call ? S_n - K : K - S_n
            V[j+1] = max(cont, intrinsic)
        end
    end
    return V[1]
end

# ============================================================================
# Constants & hyperparameters
# ============================================================================

const TICKER       = "AVGO"
const T_DAYS       = 31           # matches the real 2026-05-29 expiry from the 04-28 capture
const N_PATHS      = 1000
const N_STEPS_LR   = 201          # LR with N≈200 is more accurate than CRR with N≈1500
const SEED         = 20260429
const R_FREE       = 0.0425
const Q_DIV        = 0.0

# Long-run drift anchor for the JumpHMM marginal. The trained AVGO model
# carries a +34.5%/yr unconditional CCGR baked in from the 2014–2024 AI/
# semiconductor cycle; projecting that forward inflates the path bundle
# unrealistically. We additively shift every per-step observation in CCGR
# space so the simulated unconditional mean equals AVGO_PRIOR_CCGR_PCT.
# ~12%/yr is appropriate for a high-β tech compounder
# (rf 4.25% + ~1.3·5pp tech-ERP + ~1pp alpha).
const AVGO_PRIOR_CCGR_PCT = 12.0

# Real AVGO contracts pulled from the 04-28-2026 capture (closest to 30 DTE / 30Δ):
#   put : AVGO 2026-05-29 K=$375  Δ = -0.292   bid/ask =  9.97/11.43  mid = $10.70  IV = 46.6%
#   call: AVGO 2026-05-29 K=$435  Δ = +0.289   bid/ask =  8.14/ 9.85  mid =  $9.00  IV = 45.8%
const EXPIRY              = "2026-05-29"
const K_PUT               = 375.0
const K_CALL              = 435.0
const MARKET_PREMIUM_PUT  = 10.70
const MARKET_PREMIUM_CALL = 9.00
const MARKET_IV_PUT       = 0.466
const MARKET_IV_CALL      = 0.458
const MARKET_DELTA_PUT    = -0.292
const MARKET_DELTA_CALL   = +0.289

# Heston (CIR) variance dynamics
const HESTON_KAPPA   = 2.0    # mean-reversion speed (~0.5y half-life)
const HESTON_SIGMA_V = 0.5    # vol of vol
const HESTON_RHO     = -0.6   # leverage: dW_v ~ ρ·dW_S + √(1-ρ²)·dW_v_indep

# Paths
const LADDER_DIR     = joinpath(@__DIR__, "..", "data", "ladder")
const FIG_CACHE_DIR  = joinpath(@__DIR__, "..", "figures")    # cached NN + sim artifacts
const PLOT_DIR       = joinpath(@__DIR__, "..", "..", "paper-jcf", "sections", "figures", "avgo")
const NN_CACHE       = joinpath(FIG_CACHE_DIR, "calibrate_ladders_per_ticker_nn_cache.jld2")
const SIM_CACHE      = joinpath(FIG_CACHE_DIR, "avgo_short_premium_simulation_cache.jld2")
const PORT_PATH      = joinpath(@__DIR__, "..", "data", "pretrained-portfolio-surrogate.jld2")

const RESIM = "--resim" in ARGS

const SECTORS = Dict(
    "AAPL" => "Tech", "AMD" => "Tech", "AVGO" => "Tech", "GOOG" => "Tech",
    "INTC" => "Tech", "META" => "Tech", "MSFT" => "Tech", "MU" => "Tech",
    "NVDA" => "Tech", "QCOM" => "Tech",
    "BAC" => "Financials", "GS" => "Financials", "JPM" => "Financials",
    "WFC" => "Financials",
    "CVX" => "Energy", "OXY" => "Energy", "XOM" => "Energy",
    "ABBV" => "Healthcare", "AMGN" => "Healthcare", "BMY" => "Healthcare",
    "JNJ" => "Healthcare", "LLY" => "Healthcare", "MRNA" => "Healthcare",
    "PFE" => "Healthcare", "UNH" => "Healthcare",
    "TGT" => "Retail", "UPS" => "Retail", "WMT" => "Retail",
    "IWM" => "ETF", "QQQ" => "ETF", "SPY" => "ETF",
)

isfile(NN_CACHE) || error("NN cache missing at $(NN_CACHE). Run calibrate_ladders_per_ticker_nn.jl --retrain first.")
isfile(PORT_PATH) || error("Pretrained portfolio model missing at $(PORT_PATH).")

# ============================================================================
# Step 1: Load ladder corpus → standardisation + ticker spot
# ============================================================================

function load_ladder(filepath)
    df = CSV.read(filepath, DataFrame)
    ticker = string(df.underlying[1])
    S = df.und_close[1]
    df[!, :ticker] .= ticker
    df[!, :S] .= S
    df[!, :moneyness] = df.strike ./ S
    valid = df[
        .!ismissing.(df.implied_vol) .&
        .!isnan.(coalesce.(df.implied_vol, NaN)) .&
        (coalesce.(df.implied_vol, 0.0) .> 0.01) .&
        (coalesce.(df.implied_vol, 999.0) .< 2.0) .&
        (df.bid .> 0) .&
        (df.moneyness .>= 0.80) .&
        (df.moneyness .<= 1.20) .&
        (df.actual_dte .> 0),
    :]
    return valid
end

function load_all_ladders(dir)
    files = String[]
    for (root, _, fs) in walkdir(dir)
        occursin("VIX-data", root) && continue
        for f in fs
            endswith(f, ".csv") && occursin("_dte_ladder_", f) &&
                push!(files, joinpath(root, f))
        end
    end
    frames = DataFrame[]
    for f in files
        df = load_ladder(f)
        nrow(df) > 0 && push!(frames, df)
    end
    return vcat(frames...)
end

println("Loading ladder corpus for standardisation constants...")
all_data = load_all_ladders(LADDER_DIR)
const MU_DTE    = mean(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const SIGMA_DTE = std(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const MU_M      = mean(log.(Float64.(all_data.moneyness)))
const SIGMA_M   = std(log.(Float64.(all_data.moneyness)))

# Anchor S_0 to the 04-28-2026 capture, since the contract strikes, premiums,
# IVs, and deltas hardcoded above were all observed on that date. Pulling the
# "latest" spot would drift the scenario off the contract design.
const ANCHOR_DATE = Date("2026-04-28")
ticker_slice = all_data[all_data.ticker .== TICKER, :]
anchor_rows = ticker_slice[ticker_slice.und_session_date .== ANCHOR_DATE, :]
isempty(anchor_rows) && error("No $TICKER rows for anchor date $ANCHOR_DATE in the ladder corpus.")
S_0 = Float64(anchor_rows.S[1])
@printf("  S_0 (%s, anchored to %s) = \$%.2f\n", TICKER, ANCHOR_DATE, S_0)

# ============================================================================
# Step 2: Restore per-ticker NN ψ surface for the underlying
# ============================================================================

build_psi_nn(n_obs::Integer, threshold::Integer) = n_obs >= threshold ?
    Chain(Dense(2 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 1)) :
    Chain(Dense(2 => 8,  tanh), Dense(8  => 8,  tanh), Dense(8  => 1))

println("Loading per-ticker NN ψ surface for $TICKER ...")
nn_cache = JLD2.load(NN_CACHE)
const PT_PAYLOAD = nn_cache["per_ticker_payload"][TICKER]
const PSI_NN     = build_psi_nn(PT_PAYLOAD.n_obs, 5000)
Flux.loadmodel!(PSI_NN, PT_PAYLOAD.state)
const LOG_THETA  = Float64(PT_PAYLOAD.log_theta[1])
const THETA_TICKER  = exp(LOG_THETA)
@printf("  θ̄ (%s long-term variance) = %.4f  →  IV %.1f%%\n", TICKER, THETA_TICKER, sqrt(THETA_TICKER)*100)

"""ψ_NN((K/S)_std, (DTE_days)_std) → ψ scalar (positive)"""
function psi_ticker(K::Float64, S::Float64, dte_days::Real)
    log_dte_s = Float32((log(max(Float64(dte_days), 1.0)) - MU_DTE) / SIGMA_DTE)
    log_m_s   = Float32((log(K / S) - MU_M) / SIGMA_M)
    x = reshape(Float32[log_dte_s, log_m_s], 2, 1)
    log_psi = first(PSI_NN(x))
    return Float64(exp(log_psi))
end

"""IV implied by the calibrated NN at (K, S, DTE)."""
nn_iv(K, S, dte) = sqrt(max(THETA_TICKER * psi_ticker(K, S, dte), 1e-10))

"""IV given a stochastic level v and the NN cross-section shape."""
heston_smile_iv(v, K, S, dte) = sqrt(max(v * psi_ticker(K, S, dte), 1e-10))

# ============================================================================
# Step 3: Report the real ticker contracts being simulated
# ============================================================================

@printf("  Real-contract setup (%s %s, %d DTE):\n", TICKER, EXPIRY, T_DAYS)
@printf("    Put  K=\$%.2f   market Δ=%+0.3f   mid=\$%.2f   market IV=%.1f%%   (model NN-IV=%.1f%%)\n",
        K_PUT,  MARKET_DELTA_PUT,  MARKET_PREMIUM_PUT,  MARKET_IV_PUT*100,
        nn_iv(K_PUT,  S_0, T_DAYS)*100)
@printf("    Call K=\$%.2f   market Δ=%+0.3f   mid=\$%.2f   market IV=%.1f%%   (model NN-IV=%.1f%%)\n",
        K_CALL, MARKET_DELTA_CALL, MARKET_PREMIUM_CALL, MARKET_IV_CALL*100,
        nn_iv(K_CALL, S_0, T_DAYS)*100)

# ============================================================================
# Step 4: Simulate price + variance + premium paths (with caching)
# ============================================================================

function simulate_all()
    println("\nLoading pretrained portfolio model...")
    portfolio = JLD2.load(PORT_PATH)
    ticker_model = portfolio["marginals"][TICKER]

    println("Simulating $N_PATHS $TICKER price paths over $T_DAYS days...")
    sim = JumpHMM.simulate(ticker_model, T_DAYS; n_paths=N_PATHS, seed=SEED)
    n_actual = length(sim.paths)

    # Anchor the projection drift to a documented long-run prior.
    target_drift = AVGO_PRIOR_CCGR_PCT / 100.0
    empirical_drift = mean(vcat([sim.paths[p].observations for p in 1:n_actual]...))
    drift_shift = target_drift - empirical_drift
    @printf("  Drift anchor: empirical %+.2f%%/yr → target %+.2f%%/yr   (shift %+.2f%%/yr)\n",
            empirical_drift*100, target_drift*100, drift_shift*100)
    for p in 1:n_actual
        sim.paths[p].observations .+= drift_shift
    end

    # Reconstruct prices: time index 0..T_DAYS, so length T_DAYS+1
    S_paths = Array{Float64}(undef, T_DAYS + 1, n_actual)
    S_paths[1, :] .= S_0
    for p in 1:n_actual
        prices = JumpHMM.prices_from_growth_rates(sim.paths[p].observations,
                                                  S_0; rf=ticker_model.rf,
                                                  dt=ticker_model.dt)
        # JumpHMM returns one entry per observation step; place at t=1..T_DAYS.
        S_paths[2:end, p] .= prices[1:T_DAYS]
    end

    println("Evolving CIR variance with leverage coupling (κ=$HESTON_KAPPA, σ_v=$HESTON_SIGMA_V, ρ=$HESTON_RHO)...")
    v_paths = Array{Float64}(undef, T_DAYS + 1, n_actual)
    v_paths[1, :] .= THETA_TICKER
    rng = MersenneTwister(SEED + 1)
    Δt = 1.0 / 365.0
    sqrtΔt = sqrt(Δt)
    rho2 = sqrt(1.0 - HESTON_RHO^2)
    for p in 1:n_actual
        for t in 2:(T_DAYS + 1)
            # Standardise the realized log-return to extract Z_S
            Δlogs = log(S_paths[t, p] / S_paths[t-1, p])
            σ_prev = sqrt(max(v_paths[t-1, p], 1e-8))
            Z_S = (Δlogs - R_FREE * Δt) / (σ_prev * sqrtΔt + 1e-12)
            Z_S = clamp(Z_S, -6.0, 6.0)  # tame jump-induced extremes
            Z_v_indep = randn(rng)
            Z_v = HESTON_RHO * Z_S + rho2 * Z_v_indep
            v_prev = v_paths[t-1, p]
            dv = HESTON_KAPPA * (THETA_TICKER - v_prev) * Δt +
                 HESTON_SIGMA_V * sqrt(max(v_prev, 1e-10)) * sqrtΔt * Z_v
            v_paths[t, p] = max(v_prev + dv, 1e-10)
        end
    end

    println("Pricing put + call at every (t, path) via Leisen–Reimer American (n_steps=$N_STEPS_LR)...")
    V_put  = Array{Float64}(undef, T_DAYS + 1, n_actual)
    V_call = Array{Float64}(undef, T_DAYS + 1, n_actual)
    for p in 1:n_actual
        for t in 0:T_DAYS
            S_t = S_paths[t+1, p]
            v_t = v_paths[t+1, p]
            if t == T_DAYS
                V_put[t+1, p]  = max(K_PUT  - S_t, 0.0)
                V_call[t+1, p] = max(S_t - K_CALL, 0.0)
            else
                dte_remaining = T_DAYS - t
                T_rem_yrs = dte_remaining / 365.0
                σ_put  = heston_smile_iv(v_t, K_PUT,  S_t, dte_remaining)
                σ_call = heston_smile_iv(v_t, K_CALL, S_t, dte_remaining)
                V_put[t+1, p]  = lr_american_price(S_t, K_PUT,  σ_put,
                                                    R_FREE, T_rem_yrs, N_STEPS_LR,
                                                    :put;  q=Q_DIV)
                V_call[t+1, p] = lr_american_price(S_t, K_CALL, σ_call,
                                                    R_FREE, T_rem_yrs, N_STEPS_LR,
                                                    :call; q=Q_DIV)
            end
        end
        if p % 100 == 0 || p == n_actual
            @printf("    %4d / %d paths priced\n", p, n_actual)
        end
    end

    return (S_paths=S_paths, v_paths=v_paths,
            V_put=V_put, V_call=V_call,
            n_actual=n_actual)
end

function load_or_simulate()
    if !RESIM && isfile(SIM_CACHE)
        cache = JLD2.load(SIM_CACHE)
        cache_S0    = get(cache, "S_0", NaN)
        cache_kput  = get(cache, "K_put", NaN)
        cache_kcal  = get(cache, "K_call", NaN)
        cache_prior = get(cache, "avgo_prior_ccgr_pct", NaN)
        if cache_S0 ≈ S_0 && cache_kput ≈ K_PUT && cache_kcal ≈ K_CALL &&
           cache_prior ≈ AVGO_PRIOR_CCGR_PCT
            println("Cache hit: loading prior simulation from $(basename(SIM_CACHE))")
            return (S_paths=cache["S_paths"],
                    v_paths=cache["v_paths"],
                    V_put=cache["V_put"],
                    V_call=cache["V_call"],
                    n_actual=size(cache["S_paths"], 2))
        else
            println("Cache parameters drifted from current spot/strikes — resimulating.")
        end
    end
    art = simulate_all()
    JLD2.jldsave(SIM_CACHE;
        S_paths=art.S_paths, v_paths=art.v_paths,
        V_put=art.V_put, V_call=art.V_call,
        S_0=S_0, K_put=K_PUT, K_call=K_CALL,
        theta_ticker=THETA_TICKER,
        heston_kappa=HESTON_KAPPA, heston_sigma_v=HESTON_SIGMA_V, heston_rho=HESTON_RHO,
        avgo_prior_ccgr_pct=AVGO_PRIOR_CCGR_PCT,
        seed=SEED)
    println("[cache] saved -> $(SIM_CACHE)")
    return art
end

art = load_or_simulate()
S_paths = art.S_paths
v_paths = art.v_paths
V_put   = art.V_put
V_call  = art.V_call
n_paths_actual = art.n_actual

# Entry premium = real market mid. The model's t=0 fair value is reported
# alongside so we can see the entry-edge gap (model − market mid).
const PREMIUM_PUT_T0  = MARKET_PREMIUM_PUT
const PREMIUM_CALL_T0 = MARKET_PREMIUM_CALL
const MODEL_FV_PUT_T0  = V_put[1, 1]
const MODEL_FV_CALL_T0 = V_call[1, 1]
@printf("  Premium received at sale (market mid):  put = \$%.2f   call = \$%.2f\n",
        PREMIUM_PUT_T0, PREMIUM_CALL_T0)
@printf("  Model t=0 fair value (CRR @ NN-IV):     put = \$%.2f   call = \$%.2f\n",
        MODEL_FV_PUT_T0, MODEL_FV_CALL_T0)
@printf("  Entry edge (model − market):            put = \$%+.2f   call = \$%+.2f\n",
        MODEL_FV_PUT_T0 - PREMIUM_PUT_T0, MODEL_FV_CALL_T0 - PREMIUM_CALL_T0)

# Tail bins by terminal price
S_terminal   = S_paths[end, :]
n_tail       = max(1, round(Int, 0.05 * n_paths_actual))
worst_idx    = sortperm(S_terminal)[1:n_tail]              # bottom 5%
best_idx     = sortperm(S_terminal; rev=true)[1:n_tail]    # top    5%
@printf("  Tail bin size: %d of %d paths (5%%)\n", n_tail, n_paths_actual)
@printf("  Worst-5%%  terminal-price range: \$%.2f .. \$%.2f\n",
        minimum(S_terminal[worst_idx]), maximum(S_terminal[worst_idx]))
@printf("  Top-5%%    terminal-price range: \$%.2f .. \$%.2f\n",
        minimum(S_terminal[best_idx]),  maximum(S_terminal[best_idx]))

# Display-only overlays: drop the extreme top/bottom 1% so the figures aren't
# dominated by a handful of spike outliers. Underlying stats still use the
# full worst_idx / best_idx.
n_extreme = max(1, round(Int, 0.01 * n_paths_actual))
best_display_idx  = sortperm(S_terminal; rev=true)[(n_extreme+1):n_tail]
worst_display_idx = sortperm(S_terminal)[(n_extreme+1):n_tail]

# Short-position P&L at terminal: keep the premium minus what the option pays out
pnl_put  = PREMIUM_PUT_T0  .- V_put[end,  :]
pnl_call = PREMIUM_CALL_T0 .- V_call[end, :]

# ============================================================================
# Step 5: Figures
# ============================================================================

const COL_PATH    = RGBA(0.30, 0.30, 0.30, 0.06)
const COL_MEDIAN  = RGB(0.00, 0.00, 0.00)
const COL_WORST   = RGBA(0.78, 0.20, 0.20, 0.45)
const COL_BEST    = RGBA(0.20, 0.55, 0.30, 0.45)
const COL_PREMIUM = RGBA(0.30, 0.40, 0.85, 0.55)

t_axis = collect(0:T_DAYS)

function path_panel(t_axis, paths_mat, title_str, ylabel_str;
                    tails,                    # Vector of (idx=, color=, label=) NamedTuples
                    legend_pos = :topleft,
                    overlay_zero = false)
    p = plot(legend = legend_pos,
             legendfontsize = 9,
             titlefontsize  = 12,
             guidefontsize  = 11,
             tickfontsize   = 10,
             xlabel = "Days from sale",
             ylabel = ylabel_str,
             title  = title_str,
             framestyle = :box,
             grid = true, gridalpha = 0.25,
             foreground_color_grid = :gray,
             background_color = :white,
             background_color_legend  = RGBA(1.0, 1.0, 1.0, 0.92),
             foreground_color_legend  = RGB(0.55, 0.55, 0.55),
             left_margin = 9mm, right_margin = 4mm,
             top_margin = 4mm, bottom_margin = 6mm)

    # All paths in faint gray
    for j in 1:size(paths_mat, 2)
        plot!(p, t_axis, paths_mat[:, j], color = COL_PATH, lw = 0.4, label = "")
    end

    # Tail-binned overlays (each layer = one bin: top or bottom)
    for tail in tails
        for (k, j) in enumerate(tail.idx)
            plot!(p, t_axis, paths_mat[:, j], color = tail.color, lw = 1.0,
                  label = (k == 1 ? tail.label : ""))
        end
    end

    # Median + IQR
    med = [median(paths_mat[t, :]) for t in 1:size(paths_mat, 1)]
    q25 = [quantile(paths_mat[t, :], 0.25) for t in 1:size(paths_mat, 1)]
    q75 = [quantile(paths_mat[t, :], 0.75) for t in 1:size(paths_mat, 1)]
    plot!(p, t_axis, med, ribbon = (med .- q25, q75 .- med),
          color = COL_MEDIAN, fillalpha = 0.12, lw = 2.2,
          label = "Median  (band: 25–75%)")

    overlay_zero && hline!(p, [0.0], color = :gray, ls = :dash, alpha = 0.5, label = "")
    return p
end

# Both tails (worst 1–5% red + top 1–5% green): every path figure shows the
# full dispersion picture so the reader can see what drives short-PnL on
# either side of the trade.
both_tails = [
    (idx = worst_display_idx, color = COL_WORST,
     label = "Worst 1–5% by terminal price"),
    (idx = best_display_idx,  color = COL_BEST,
     label = "Top 1–5% by terminal price"),
]

# --- Figure A: stock path bundle + short-put premium bundle (worst-5% red) ---
println("\nRendering Figure A — stock + put premium ...")
title_share = @sprintf("%s share price paths   (S₀ = \$%.2f)", TICKER, S_0)
pA1 = path_panel(t_axis, S_paths, title_share, "Share price (\$)";
                 tails = both_tails,
                 legend_pos = :topleft)
hline!(pA1, [K_PUT], color = RGB(0.85, 0.65, 0.13), ls = :dash, lw = 3.5, alpha = 1.0,
       label = @sprintf("30Δ put strike  K = \$%.2f", K_PUT))

pA2 = path_panel(t_axis, V_put,
                 "Short 30-day 30Δ put — option price",
                 "Put price (\$)";
                 tails = both_tails,
                 legend_pos = :topleft)
hline!(pA2, [PREMIUM_PUT_T0], color = COL_PREMIUM, ls = :dash, lw = 1.5,
       label = @sprintf("Premium received  \$%.2f", PREMIUM_PUT_T0))

p_A = plot(pA1, pA2, layout = (1, 2), size = (1500, 600), dpi = 220,
           left_margin = 9mm, right_margin = 4mm,
           bottom_margin = 7mm, top_margin = 4mm)
mkpath(PLOT_DIR)
out_A_pdf = joinpath(PLOT_DIR, "avgo_short_put_paths.pdf")
out_A_png = joinpath(PLOT_DIR, "avgo_short_put_paths.png")
savefig(p_A, out_A_pdf); savefig(p_A, out_A_png)

# --- Figure B: stock path bundle + short-call premium bundle (top-5% green) ---
println("Rendering Figure B — stock + call premium ...")
pB1 = path_panel(t_axis, S_paths, title_share, "Share price (\$)";
                 tails = both_tails,
                 legend_pos = :topleft)
hline!(pB1, [K_CALL], color = RGB(0.85, 0.65, 0.13), ls = :dash, lw = 3.5, alpha = 1.0,
       label = @sprintf("30Δ call strike  K = \$%.2f", K_CALL))

pB2 = path_panel(t_axis, V_call,
                 "Short 30-day 30Δ call — option price",
                 "Call price (\$)";
                 tails = both_tails,
                 legend_pos = :topleft)
hline!(pB2, [PREMIUM_CALL_T0], color = COL_PREMIUM, ls = :dash, lw = 1.5,
       label = @sprintf("Premium received  \$%.2f", PREMIUM_CALL_T0))

p_B = plot(pB1, pB2, layout = (1, 2), size = (1500, 600), dpi = 220,
           left_margin = 9mm, right_margin = 4mm,
           bottom_margin = 7mm, top_margin = 4mm)
out_B_pdf = joinpath(PLOT_DIR, "avgo_short_call_paths.pdf")
out_B_png = joinpath(PLOT_DIR, "avgo_short_call_paths.png")
savefig(p_B, out_B_pdf); savefig(p_B, out_B_png)

# --- Figure C: terminal short P&L histograms ---
println("Rendering Figure C — terminal short P&L distributions ...")
function pnl_panel(pnl_vec, premium_t0, title_str)
    q05    = quantile(pnl_vec, 0.05)
    q50    = median(pnl_vec)
    qmean  = mean(pnl_vec)
    qmin   = minimum(pnl_vec)
    pct_keep_full = 100 * mean(pnl_vec .>= premium_t0 - 1e-6)

    # Fixed-width bins from worst loss to a hair above the premium ceiling so
    # both panels share a comparable bin geometry. Bin width chosen so the
    # premium spike lands in its own bin.
    lo  = floor(qmin / 10) * 10 - 5
    hi  = ceil(premium_t0 / 5) * 5 + 5
    bin_w = max(2.5, (hi - lo) / 60)
    edges = lo:bin_w:hi

    p = histogram(pnl_vec, bins = edges, normalize = :probability,
                  color = RGBA(0.45, 0.45, 0.45, 0.85), lw = 0.2,
                  linecolor = RGBA(0.20, 0.20, 0.20, 0.85),
                  yscale = :log10, ylims = (5e-4, 1.0),
                  yticks = ([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0],
                            ["0.001", "0.003", "0.01", "0.03",
                             "0.1", "0.3", "1.0"]),
                  legend = :topleft, legendfontsize = 9,
                  titlefontsize = 12, guidefontsize = 11, tickfontsize = 10,
                  framestyle = :box, grid = true, gridalpha = 0.25,
                  foreground_color_grid = :gray, background_color = :white,
                  background_color_legend = RGBA(1.0, 1.0, 1.0, 0.92),
                  foreground_color_legend = RGB(0.55, 0.55, 0.55),
                  title  = title_str,
                  xlabel = "Terminal P&L per contract (\$)",
                  ylabel = "Fraction of paths  (log scale)",
                  left_margin = 11mm, right_margin = 4mm,
                  top_margin = 4mm, bottom_margin = 7mm,
                  label = "")
    vline!(p, [0.0],     color = :black,             ls = :solid,   lw = 0.8,
           alpha = 0.7,  label = "")
    vline!(p, [q05],     color = RGB(0.78, 0.20, 0.20), ls = :dash, lw = 1.6,
           label = @sprintf("5%% quantile  =  \$%+8.2f", q05))
    vline!(p, [q50],     color = RGB(0.10, 0.10, 0.10), ls = :solid, lw = 1.8,
           label = @sprintf("Median        =  \$%+8.2f", q50))
    vline!(p, [qmean],   color = RGB(0.10, 0.30, 0.70), ls = :dashdot, lw = 1.6,
           label = @sprintf("Mean          =  \$%+8.2f", qmean))
    vline!(p, [premium_t0], color = COL_PREMIUM,        ls = :dot,    lw = 1.6,
           label = @sprintf("Premium kept  =  \$%+8.2f  (%.1f%% of paths)",
                            premium_t0, pct_keep_full))
    return p
end

pC1 = pnl_panel(pnl_put,  PREMIUM_PUT_T0,
                "Short 30-day 30Δ put — terminal P&L distribution")
pC2 = pnl_panel(pnl_call, PREMIUM_CALL_T0,
                "Short 30-day 30Δ call — terminal P&L distribution")
p_C = plot(pC1, pC2, layout = (1, 2), size = (1500, 600), dpi = 220,
           left_margin = 11mm, right_margin = 4mm,
           bottom_margin = 7mm, top_margin = 4mm)
out_C_pdf = joinpath(PLOT_DIR, "avgo_short_pnl_distributions.pdf")
out_C_png = joinpath(PLOT_DIR, "avgo_short_pnl_distributions.png")
savefig(p_C, out_C_pdf); savefig(p_C, out_C_png)

# --- Figure D: implied-volatility trajectories at the fixed contract strikes ---
# Same loop also computes LR American Greeks (Δ, Γ, Vega) via central finite
# differences on the same pricer used for the base marks. At each (t, path)
# we re-price at (S±h, σ) and (S, σ±dσ) — 4 extra CRR runs per leg, plus the
# base mark V_put/V_call[t+1,p] reused as the centre of the Γ stencil.
const H_S_FRAC = 0.015    # spot bump: 1.5% of S_t (wider stencil → cleaner Γ)
const D_SIGMA  = 0.005    # σ bump: ±0.5 vol points (total 1% IV span for Vega)

println("\nComputing Leisen–Reimer American Greeks via finite differences (h=1.5% S, dσ=±0.5 IV pts)...")
σ_put_paths    = fill(NaN, size(S_paths))
σ_call_paths   = fill(NaN, size(S_paths))
Δ_put_paths    = fill(NaN, size(S_paths))
Δ_call_paths   = fill(NaN, size(S_paths))
Γ_put_paths    = fill(NaN, size(S_paths))
Γ_call_paths   = fill(NaN, size(S_paths))
Vega_put_paths  = fill(NaN, size(S_paths))   # $ per 1% IV move
Vega_call_paths = fill(NaN, size(S_paths))

for p in 1:n_paths_actual
    for t in 0:(T_DAYS - 1)         # IV/Greeks undefined at expiry; leave NaN
        dte = T_DAYS - t
        T_rem = dte / 365.0
        S_t = S_paths[t+1, p]
        v_t = v_paths[t+1, p]
        σ_p = heston_smile_iv(v_t, K_PUT,  S_t, dte)
        σ_c = heston_smile_iv(v_t, K_CALL, S_t, dte)
        σ_put_paths[t+1,  p] = σ_p * 100
        σ_call_paths[t+1, p] = σ_c * 100

        h    = H_S_FRAC * S_t
        σ_p_lo = max(σ_p - D_SIGMA, 1e-4)
        σ_c_lo = max(σ_c - D_SIGMA, 1e-4)

        # PUT: reuse base mark as Γ centre, bump S by ±h, σ by ±D_SIGMA
        V_pS0 = V_put[t+1, p]
        V_pSp = lr_american_price(S_t + h, K_PUT, σ_p,           R_FREE, T_rem, N_STEPS_LR, :put; q=Q_DIV)
        V_pSm = lr_american_price(S_t - h, K_PUT, σ_p,           R_FREE, T_rem, N_STEPS_LR, :put; q=Q_DIV)
        V_pσp = lr_american_price(S_t,     K_PUT, σ_p + D_SIGMA, R_FREE, T_rem, N_STEPS_LR, :put; q=Q_DIV)
        V_pσm = lr_american_price(S_t,     K_PUT, σ_p_lo,        R_FREE, T_rem, N_STEPS_LR, :put; q=Q_DIV)
        Δ_put_paths[t+1,  p]    = (V_pSp - V_pSm) / (2h)
        Γ_put_paths[t+1,  p]    = (V_pSp - 2*V_pS0 + V_pSm) / (h^2)
        Vega_put_paths[t+1, p]  = (V_pσp - V_pσm) / (2 * D_SIGMA * 100)

        # CALL
        V_cS0 = V_call[t+1, p]
        V_cSp = lr_american_price(S_t + h, K_CALL, σ_c,           R_FREE, T_rem, N_STEPS_LR, :call; q=Q_DIV)
        V_cSm = lr_american_price(S_t - h, K_CALL, σ_c,           R_FREE, T_rem, N_STEPS_LR, :call; q=Q_DIV)
        V_cσp = lr_american_price(S_t,     K_CALL, σ_c + D_SIGMA, R_FREE, T_rem, N_STEPS_LR, :call; q=Q_DIV)
        V_cσm = lr_american_price(S_t,     K_CALL, σ_c_lo,        R_FREE, T_rem, N_STEPS_LR, :call; q=Q_DIV)
        Δ_call_paths[t+1,  p]    = (V_cSp - V_cSm) / (2h)
        Γ_call_paths[t+1,  p]    = (V_cSp - 2*V_cS0 + V_cSm) / (h^2)
        Vega_call_paths[t+1, p]  = (V_cσp - V_cσm) / (2 * D_SIGMA * 100)
    end
    if p % 100 == 0 || p == n_paths_actual
        @printf("    Greeks: %4d / %d paths done\n", p, n_paths_actual)
    end
end

println("Rendering Figure D — IV trajectories at K_put and K_call ...")
t_iv = collect(0:(T_DAYS - 1))                # x-axis up to day 29 (last day with IV defined)
σ_put_plot  = σ_put_paths[1:T_DAYS,  :]
σ_call_plot = σ_call_paths[1:T_DAYS, :]

pD1 = path_panel(t_iv, σ_put_plot,
                 "Implied volatility at the put strike  (K = \$$(round(K_PUT, digits=2)))",
                 "Annualized IV (%)";
                 tails = both_tails,
                 legend_pos = :topleft)
pD2 = path_panel(t_iv, σ_call_plot,
                 "Implied volatility at the call strike  (K = \$$(round(K_CALL, digits=2)))",
                 "Annualized IV (%)";
                 tails = both_tails,
                 legend_pos = :topleft)
p_D = plot(pD1, pD2, layout = (1, 2), size = (1500, 600), dpi = 220,
           left_margin = 9mm, right_margin = 4mm,
           bottom_margin = 7mm, top_margin = 4mm)
out_D_pdf = joinpath(PLOT_DIR, "avgo_iv_trajectories.pdf")
out_D_png = joinpath(PLOT_DIR, "avgo_iv_trajectories.png")
savefig(p_D, out_D_pdf); savefig(p_D, out_D_png)

# --- Figure E: Greek trajectories (Δ, Γ, Vega) for both legs ---
println("Rendering Figure E — Greek trajectories (Δ, Γ, Vega) ...")
t_g = collect(0:(T_DAYS - 1))
Δ_put_plot    = Δ_put_paths[1:T_DAYS,    :]
Δ_call_plot   = Δ_call_paths[1:T_DAYS,   :]
Γ_put_plot    = Γ_put_paths[1:T_DAYS,    :]
Γ_call_plot   = Γ_call_paths[1:T_DAYS,   :]
Vega_put_plot  = Vega_put_paths[1:T_DAYS,  :]
Vega_call_plot = Vega_call_paths[1:T_DAYS, :]

pE1 = path_panel(t_g, Δ_put_plot,
                 "Long-put Δ at K_put",
                 "Δ (long put)";
                 tails = both_tails, legend_pos = :bottomleft)
pE2 = path_panel(t_g, Δ_call_plot,
                 "Long-call Δ at K_call",
                 "Δ (long call)";
                 tails = both_tails, legend_pos = :topleft)
pE3 = path_panel(t_g, Γ_put_plot,
                 "Γ at K_put",
                 "Γ (per share, per \$ in S)";
                 tails = both_tails, legend_pos = :topleft)
pE4 = path_panel(t_g, Γ_call_plot,
                 "Γ at K_call",
                 "Γ (per share, per \$ in S)";
                 tails = both_tails, legend_pos = :topleft)
pE5 = path_panel(t_g, Vega_put_plot,
                 "Vega at K_put",
                 "Vega (\$ per 1% IV move)";
                 tails = both_tails, legend_pos = :topright)
pE6 = path_panel(t_g, Vega_call_plot,
                 "Vega at K_call",
                 "Vega (\$ per 1% IV move)";
                 tails = both_tails, legend_pos = :topright)

p_E = plot(pE1, pE2, pE3, pE4, pE5, pE6, layout = (3, 2),
           size = (1500, 1500), dpi = 220,
           left_margin = 9mm, right_margin = 4mm,
           bottom_margin = 5mm, top_margin = 4mm)
out_E_pdf = joinpath(PLOT_DIR, "avgo_short_greeks.pdf")
out_E_png = joinpath(PLOT_DIR, "avgo_short_greeks.png")
savefig(p_E, out_E_pdf); savefig(p_E, out_E_png)

# Output goes directly to paper-jcf/sections/figures/avgo/ — no promote_figures()
# step needed. promote_figures() owns the flat code/figures/ namespace; nested
# per-ticker figures like these are written into the paper tree by the scenario
# scripts themselves, and it deliberately leaves them alone.

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^78)
println("  SHORT-PREMIUM SCENARIO SUMMARY  ($TICKER, $T_DAYS-day, 30Δ, $N_PATHS paths)")
println("="^78)
@printf("  Spot:                S_0 = \$%.2f\n", S_0)
@printf("  30Δ put strike:      K = \$%.2f   (K/S = %.3f)\n", K_PUT,  K_PUT/S_0)
@printf("  30Δ call strike:     K = \$%.2f   (K/S = %.3f)\n", K_CALL, K_CALL/S_0)
@printf("  Heston (κ, σ_v, ρ):  (%.1f, %.1f, %.2f)\n",
        HESTON_KAPPA, HESTON_SIGMA_V, HESTON_RHO)
println()
@printf("  %-12s  %-22s  %-22s\n", "stat", "Short PUT P&L (\$)", "Short CALL P&L (\$)")
println("  " * "-"^60)
for (lbl, fp, fc) in [("Premium",  PREMIUM_PUT_T0,        PREMIUM_CALL_T0),
                      ("Mean",     mean(pnl_put),         mean(pnl_call)),
                      ("Median",   median(pnl_put),       median(pnl_call)),
                      ("Std",      std(pnl_put),          std(pnl_call)),
                      ("5%-tile",  quantile(pnl_put,0.05),quantile(pnl_call,0.05)),
                      ("Min",      minimum(pnl_put),      minimum(pnl_call)),
                      ("Max",      maximum(pnl_put),      maximum(pnl_call))]
    @printf("  %-12s  %+22.2f  %+22.2f\n", lbl, fp, fc)
end
pct_keep_put  = 100 * mean(pnl_put  .>= PREMIUM_PUT_T0  - 1e-6)
pct_keep_call = 100 * mean(pnl_call .>= PREMIUM_CALL_T0 - 1e-6)
@printf("\n  Premium kept in full (PUT):   %5.1f%% of paths\n",  pct_keep_put)
@printf("  Premium kept in full (CALL):  %5.1f%% of paths\n",    pct_keep_call)
@printf("\n  Worst-5%% terminal-price range:  \$%.2f .. \$%.2f\n",
        minimum(S_terminal[worst_idx]), maximum(S_terminal[worst_idx]))
@printf("  Mean short-PUT P&L on worst-5%% paths:  \$%+.2f\n",
        mean(pnl_put[worst_idx]))
@printf("  Mean short-CALL P&L on top-5%%   paths: \$%+.2f\n",
        mean(pnl_call[best_idx]))
