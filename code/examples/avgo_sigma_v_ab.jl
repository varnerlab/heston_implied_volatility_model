"""
A/B test: how much of the AVGO short-put price jumpiness is driven by
Heston vol-of-vol versus by the JumpHMM emission tails + ψ_NN smile
blow-up at low DTE?

Runs the full forward-simulation + Leisen–Reimer pricing pipeline
*twice*, identical except for `σ_v`:

    A. σ_v = 0.0   — variance frozen at θ_AVGO; only the JumpHMM marginal
                     drives share moves; the IV input to the option pricer
                     varies only via ψ_NN((K/S_t)_std, DTE_std).
    B. σ_v = 0.5   — production regime (matches avgo_short_premium_simulation.jl).

Renders a 2×2 grid: rows = σ_v regime, columns = (share paths, put price).
If row A is visibly as jumpy as row B, σ_v is innocent and the apparent
noise is JumpHMM jumps + smile-surface effects. If row A is materially
calmer, σ_v contributes.

Run:
    julia --project=. examples/avgo_sigma_v_ab.jl
"""

using CSV, DataFrames, Distributions, Flux, JLD2, JumpHMM
using Plots, Plots.PlotMeasures
using Printf, Random, Statistics

# ---------- LR American pricer (copy from avgo_short_premium_simulation.jl) ----
function _peizer_pratt(z::Float64, n::Int)
    arg = (z / (n + 1/3 + 0.1/(n+1)))^2 * (n + 1/6)
    return 0.5 + sign(z) * sqrt(0.25 - 0.25 * exp(-arg))
end
function lr_american_price(S, K, σ, r, T, N, otype; q=0.0)
    N % 2 == 0 && (N += 1)
    Δt = T / N
    disc = exp(-r * Δt)
    σsqrtT = σ * sqrt(T)
    d1 = (log(S/K) + (r - q + 0.5*σ^2)*T) / σsqrtT
    d2 = d1 - σsqrtT
    p_bar = _peizer_pratt(d2, N)
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

# ---------- Constants (mirror avgo_short_premium_simulation.jl) ---------------
const TICKER       = "AVGO"
const T_DAYS       = 31
const N_PATHS      = 1000
const N_STEPS_LR   = 201
const SEED         = 20260429
const R_FREE       = 0.0425
const Q_DIV        = 0.0
const AVGO_PRIOR_CCGR_PCT = 12.0
const K_PUT        = 375.0
const K_CALL       = 435.0
const PREMIUM_PUT  = 10.70

const HESTON_KAPPA = 2.0
const HESTON_RHO   = -0.6

const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladder")
const NN_CACHE   = joinpath(@__DIR__, "..", "figures",
                             "calibrate_ladders_per_ticker_nn_cache.jld2")
const PORT_PATH  = joinpath(@__DIR__, "..", "data",
                             "pretrained-portfolio-surrogate.jld2")
const OUT_DIR    = joinpath(@__DIR__, "..", "..", "paper", "sections",
                             "figures", "avgo")

# ---------- Load standardisation constants + S_0 ------------------------------
function load_ladder(filepath)
    df = CSV.read(filepath, DataFrame)
    df[!, :ticker] .= string(df.underlying[1])
    df[!, :S] .= df.und_close[1]
    df[!, :moneyness] = df.strike ./ df.und_close[1]
    df[
        .!ismissing.(df.implied_vol) .&
        .!isnan.(coalesce.(df.implied_vol, NaN)) .&
        (coalesce.(df.implied_vol, 0.0) .> 0.01) .&
        (coalesce.(df.implied_vol, 999.0) .< 2.0) .&
        (df.bid .> 0) .&
        (df.moneyness .>= 0.80) .&
        (df.moneyness .<= 1.20) .&
        (df.actual_dte .> 0), :]
end

println("Loading ladder corpus for standardisation...")
files = String[]
for (root, _, fs) in walkdir(LADDER_DIR)
    occursin("VIX-data", root) && continue
    for f in fs
        endswith(f, ".csv") && occursin("_dte_ladder_", f) && push!(files, joinpath(root, f))
    end
end
all_data = vcat([load_ladder(f) for f in files if nrow(load_ladder(f)) > 0]...)
const MU_DTE    = mean(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const SIGMA_DTE = std( log.(max.(Float64.(all_data.actual_dte), 1.0)))
const MU_M      = mean(log.(Float64.(all_data.moneyness)))
const SIGMA_M   = std( log.(Float64.(all_data.moneyness)))
const S_0 = Float64(all_data[all_data.ticker .== TICKER, :S][end])
@printf("  S_0 (%s) = \$%.2f\n", TICKER, S_0)

# ---------- Restore NN ψ surface ----------------------------------------------
build_psi_nn(n_obs::Integer, threshold::Integer) = n_obs >= threshold ?
    Chain(Dense(2 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 1)) :
    Chain(Dense(2 => 8,  tanh), Dense(8  => 8,  tanh), Dense(8  => 1))

println("Loading per-ticker NN ψ surface for $TICKER...")
nn_cache = JLD2.load(NN_CACHE)
const PT_PAYLOAD = nn_cache["per_ticker_payload"][TICKER]
const PSI_NN     = build_psi_nn(PT_PAYLOAD.n_obs, 5000)
Flux.loadmodel!(PSI_NN, PT_PAYLOAD.state)
const LOG_THETA  = Float64(PT_PAYLOAD.log_theta[1])
const THETA_TICKER = exp(LOG_THETA)

function psi_ticker(K, S, dte_days)
    log_dte_s = Float32((log(max(Float64(dte_days), 1.0)) - MU_DTE) / SIGMA_DTE)
    log_m_s   = Float32((log(K / S) - MU_M) / SIGMA_M)
    x = reshape(Float32[log_dte_s, log_m_s], 2, 1)
    Float64(exp(first(PSI_NN(x))))
end
heston_smile_iv(v, K, S, dte) = sqrt(max(v * psi_ticker(K, S, dte), 1e-10))

# ---------- The pipeline, parameterized by σ_v --------------------------------
function run_pipeline(sigma_v::Float64)
    println("\nRunning pipeline at σ_v = $sigma_v ...")
    portfolio = JLD2.load(PORT_PATH)
    ticker_model = portfolio["marginals"][TICKER]
    sim = JumpHMM.simulate(ticker_model, T_DAYS; n_paths=N_PATHS, seed=SEED)
    n = length(sim.paths)

    # Drift anchor (same as production)
    target_drift = AVGO_PRIOR_CCGR_PCT / 100.0
    emp_drift = mean(vcat([sim.paths[p].observations for p in 1:n]...))
    shift = target_drift - emp_drift
    for p in 1:n
        sim.paths[p].observations .+= shift
    end

    # Reconstruct prices
    S_paths = Array{Float64}(undef, T_DAYS + 1, n)
    S_paths[1, :] .= S_0
    for p in 1:n
        prices = JumpHMM.prices_from_growth_rates(sim.paths[p].observations, S_0;
                                                   rf=ticker_model.rf, dt=ticker_model.dt)
        S_paths[2:end, p] .= prices[1:T_DAYS]
    end

    # CIR variance evolution with parameterized σ_v (when σ_v=0 the variance
    # stays at θ except for the deterministic mean-reversion term which is
    # 0 since v_prev = θ → v stays at θ exactly)
    v_paths = Array{Float64}(undef, T_DAYS + 1, n)
    v_paths[1, :] .= THETA_TICKER
    rng = MersenneTwister(SEED + 1)
    Δt = 1.0 / 365.0
    sqrtΔt = sqrt(Δt)
    rho2 = sqrt(1.0 - HESTON_RHO^2)
    for p in 1:n
        for t in 2:(T_DAYS + 1)
            Δlogs = log(S_paths[t, p] / S_paths[t-1, p])
            σ_prev = sqrt(max(v_paths[t-1, p], 1e-8))
            Z_S = clamp((Δlogs - R_FREE * Δt) / (σ_prev * sqrtΔt + 1e-12), -6.0, 6.0)
            Z_v = HESTON_RHO * Z_S + rho2 * randn(rng)
            v_prev = v_paths[t-1, p]
            dv = HESTON_KAPPA * (THETA_TICKER - v_prev) * Δt +
                 sigma_v * sqrt(max(v_prev, 1e-10)) * sqrtΔt * Z_v
            v_paths[t, p] = max(v_prev + dv, 1e-10)
        end
    end

    # Price the put at every (t, path) via LR
    println("  Pricing put paths via LR...")
    V_put = Array{Float64}(undef, T_DAYS + 1, n)
    for p in 1:n
        for t in 0:T_DAYS
            S_t = S_paths[t+1, p]; v_t = v_paths[t+1, p]
            if t == T_DAYS
                V_put[t+1, p] = max(K_PUT - S_t, 0.0)
            else
                dte = T_DAYS - t
                σ = heston_smile_iv(v_t, K_PUT, S_t, dte)
                V_put[t+1, p] = lr_american_price(S_t, K_PUT, σ, R_FREE,
                                                   dte/365.0, N_STEPS_LR, :put; q=Q_DIV)
            end
        end
        p % 200 == 0 && @printf("    %d/%d\n", p, n)
    end
    return (S_paths=S_paths, V_put=V_put)
end

art_a = run_pipeline(0.0)   # vol-of-vol off
art_b = run_pipeline(0.5)   # production

# ---------- Tail bins (use the σ_v=0 paths' terminal ranking for both rows
#            so the same tail set is highlighted in both — apples to apples) --
S_terminal = art_b.S_paths[end, :]
n_tail = max(1, round(Int, 0.05 * N_PATHS))
n_extreme = max(1, round(Int, 0.01 * N_PATHS))
worst_disp = sortperm(S_terminal)[(n_extreme+1):n_tail]
best_disp  = sortperm(S_terminal; rev=true)[(n_extreme+1):n_tail]

# ---------- Plot helpers ------------------------------------------------------
const COL_PATH   = RGBA(0.30, 0.30, 0.30, 0.06)
const COL_MEDIAN = RGB(0.00, 0.00, 0.00)
const COL_WORST  = RGBA(0.78, 0.20, 0.20, 0.45)
const COL_BEST   = RGBA(0.20, 0.55, 0.30, 0.45)

function panel(t_axis, mat, title_str, ylabel_str)
    p = plot(legend = :topleft, legendfontsize = 8,
             titlefontsize = 11, guidefontsize = 10, tickfontsize = 9,
             xlabel = "Days from sale", ylabel = ylabel_str, title = title_str,
             framestyle = :box, grid = true, gridalpha = 0.25,
             foreground_color_grid = :gray, background_color = :white,
             background_color_legend = RGBA(1.0, 1.0, 1.0, 0.92),
             foreground_color_legend = RGB(0.55, 0.55, 0.55),
             left_margin = 9mm, right_margin = 4mm,
             top_margin = 4mm, bottom_margin = 6mm)
    for j in 1:size(mat, 2)
        plot!(p, t_axis, mat[:, j], color = COL_PATH, lw = 0.4, label = "")
    end
    for (k, j) in enumerate(worst_disp)
        plot!(p, t_axis, mat[:, j], color = COL_WORST, lw = 1.0,
              label = (k == 1 ? "Worst 1–5%" : ""))
    end
    for (k, j) in enumerate(best_disp)
        plot!(p, t_axis, mat[:, j], color = COL_BEST, lw = 1.0,
              label = (k == 1 ? "Top 1–5%" : ""))
    end
    med = [median(mat[t, :]) for t in 1:size(mat, 1)]
    q25 = [quantile(mat[t, :], 0.25) for t in 1:size(mat, 1)]
    q75 = [quantile(mat[t, :], 0.75) for t in 1:size(mat, 1)]
    plot!(p, t_axis, med, ribbon = (med .- q25, q75 .- med),
          color = COL_MEDIAN, fillalpha = 0.12, lw = 2.0,
          label = "Median (band: 25–75%)")
    return p
end

t_axis = collect(0:T_DAYS)

p_aS = panel(t_axis, art_a.S_paths, "AVGO share paths   (σ_v = 0.0, frozen variance)", "Share price (\$)")
hline!(p_aS, [K_PUT], color = RGB(0.85, 0.65, 0.13), ls = :dash, lw = 3.0,
       label = @sprintf("K_put = \$%.0f", K_PUT))
p_aV = panel(t_axis, art_a.V_put, "Short put — option price   (σ_v = 0.0)", "Put price (\$)")
hline!(p_aV, [PREMIUM_PUT], color = RGBA(0.30, 0.40, 0.85, 0.55), ls = :dash, lw = 1.2,
       label = @sprintf("Premium  \$%.2f", PREMIUM_PUT))

p_bS = panel(t_axis, art_b.S_paths, "AVGO share paths   (σ_v = 0.5, production)", "Share price (\$)")
hline!(p_bS, [K_PUT], color = RGB(0.85, 0.65, 0.13), ls = :dash, lw = 3.0,
       label = @sprintf("K_put = \$%.0f", K_PUT))
p_bV = panel(t_axis, art_b.V_put, "Short put — option price   (σ_v = 0.5)", "Put price (\$)")
hline!(p_bV, [PREMIUM_PUT], color = RGBA(0.30, 0.40, 0.85, 0.55), ls = :dash, lw = 1.2,
       label = @sprintf("Premium  \$%.2f", PREMIUM_PUT))

fig = plot(p_aS, p_aV, p_bS, p_bV, layout = (2, 2), size = (1500, 1100), dpi = 220,
           left_margin = 9mm, right_margin = 4mm,
           bottom_margin = 6mm, top_margin = 4mm)

mkpath(OUT_DIR)
out_pdf = joinpath(OUT_DIR, "avgo_sigma_v_ab.pdf")
out_png = joinpath(OUT_DIR, "avgo_sigma_v_ab.png")
savefig(fig, out_pdf); savefig(fig, out_png)
println("\n→ ", out_pdf)

# Quantitative comparison
function jumpiness_metrics(V)
    daily_changes = diff(V; dims = 1)
    abs_change = abs.(daily_changes)
    return (median_abs = median(abs_change),
            p95_abs    = quantile(vec(abs_change), 0.95),
            std_terminal = std(V[end, :]))
end
ma = jumpiness_metrics(art_a.V_put)
mb = jumpiness_metrics(art_b.V_put)
println("\n" * "="^60)
println("  PUT-PRICE JUMPINESS METRICS  (|day-over-day Δ| in \$)")
println("="^60)
@printf("                       σ_v=0.0      σ_v=0.5     ratio (B/A)\n")
@printf("  Median |ΔV/day|     \$%.3f       \$%.3f      %.2f×\n",
        ma.median_abs, mb.median_abs, mb.median_abs / ma.median_abs)
@printf("  95%%ile |ΔV/day|     \$%.3f       \$%.3f      %.2f×\n",
        ma.p95_abs, mb.p95_abs, mb.p95_abs / ma.p95_abs)
@printf("  std(V_terminal)     \$%.2f       \$%.2f      %.2f×\n",
        ma.std_terminal, mb.std_terminal, mb.std_terminal / ma.std_terminal)
println("\n  Interpretation: ratio ≈ 1.0 → σ_v innocent (jumpiness from")
println("  JumpHMM emissions + ψ_NN smile blow-up at low DTE).")
println("  Ratio ≫ 1.0 → σ_v contributes materially.")
