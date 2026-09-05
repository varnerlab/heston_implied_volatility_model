"""
    ScenarioTemplate

Parametrized short-premium scenario engine. Drives the per-ticker scenarios
in §5 of the paper plus the expanded cross-ticker (B5) and earnings-window
(B6) sweeps from the JCF revision plan.

The pipeline is the one from `gs_short_premium_simulation.jl`, lifted to a
module so the only per-ticker code is a ~30-line driver that builds a
`ScenarioSpec` and calls `run_short_scenario` / `render_scenario_figures`.

Public API:
- `ScenarioSpec(ticker, anchor_date, expiry_date, K_put, K_call, …)`
- `VarianceSpec(; kappa=15.0, sigma_v=0.5, rho=-0.6, r_free=0.0425, q_div=0.0,
               n_steps_lr=201)`
- `run_short_scenario(spec, hspec; nn_cache_path, port_path, ladder_dir,
               sim_cache_path, resim=false, sector=nothing, use_per_ticker=true,
               compute_greeks=true)`
- `render_scenario_figures(result, spec; plot_dir)` — `plot_dir` may be one
  directory or several; every figure is written to all of them.
- `paper_figure_dirs(subdir)` — that subdirectory in every paper variant

The Leisen-Reimer pricer is imported from `HestonIV.lr_american_price`.
"""
module ScenarioTemplate

using CSV
using DataFrames
using Dates
using Distributions
using Flux
using JLD2
using JumpHMM
using Plots
using Plots.PlotMeasures
using Printf
using Random
using Statistics

# We pull LR from the package so the pricer lives in one place.
using HestonIV: lr_american_price, crr_american_price

export ScenarioSpec, VarianceSpec, HestonSpec, run_short_scenario, render_scenario_figures,
       paper_figure_dirs

# ============================================================================
# Cache validation
# ============================================================================

"""
    _cache_matches(cache, ref; rtol=1e-9) → Bool

True iff `cache` contains every key of `ref` with a matching value. Numbers
compare with `isapprox`, everything else with `==`. A missing key counts as a
mismatch, so caches written before a config field existed invalidate loudly
instead of silently serving stale simulations.
"""
function _cache_matches(cache::AbstractDict, ref::AbstractDict; rtol::Float64=1e-9)::Bool
    for (k, v) in ref
        haskey(cache, k) || return false
        c = cache[k]
        if v isa Number && c isa Number
            isapprox(c, v; rtol=rtol) || return false
        else
            c == v || return false
        end
    end
    return true
end

"""
    _psi_checksum(nn) → Float64

Cheap content checksum of a Flux network's weights (sum of absolute values of
the flattened parameter vector). Stored with each simulation cache so a
regenerated ψ_NN invalidates caches built from the old weights.
"""
_psi_checksum(nn)::Float64 = sum(abs, Flux.destructure(nn)[1])

"""Two-sided Wilson score interval for a binomial proportion."""
function _wilson_interval(successes::Integer, trials::Integer;
                          z::Float64=1.959963984540054)::Tuple{Float64,Float64}
    trials > 0 || error("Wilson interval requires at least one trial")
    p = successes / trials
    z2_over_n = z^2 / trials
    center = (p + z2_over_n / 2) / (1 + z2_over_n)
    half_width = z / (1 + z2_over_n) *
                 sqrt(p * (1 - p) / trials + z^2 / (4 * trials^2))
    return center - half_width, center + half_width
end

# ============================================================================
# Figure destinations
# ============================================================================

const _REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const _PAPER_VARIANTS = ("paper-arxiv", "paper-jcf")

"""True when any `.tex` under `sections_dir` includes a figure from `subdir/`."""
function _cites_subdir(sections_dir::AbstractString, subdir::AbstractString)
    isdir(sections_dir) || return false
    pat = Regex("sections/figures/" * subdir * "/")
    for (root, _, files) in walkdir(sections_dir), f in files
        endswith(f, ".tex") || continue
        occursin(pat, read(joinpath(root, f), String)) && return true
    end
    return false
end

"""
    paper_figure_dirs(subdir; repo_root, variants)

`sections/figures/<subdir>` in each paper variant that should receive this
figure set.

The scenario scripts write their figures straight into the paper tree instead
of going through `code/figures/`, so `promote_figures` never sees them on the
way out. A driver that names a single variant therefore leaves the other one
holding figures from whatever run last targeted it, which is how the GS and
LLY panels in `paper-jcf` came to disagree with `paper-arxiv`. Deriving the
destinations here makes them a property of the repository rather than of
whichever driver was edited last.

A variant qualifies when its own `.tex` files cite the subdirectory, matching
the rule `promote_figures` applies to flat figures, or when it already holds a
copy that would otherwise go stale. Both papers cite `gs/` and `lly/`, so
those render to both; the expanded cross-ticker and INTC panels are cited by
neither and stay in the one variant that already carries them instead of being
duplicated. A brand-new set that nothing cites yet falls back to the first
present variant.

Throws when no variant is found: a silent no-op here is exactly what hid the
equivalent `promote_figures` bug for two months.
"""
function paper_figure_dirs(subdir::AbstractString;
                           repo_root::AbstractString=_REPO_ROOT,
                           variants=_PAPER_VARIANTS)
    present = [v for v in variants if isdir(joinpath(repo_root, v, "sections"))]
    isempty(present) && error(
        "paper_figure_dirs: no paper variant found under $repo_root " *
        "(looked for $(join(variants, ", "))). If the paper directories moved, " *
        "update _PAPER_VARIANTS here and in code/scripts/promote_figures.jl.")
    wanted = [v for v in present
              if _cites_subdir(joinpath(repo_root, v, "sections"), subdir) ||
                 isdir(joinpath(repo_root, v, "sections", "figures", subdir))]
    isempty(wanted) && (wanted = [first(present)])
    return [joinpath(repo_root, v, "sections", "figures", subdir) for v in wanted]
end

"""Copy every `<pref>_*.{pdf,png}` from `src` into each of `dsts`."""
function _mirror_figures(src::AbstractString, dsts::AbstractVector{<:AbstractString},
                         pref::AbstractString)
    isempty(dsts) && return String[]
    names = filter(readdir(src)) do f
        startswith(f, pref * "_") && (endswith(f, ".pdf") || endswith(f, ".png"))
    end
    for d in dsts
        mkpath(d)
        for f in names
            cp(joinpath(src, f), joinpath(d, f); force=true)
        end
    end
    return names
end

# ============================================================================
# Specs
# ============================================================================

Base.@kwdef struct ScenarioSpec
    ticker::String
    anchor_date::Date
    expiry_date::Date
    K_put::Float64
    K_call::Float64
    market_premium_put::Float64
    market_premium_call::Float64
    market_iv_put::Float64
    market_iv_call::Float64
    market_delta_put::Float64
    market_delta_call::Float64
    expiry_label::String
    ticker_prior_ccgr_pct::Float64
    market_holidays::Vector{Date} = Date[]
    n_paths::Int = 1000
    seed::Int = 20260429
    fig_subdir::String = ""  # plot output subdirectory; defaults to lowercase(ticker)
end

"""Return the observation date followed by every simulated exchange session."""
function _trading_dates(spec::ScenarioSpec)::Vector{Date}
    spec.expiry_date > spec.anchor_date ||
        error("expiry_date must be later than anchor_date")
    holidays = Set(spec.market_holidays)
    dates = Date[spec.anchor_date]
    for d in (spec.anchor_date + Day(1)):Day(1):spec.expiry_date
        dayofweek(d) <= 5 && !(d in holidays) && push!(dates, d)
    end
    dates[end] == spec.expiry_date ||
        error("expiry_date $(spec.expiry_date) is not a simulated exchange session")
    return dates
end

_calendar_dtes(spec::ScenarioSpec, dates::Vector{Date}) =
    [Dates.value(spec.expiry_date - d) for d in dates]

Base.@kwdef struct VarianceSpec
    kappa::Float64 = 15.0
    sigma_v::Float64 = 0.5
    rho::Float64 = -0.6
    r_free::Float64 = 0.0425
    q_div::Float64 = 0.0
    n_steps_lr::Int = 201
end

# Backward-compatible name for older experiment drivers.
const HestonSpec = VarianceSpec

fig_subdir_of(spec::ScenarioSpec) = isempty(spec.fig_subdir) ? lowercase(spec.ticker) : spec.fig_subdir

# ============================================================================
# Ladder corpus loader + standardization
# ============================================================================

function _load_ladder(filepath)
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

function _load_all_ladders(dir)
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
        df = _load_ladder(f)
        nrow(df) > 0 && push!(frames, df)
    end
    return vcat(frames...)
end

struct _Standardizer
    mu_dte::Float64
    sigma_dte::Float64
    mu_m::Float64
    sigma_m::Float64
end

function _build_standardizer(all_data::DataFrame)
    log_dte = log.(max.(Float64.(all_data.actual_dte), 1.0))
    log_m   = log.(Float64.(all_data.moneyness))
    return _Standardizer(mean(log_dte), std(log_dte), mean(log_m), std(log_m))
end

# ============================================================================
# NN cache → ψ surface + θ̄ for one ticker
# ============================================================================

# Replicates calibrate_ladders_per_ticker_nn.jl architecture selector.
_build_psi_nn(n_obs::Integer, threshold::Integer) = n_obs >= threshold ?
    Chain(Dense(2 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 1)) :
    Chain(Dense(2 => 8,  tanh), Dense(8  => 8,  tanh), Dense(8  => 1))

# JLD2 deserializes Dict{String, NamedTuple} as a SerializedDict whose .kvvec
# field is a Vector{Pair{String, ReconstructedMutable}}. Provide a uniform
# accessor so we can look up by string key regardless of whether the cache was
# written as a real Dict or reconstructed.
function _payload_lookup(container, key::String)
    if container isa AbstractDict && haskey(container, key)
        return container[key]
    end
    if hasfield(typeof(container), :kvvec)
        for entry in getfield(container, :kvvec)
            entry.first == key && return entry.second
        end
    elseif applicable(iterate, container)
        for entry in container
            entry isa Pair && entry.first == key && return entry.second
        end
    end
    return nothing
end

function _payload_field(p, name::Symbol)
    # ReconstructedMutable exposes fields via getfield; NamedTuples via getproperty
    if hasfield(typeof(p), name)
        return getfield(p, name)
    end
    return getproperty(p, name)
end

"""
    _restore_nn(nn_cache, ticker; use_per_ticker, sector)

Restore the trained ψ network and `log θ` for one ticker. Tries the
per-ticker payload first; falls back to the sector payload if the ticker
didn't qualify for a per-ticker fit (i.e., <5000 ladder observations).
Returns `(psi_nn, log_theta::Float64, source::Symbol)` where `source` is
`:per_ticker` or `:sector`.
"""
function _restore_nn(nn_cache::Dict, ticker::String;
                     use_per_ticker::Bool, sector::Union{Nothing,String})
    if use_per_ticker && haskey(nn_cache, "per_ticker_payload")
        pt = _payload_lookup(nn_cache["per_ticker_payload"], ticker)
        if pt !== nothing
            n_obs = Int(_payload_field(pt, :n_obs))
            psi = _build_psi_nn(n_obs, 5000)
            Flux.loadmodel!(psi, _payload_field(pt, :state))
            log_theta_vec = _payload_field(pt, :log_theta)
            return psi, Float64(log_theta_vec[1]), :per_ticker
        end
    end
    sector === nothing && error("ticker $(ticker) not in per_ticker_payload; pass `sector=...` to use the sector fallback.")
    haskey(nn_cache, "sector_payload") || error("NN cache has no sector_payload.")
    sp = _payload_lookup(nn_cache["sector_payload"], sector)
    sp === nothing && error("sector $(sector) missing from NN cache.")

    sp_n_obs = Int(_payload_field(sp, :n_obs))
    psi = _build_psi_nn(sp_n_obs, 2000)
    Flux.loadmodel!(psi, _payload_field(sp, :state))
    sp_tickers = _payload_field(sp, :tickers)
    sp_log_theta = _payload_field(sp, :log_theta)
    idx = findfirst(==(ticker), sp_tickers)
    idx === nothing && error("ticker $(ticker) missing from sector $(sector) payload (tickers = $(sp_tickers)).")
    return psi, Float64(sp_log_theta[idx]), :sector
end

"""ψ((K/S)_std, DTE_std) → ψ scalar"""
function _psi(psi_nn, std::_Standardizer, K::Float64, S::Float64, dte_days::Real)
    log_dte_s = Float32((log(max(Float64(dte_days), 1.0)) - std.mu_dte) / std.sigma_dte)
    log_m_s   = Float32((log(K / S) - std.mu_m) / std.sigma_m)
    x = reshape(Float32[log_dte_s, log_m_s], 2, 1)
    log_psi = first(psi_nn(x))
    return Float64(exp(log_psi))
end

# ============================================================================
# Simulation core
# ============================================================================

struct _RestoredModel
    psi_nn::Any
    log_theta::Float64
    theta_bar::Float64
    source::Symbol
    standardizer::_Standardizer
end

function _resolve_anchor_spot(all_data::DataFrame, ticker::String, anchor_date::Date)
    slice = all_data[all_data.ticker .== ticker, :]
    anchor_rows = slice[slice.und_session_date .== anchor_date, :]
    isempty(anchor_rows) && error("No $(ticker) rows for anchor date $(anchor_date) in the ladder corpus.")
    return Float64(anchor_rows.S[1])
end

function _reconstruct_stock_path(observations, S0; rf, dt)
    prices = JumpHMM.prices_from_growth_rates(observations,S0;rf,dt)
    length(prices) == length(observations)+1 || error("Unexpected JumpHMM price-path length")
    return prices
end

function _simulate_paths(spec::ScenarioSpec, hspec::VarianceSpec, model::_RestoredModel,
                         S_0::Float64, port_path::String,
                         trading_dates::Vector{Date}, calendar_dtes::Vector{Int})
    println("\nLoading pretrained portfolio model...")
    portfolio = JLD2.load(port_path)
    marginal = portfolio["marginals"][spec.ticker]

    n_sim_steps = length(trading_dates) - 1
    println("Simulating $(spec.n_paths) $(spec.ticker) price paths over $(n_sim_steps) trading steps...")
    sim = JumpHMM.simulate(marginal, n_sim_steps; n_paths=spec.n_paths, seed=spec.seed)
    n_actual = length(sim.paths)

    target_drift = spec.ticker_prior_ccgr_pct / 100.0
    empirical_drift = mean(vcat([sim.paths[p].observations for p in 1:n_actual]...))
    drift_shift = target_drift - empirical_drift
    @printf("  Drift anchor: empirical %+.2f%%/yr → target %+.2f%%/yr   (shift %+.2f%%/yr)\n",
            empirical_drift*100, target_drift*100, drift_shift*100)
    for p in 1:n_actual
        sim.paths[p].observations .+= drift_shift
    end

    S_paths = Array{Float64}(undef, n_sim_steps + 1, n_actual)
    S_paths[1, :] .= S_0
    for p in 1:n_actual
        prices = _reconstruct_stock_path(sim.paths[p].observations,
                                                  S_0; rf=marginal.rf,
                                                  dt=marginal.dt)
        # JumpHMM returns the initial price followed by every transition.
        length(prices) == n_sim_steps + 1 || error("Unexpected JumpHMM price-path length")
        S_paths[:, p] .= prices
    end

    println("Evolving per-contract CIR variance with leverage coupling (κ=$(hspec.kappa), σ_v=$(hspec.sigma_v), ρ=$(hspec.rho))...")
    # Variance floor: 0.5% IV equivalent. Path A's per-contract target θ̄·ψ for
    # deep-OTM strikes on low-IV tickers (e.g. SPY put) is small enough that an
    # unfloored CIR with σ_v=0.5 can stall the LR pricer with σ → 0.
    VAR_FLOOR = 2.5e-5
    v_put_paths  = Array{Float64}(undef, n_sim_steps + 1, n_actual)
    v_call_paths = Array{Float64}(undef, n_sim_steps + 1, n_actual)
    # v_0 = θ(i,0;K) = θ̄ · ψ at each contract's entry coordinates (Path A).
    entry_dte = calendar_dtes[1]
    psi_put_0  = _psi(model.psi_nn, model.standardizer, spec.K_put,  S_0, entry_dte)
    psi_call_0 = _psi(model.psi_nn, model.standardizer, spec.K_call, S_0, entry_dte)
    v_put_paths[1,  :] .= model.theta_bar * psi_put_0
    v_call_paths[1, :] .= model.theta_bar * psi_call_0

    rng = MersenneTwister(spec.seed + 1)
    # The JumpHMM advances in exchange sessions, so the variance factor uses a
    # trading-year clock. Calendar DTE is retained separately for option pricing.
    Δt = 1.0 / 252.0
    sqrtΔt = sqrt(Δt)
    rho2 = sqrt(1.0 - hspec.rho^2)

    # Standardize the simulated one-session log returns across the ensemble.
    # This makes rho a transparent return/variance-shock coupling coefficient;
    # it is not an estimate of a Heston Brownian correlation.
    ΔlogS = log.(S_paths[2:end, :] ./ S_paths[1:end-1, :])
    μ_ΔlogS = mean(ΔlogS)
    σ_ΔlogS = std(vec(ΔlogS))
    σ_ΔlogS > eps(Float64) || error("Simulated return dispersion is zero")
    Z_S_paths = (ΔlogS .- μ_ΔlogS) ./ σ_ΔlogS

    for p in 1:n_actual
        for t in 2:(n_sim_steps + 1)
            Z_S = Z_S_paths[t-1, p]
            Z_v_indep = randn(rng)
            Z_v = hspec.rho * Z_S + rho2 * Z_v_indep

            # Per-contract drift target θ(i,t;K) = θ̄ · ψ(K, S_{t-1}, DTE_remaining).
            dte_remaining = max(calendar_dtes[t-1], 1)
            psi_put_t  = _psi(model.psi_nn, model.standardizer, spec.K_put,
                              S_paths[t-1, p], dte_remaining)
            psi_call_t = _psi(model.psi_nn, model.standardizer, spec.K_call,
                              S_paths[t-1, p], dte_remaining)
            θ_put_t  = model.theta_bar * psi_put_t
            θ_call_t = model.theta_bar * psi_call_t

            v_put_prev  = v_put_paths[t-1, p]
            v_call_prev = v_call_paths[t-1, p]
            dv_put  = hspec.kappa * (θ_put_t  - v_put_prev)  * Δt +
                      hspec.sigma_v * sqrt(max(v_put_prev,  1e-10)) * sqrtΔt * Z_v
            dv_call = hspec.kappa * (θ_call_t - v_call_prev) * Δt +
                      hspec.sigma_v * sqrt(max(v_call_prev, 1e-10)) * sqrtΔt * Z_v
            v_put_paths[t,  p] = max(v_put_prev  + dv_put,  VAR_FLOOR)
            v_call_paths[t, p] = max(v_call_prev + dv_call, VAR_FLOOR)
        end
    end
    return S_paths, v_put_paths, v_call_paths
end

"""
    _safe_lr_price(S, K, σ, r, T, N, otype; q)

LR American pricer with an American CRR safeguard when low volatility or
extreme moneyness saturates the Peizer-Pratt probabilities. The fallback
retains early exercise and chooses a valid CRR depth. It currently requires
zero dividends, as used in the paper scenarios.
"""
function _safe_lr_price(S::Float64, K::Float64, σ::Float64,
                        r::Float64, T::Float64, N::Int,
                        otype::Symbol; q::Float64=0.0)
    T <= 0 && return otype === :call ? max(S-K,0.0) : max(K-S,0.0)
    value = σ * sqrt(T) < 0.01 ? NaN : lr_american_price(S,K,σ,r,T,N,otype;q=q)
    if !isfinite(value)
        q == 0 || error("The CRR safeguard currently requires q=0")
        depth = max(N, ceil(Int,(r-q)^2*T/σ^2)+1)
        return crr_american_price(S,K,σ,r,T,depth,otype;q=q)
    end
    return value
end

function _price_paths(spec::ScenarioSpec, hspec::VarianceSpec, model::_RestoredModel,
                      S_paths::Array{Float64,2},
                      v_put_paths::Array{Float64,2}, v_call_paths::Array{Float64,2},
                      calendar_dtes::Vector{Int})
    println("Pricing put + call at every (t, path) via Leisen-Reimer American (n_steps=$(hspec.n_steps_lr))...")
    n_actual = size(S_paths, 2)
    n_dates = length(calendar_dtes)
    V_put  = Array{Float64}(undef, n_dates, n_actual)
    V_call = Array{Float64}(undef, n_dates, n_actual)
    for p in 1:n_actual
        for t in 1:n_dates
            S_t = S_paths[t, p]
            dte_remaining = calendar_dtes[t]
            if dte_remaining == 0
                V_put[t, p]  = max(spec.K_put  - S_t, 0.0)
                V_call[t, p] = max(S_t - spec.K_call, 0.0)
            else
                T_rem_yrs = dte_remaining / 365.0
                σ_put  = sqrt(max(v_put_paths[t,  p], 1e-10))
                σ_call = sqrt(max(v_call_paths[t, p], 1e-10))
                V_put[t, p]  = _safe_lr_price(S_t, spec.K_put,  σ_put,
                                              hspec.r_free, T_rem_yrs, hspec.n_steps_lr,
                                              :put;  q=hspec.q_div)
                V_call[t, p] = _safe_lr_price(S_t, spec.K_call, σ_call,
                                              hspec.r_free, T_rem_yrs, hspec.n_steps_lr,
                                              :call; q=hspec.q_div)
            end
        end
        (p % 100 == 0 || p == n_actual) && @printf("    %4d / %d paths priced\n", p, n_actual)
    end
    return V_put, V_call
end

function _compute_greeks(spec::ScenarioSpec, hspec::VarianceSpec, model::_RestoredModel,
                         S_paths::Array{Float64,2},
                         v_put_paths::Array{Float64,2}, v_call_paths::Array{Float64,2},
                         V_put::Array{Float64,2}, V_call::Array{Float64,2},
                         calendar_dtes::Vector{Int})
    n_actual = size(S_paths, 2)
    H_S_FRAC = 0.015
    D_SIGMA  = 0.005
    σ_put_paths    = fill(NaN, size(S_paths))
    σ_call_paths   = fill(NaN, size(S_paths))
    Δ_put_paths    = fill(NaN, size(S_paths))
    Δ_call_paths   = fill(NaN, size(S_paths))
    Γ_put_paths    = fill(NaN, size(S_paths))
    Γ_call_paths   = fill(NaN, size(S_paths))
    Vega_put_paths  = fill(NaN, size(S_paths))
    Vega_call_paths = fill(NaN, size(S_paths))

    println("\nComputing Leisen-Reimer American Greeks via finite differences (h=1.5% S, dσ=±0.5 IV pts)...")
    for p in 1:n_actual
        for t in 1:(length(calendar_dtes) - 1)
            dte = calendar_dtes[t]
            T_rem = dte / 365.0
            S_t = S_paths[t, p]
            σ_p = sqrt(max(v_put_paths[t,  p], 1e-10))
            σ_c = sqrt(max(v_call_paths[t, p], 1e-10))
            σ_put_paths[t,  p] = σ_p * 100
            σ_call_paths[t, p] = σ_c * 100

            h = H_S_FRAC * S_t
            σ_p_lo = max(σ_p - D_SIGMA, 1e-4)
            σ_c_lo = max(σ_c - D_SIGMA, 1e-4)

            V_pS0 = V_put[t, p]
            V_pSp = _safe_lr_price(S_t + h, spec.K_put, σ_p,           hspec.r_free, T_rem, hspec.n_steps_lr, :put; q=hspec.q_div)
            V_pSm = _safe_lr_price(S_t - h, spec.K_put, σ_p,           hspec.r_free, T_rem, hspec.n_steps_lr, :put; q=hspec.q_div)
            V_pσp = _safe_lr_price(S_t,     spec.K_put, σ_p + D_SIGMA, hspec.r_free, T_rem, hspec.n_steps_lr, :put; q=hspec.q_div)
            V_pσm = _safe_lr_price(S_t,     spec.K_put, σ_p_lo,        hspec.r_free, T_rem, hspec.n_steps_lr, :put; q=hspec.q_div)
            Δ_put_paths[t, p]    = (V_pSp - V_pSm) / (2h)
            Γ_put_paths[t, p]    = (V_pSp - 2*V_pS0 + V_pSm) / (h^2)
            Vega_put_paths[t, p] = (V_pσp - V_pσm) / (2 * D_SIGMA * 100)

            V_cS0 = V_call[t, p]
            V_cSp = _safe_lr_price(S_t + h, spec.K_call, σ_c,           hspec.r_free, T_rem, hspec.n_steps_lr, :call; q=hspec.q_div)
            V_cSm = _safe_lr_price(S_t - h, spec.K_call, σ_c,           hspec.r_free, T_rem, hspec.n_steps_lr, :call; q=hspec.q_div)
            V_cσp = _safe_lr_price(S_t,     spec.K_call, σ_c + D_SIGMA, hspec.r_free, T_rem, hspec.n_steps_lr, :call; q=hspec.q_div)
            V_cσm = _safe_lr_price(S_t,     spec.K_call, σ_c_lo,        hspec.r_free, T_rem, hspec.n_steps_lr, :call; q=hspec.q_div)
            Δ_call_paths[t, p]    = (V_cSp - V_cSm) / (2h)
            Γ_call_paths[t, p]    = (V_cSp - 2*V_cS0 + V_cSm) / (h^2)
            Vega_call_paths[t, p] = (V_cσp - V_cσm) / (2 * D_SIGMA * 100)
        end
        (p % 100 == 0 || p == n_actual) && @printf("    Greeks: %4d / %d paths done\n", p, n_actual)
    end
    return σ_put_paths, σ_call_paths, Δ_put_paths, Δ_call_paths,
           Γ_put_paths, Γ_call_paths, Vega_put_paths, Vega_call_paths
end

# ============================================================================
# Entry point: run + cache
# ============================================================================

"""
    run_short_scenario(spec, hspec; nn_cache_path, port_path, ladder_dir,
                       sim_cache_path, resim=false, use_per_ticker=true,
                       sector=nothing, compute_greeks=true) → NamedTuple

Top-level driver. Returns a NamedTuple with everything `render_scenario_figures`
needs plus the summary statistics that the paper consumes.
"""
function run_short_scenario(spec::ScenarioSpec, hspec::VarianceSpec;
                      nn_cache_path::String, port_path::String,
                      ladder_dir::String, sim_cache_path::String,
                      resim::Bool=false, use_per_ticker::Bool=true,
                      sector::Union{Nothing,String}=nothing,
                      compute_greeks::Bool=true)
    isfile(nn_cache_path) || error("NN cache missing at $(nn_cache_path).")
    isfile(port_path)     || error("Portfolio model missing at $(port_path).")

    trading_dates = _trading_dates(spec)
    calendar_dtes = _calendar_dtes(spec, trading_dates)
    calendar_days_elapsed = [Dates.value(d - spec.anchor_date) for d in trading_dates]
    calendar_horizon = calendar_dtes[1]
    n_sim_steps = length(trading_dates) - 1
    @printf("Scenario clock: %d calendar days, %d trading steps (%s to %s)\n",
            calendar_horizon, n_sim_steps, spec.anchor_date, spec.expiry_date)

    println("Loading ladder corpus for standardisation constants...")
    all_data = _load_all_ladders(ladder_dir)
    standardizer = _build_standardizer(all_data)
    S_0 = _resolve_anchor_spot(all_data, spec.ticker, spec.anchor_date)
    @printf("  S_0 (%s, anchored to %s) = \$%.2f\n", spec.ticker, spec.anchor_date, S_0)

    println("Loading NN ψ surface for $(spec.ticker) ...")
    nn_cache = JLD2.load(nn_cache_path)
    psi_nn, log_theta, src = _restore_nn(nn_cache, spec.ticker;
                                         use_per_ticker=use_per_ticker, sector=sector)
    theta_bar = exp(log_theta)
    @printf("  ψ source: %s    θ̄ = %.4f  →  IV %.1f%%\n", src, theta_bar, sqrt(theta_bar)*100)

    model = _RestoredModel(psi_nn, log_theta, theta_bar, src, standardizer)

    # Quote model fair value at t=0 against market mid for entry edge.
    nn_iv_put  = sqrt(max(theta_bar * _psi(psi_nn, standardizer, spec.K_put,  S_0, calendar_horizon), 1e-10))
    nn_iv_call = sqrt(max(theta_bar * _psi(psi_nn, standardizer, spec.K_call, S_0, calendar_horizon), 1e-10))
    @printf("  Real-contract setup (%s %s, %d calendar DTE):\n",
            spec.ticker, spec.expiry_label, calendar_horizon)
    @printf("    Put  K=\$%.2f   market Δ=%+0.3f   mid=\$%.2f   market IV=%.1f%%   (model NN-IV=%.1f%%)\n",
            spec.K_put,  spec.market_delta_put,  spec.market_premium_put,  spec.market_iv_put*100,
            nn_iv_put*100)
    @printf("    Call K=\$%.2f   market Δ=%+0.3f   mid=\$%.2f   market IV=%.1f%%   (model NN-IV=%.1f%%)\n",
            spec.K_call, spec.market_delta_call, spec.market_premium_call, spec.market_iv_call*100,
            nn_iv_call*100)

    # Try cache first (matches the cache contract from gs_short_premium_simulation.jl).
    # The reference dict must cover EVERY input that changes the simulation:
    # market anchor, strikes, drift prior, variance dynamics, clock/horizon,
    # path count, seed, rates, pricer depth, and the ψ_NN weights themselves.
    # Caches written before a field existed fail the check and resimulate.
    psi_checksum = _psi_checksum(psi_nn)
    cache_ref = Dict{String,Any}(
        "scenario_version"      => 2,
        "S_0"                   => S_0,
        "K_put"                 => spec.K_put,
        "K_call"                => spec.K_call,
        "ticker_prior_ccgr_pct" => spec.ticker_prior_ccgr_pct,
        "return_variance_rho"   => hspec.rho,
        "cir_kappa"             => hspec.kappa,
        "cir_sigma_v"           => hspec.sigma_v,
        "anchor_date"           => string(spec.anchor_date),
        "expiry_date"           => string(spec.expiry_date),
        "trading_dates"         => string.(trading_dates),
        "calendar_dtes"         => calendar_dtes,
        "variance_dt"           => 1.0 / 252.0,
        "n_paths"               => spec.n_paths,
        "seed"                  => spec.seed,
        "r_free"                => hspec.r_free,
        "q_div"                 => hspec.q_div,
        "n_steps_lr"            => hspec.n_steps_lr,
        "theta_bar"             => theta_bar,
        "psi_src"               => src,
        "psi_checksum"          => psi_checksum)
    sim_cache_valid = !resim && isfile(sim_cache_path)
    if sim_cache_valid
        cache = JLD2.load(sim_cache_path)
        ok = _cache_matches(cache, cache_ref) &&
             haskey(cache, "v_put_paths") && haskey(cache, "v_call_paths")
        if ok
            println("Cache hit: loading prior simulation from $(basename(sim_cache_path))")
            S_paths      = cache["S_paths"]
            v_put_paths  = cache["v_put_paths"]
            v_call_paths = cache["v_call_paths"]
            V_put        = cache["V_put"]
            V_call       = cache["V_call"]
        else
            println("Cache parameters drifted from current spec; resimulating.")
            sim_cache_valid = false
        end
    end
    if !sim_cache_valid
        S_paths, v_put_paths, v_call_paths = _simulate_paths(
            spec, hspec, model, S_0, port_path, trading_dates, calendar_dtes)
        V_put, V_call = _price_paths(
            spec, hspec, model, S_paths, v_put_paths, v_call_paths, calendar_dtes)
        mkpath(dirname(sim_cache_path))
        JLD2.jldsave(sim_cache_path;
            scenario_version=2,
            S_paths=S_paths, v_put_paths=v_put_paths, v_call_paths=v_call_paths,
            V_put=V_put, V_call=V_call,
            S_0=S_0, K_put=spec.K_put, K_call=spec.K_call,
            theta_bar=theta_bar,
            cir_kappa=hspec.kappa, cir_sigma_v=hspec.sigma_v,
            return_variance_rho=hspec.rho,
            ticker_prior_ccgr_pct=spec.ticker_prior_ccgr_pct, seed=spec.seed,
            anchor_date=string(spec.anchor_date), expiry_date=string(spec.expiry_date),
            trading_dates=string.(trading_dates), calendar_dtes=calendar_dtes,
            variance_dt=1.0/252.0, n_paths=spec.n_paths,
            r_free=hspec.r_free, q_div=hspec.q_div, n_steps_lr=hspec.n_steps_lr,
            psi_src=src, psi_checksum=psi_checksum)
        println("[cache] saved -> $(sim_cache_path)")
    end

    n_actual = size(S_paths, 2)

    pnl_put  = spec.market_premium_put  .- V_put[end,  :]
    pnl_call = spec.market_premium_call .- V_call[end, :]
    model_fv_put_t0  = V_put[1, 1]
    model_fv_call_t0 = V_call[1, 1]

    greeks = if compute_greeks
        _compute_greeks(spec, hspec, model, S_paths, v_put_paths, v_call_paths,
                        V_put, V_call, calendar_dtes)
    else
        (fill(NaN, size(S_paths)), fill(NaN, size(S_paths)),
         fill(NaN, size(S_paths)), fill(NaN, size(S_paths)),
         fill(NaN, size(S_paths)), fill(NaN, size(S_paths)),
         fill(NaN, size(S_paths)), fill(NaN, size(S_paths)))
    end
    σ_put_paths, σ_call_paths,
    Δ_put_paths, Δ_call_paths,
    Γ_put_paths, Γ_call_paths,
    Vega_put_paths, Vega_call_paths = greeks

    S_terminal = S_paths[end, :]
    n_tail     = max(1, round(Int, 0.05 * n_actual))
    worst_idx  = sortperm(S_terminal)[1:n_tail]
    best_idx   = sortperm(S_terminal; rev=true)[1:n_tail]

    keep_put = pnl_put .>= spec.market_premium_put - 1e-6
    keep_call = pnl_call .>= spec.market_premium_call - 1e-6
    pct_keep_put = 100 * mean(keep_put)
    pct_keep_call = 100 * mean(keep_call)
    keep_ci_put = 100 .* _wilson_interval(count(keep_put), n_actual)
    keep_ci_call = 100 .* _wilson_interval(count(keep_call), n_actual)
    q05_put = quantile(pnl_put, 0.05)
    q05_call = quantile(pnl_call, 0.05)
    es05_put = mean(pnl_put[pnl_put .<= q05_put])
    es05_call = mean(pnl_call[pnl_call .<= q05_call])

    summary_row = (
        ticker=spec.ticker, S_0=S_0,
        K_put=spec.K_put, K_call=spec.K_call,
        market_mid_put=spec.market_premium_put,   market_mid_call=spec.market_premium_call,
        model_fv_put=model_fv_put_t0,             model_fv_call=model_fv_call_t0,
        entry_edge_put=model_fv_put_t0 - spec.market_premium_put,
        entry_edge_call=model_fv_call_t0 - spec.market_premium_call,
        median_pnl_put=median(pnl_put),           median_pnl_call=median(pnl_call),
        mean_pnl_put=mean(pnl_put),               mean_pnl_call=mean(pnl_call),
        mean_se_put=std(pnl_put) / sqrt(n_actual), mean_se_call=std(pnl_call) / sqrt(n_actual),
        q05_pnl_put=q05_put,                      q05_pnl_call=q05_call,
        es05_pnl_put=es05_put,                    es05_pnl_call=es05_call,
        worst_loss_put=minimum(pnl_put),          worst_loss_call=minimum(pnl_call),
        pct_premium_kept_put=pct_keep_put,        pct_premium_kept_call=pct_keep_call,
        pct_premium_kept_ci_put=keep_ci_put,      pct_premium_kept_ci_call=keep_ci_call,
    )

    return (
        spec=spec, hspec=hspec,
        trading_dates=trading_dates, calendar_dtes=calendar_dtes,
        calendar_days_elapsed=calendar_days_elapsed,
        calendar_horizon=calendar_horizon, n_sim_steps=n_sim_steps,
        S_0=S_0, theta_bar=theta_bar, nn_source=src,
        S_paths=S_paths, v_put_paths=v_put_paths, v_call_paths=v_call_paths,
        V_put=V_put, V_call=V_call,
        σ_put_paths=σ_put_paths, σ_call_paths=σ_call_paths,
        Δ_put_paths=Δ_put_paths, Δ_call_paths=Δ_call_paths,
        Γ_put_paths=Γ_put_paths, Γ_call_paths=Γ_call_paths,
        Vega_put_paths=Vega_put_paths, Vega_call_paths=Vega_call_paths,
        pnl_put=pnl_put, pnl_call=pnl_call,
        worst_idx=worst_idx, best_idx=best_idx, n_tail=n_tail,
        n_paths_actual=n_actual,
        summary_row=summary_row,
    )
end

# ============================================================================
# Figures
# ============================================================================

const _COL_PATH    = RGBA(0.30, 0.30, 0.30, 0.06)
const _COL_MEDIAN  = RGB(0.00, 0.00, 0.00)
const _COL_WORST   = RGBA(0.78, 0.20, 0.20, 0.45)
const _COL_BEST    = RGBA(0.20, 0.55, 0.30, 0.45)
const _COL_PREMIUM = RGBA(0.30, 0.40, 0.85, 0.55)

function _path_panel(t_axis, paths_mat, title_str, ylabel_str; tails,
                     legend_pos=:topleft, overlay_zero=false)
    p = plot(legend=legend_pos,
             legendfontsize=9, titlefontsize=12,
             guidefontsize=11, tickfontsize=10,
             xlabel="Calendar days from sale", ylabel=ylabel_str, title=title_str,
             framestyle=:box, grid=true, gridalpha=0.25,
             foreground_color_grid=:gray, background_color=:white,
             background_color_legend=RGBA(1.0, 1.0, 1.0, 0.92),
             foreground_color_legend=RGB(0.55, 0.55, 0.55),
             left_margin=9mm, right_margin=4mm, top_margin=4mm, bottom_margin=6mm)
    for j in 1:size(paths_mat, 2)
        plot!(p, t_axis, paths_mat[:, j], color=_COL_PATH, lw=0.4, label="")
    end
    for tail in tails
        for (k, j) in enumerate(tail.idx)
            plot!(p, t_axis, paths_mat[:, j], color=tail.color, lw=1.0,
                  label=(k == 1 ? tail.label : ""))
        end
    end
    med = [median(paths_mat[t, :]) for t in 1:size(paths_mat, 1)]
    q25 = [quantile(paths_mat[t, :], 0.25) for t in 1:size(paths_mat, 1)]
    q75 = [quantile(paths_mat[t, :], 0.75) for t in 1:size(paths_mat, 1)]
    plot!(p, t_axis, med, ribbon=(med .- q25, q75 .- med),
          color=_COL_MEDIAN, fillalpha=0.12, lw=2.2,
          label="Median  (band: 25-75%)")
    overlay_zero && hline!(p, [0.0], color=:gray, ls=:dash, alpha=0.5, label="")
    return p
end

function _pnl_panel(pnl_vec, premium_t0, title_str)
    q05    = quantile(pnl_vec, 0.05)
    q50    = median(pnl_vec)
    qmean  = mean(pnl_vec)
    qmin   = minimum(pnl_vec)
    pct_keep_full = 100 * mean(pnl_vec .>= premium_t0 - 1e-6)

    lo  = floor(qmin / 10) * 10 - 5
    hi  = ceil(premium_t0 / 5) * 5 + 5
    bin_w = max(2.5, (hi - lo) / 60)
    edges = lo:bin_w:hi

    p = histogram(pnl_vec, bins=edges, normalize=:probability,
                  color=RGBA(0.45, 0.45, 0.45, 0.85), lw=0.2,
                  linecolor=RGBA(0.20, 0.20, 0.20, 0.85),
                  yscale=:log10, ylims=(5e-4, 1.0),
                  yticks=([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0],
                          ["0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1.0"]),
                  legend=:topleft, legendfontsize=9,
                  titlefontsize=12, guidefontsize=11, tickfontsize=10,
                  framestyle=:box, grid=true, gridalpha=0.25,
                  foreground_color_grid=:gray, background_color=:white,
                  background_color_legend=RGBA(1.0, 1.0, 1.0, 0.92),
                  foreground_color_legend=RGB(0.55, 0.55, 0.55),
                  title=title_str,
                  xlabel="Terminal P&L per share (\$)",
                  ylabel="Fraction of paths  (log scale)",
                  left_margin=11mm, right_margin=4mm,
                  top_margin=4mm, bottom_margin=7mm, label="")
    vline!(p, [0.0],   color=:black, ls=:solid, lw=0.8, alpha=0.7, label="")
    vline!(p, [q05],   color=RGB(0.78, 0.20, 0.20), ls=:dash, lw=1.6,
           label=@sprintf("5%% quantile  =  \$%+8.2f", q05))
    vline!(p, [q50],   color=RGB(0.10, 0.10, 0.10), ls=:solid, lw=1.8,
           label=@sprintf("Median        =  \$%+8.2f", q50))
    vline!(p, [qmean], color=RGB(0.10, 0.30, 0.70), ls=:dashdot, lw=1.6,
           label=@sprintf("Mean          =  \$%+8.2f", qmean))
    vline!(p, [premium_t0], color=_COL_PREMIUM, ls=:dot, lw=1.6,
           label=@sprintf("Premium kept  =  \$%+8.2f  (%.1f%% of paths)",
                          premium_t0, pct_keep_full))
    return p
end

"""
    render_scenario_figures(result, spec; plot_dir, prefix=nothing)

Produces the five figure sets in the paper-figures directory. `plot_dir` takes
either one directory or several (see `paper_figure_dirs`); the figures are
rendered once and copied to the rest, and the full destination list is
returned.
- `{prefix}_short_put_paths.{pdf,png}`
- `{prefix}_short_call_paths.{pdf,png}`
- `{prefix}_short_paths.{pdf,png}`
- `{prefix}_short_pnl_distributions.{pdf,png}`
- `{prefix}_iv_trajectories.{pdf,png}`
- `{prefix}_short_greeks.{pdf,png}` (if Greeks present)
"""
function render_scenario_figures(result, spec::ScenarioSpec;
                                 plot_dir::Union{AbstractString,AbstractVector{<:AbstractString}},
                                 prefix::Union{Nothing,String}=nothing)
    pref = prefix === nothing ? lowercase(spec.ticker) : prefix
    dirs = plot_dir isa AbstractString ? [String(plot_dir)] : String.(plot_dir)
    isempty(dirs) && error("render_scenario_figures: plot_dir resolved to no directories")
    # Render once into the first destination, then copy; re-rendering per
    # variant would burn the plotting cost again for byte-identical output.
    primary = first(dirs)
    mkpath(primary)

    S_paths = result.S_paths
    V_put   = result.V_put
    V_call  = result.V_call
    n       = result.n_paths_actual
    S_0     = result.S_0

    # Display-only overlays: drop the extreme 1% tail.
    S_terminal = S_paths[end, :]
    n_tail   = result.n_tail
    n_ext    = max(1, round(Int, 0.01 * n))
    best_disp  = sortperm(S_terminal; rev=true)[(n_ext+1):n_tail]
    worst_disp = sortperm(S_terminal)[(n_ext+1):n_tail]

    both_tails = [
        (idx = worst_disp, color = _COL_WORST,
         label = "Worst 1-5% by terminal price"),
        (idx = best_disp,  color = _COL_BEST,
         label = "Top 1-5% by terminal price"),
    ]
    t_axis = result.calendar_days_elapsed
    horizon_label = "$(result.calendar_horizon)-calendar-day"
    title_share = @sprintf("%s share price paths   (S₀ = \$%.2f)", spec.ticker, S_0)

    # --- A: put panel ---
    println("Rendering figure A: stock + put premium ...")
    pA1 = _path_panel(t_axis, S_paths, title_share, "Share price (\$)";
                      tails=both_tails, legend_pos=:topleft)
    hline!(pA1, [spec.K_put], color=RGB(0.85, 0.65, 0.13), ls=:dash, lw=3.5, alpha=1.0,
           label=@sprintf("30Δ put strike  K = \$%.2f", spec.K_put))
    pA2 = _path_panel(t_axis, V_put, "Short $(horizon_label) 30Δ put: option price",
                      "Put price (\$)"; tails=both_tails, legend_pos=:topleft)
    hline!(pA2, [spec.market_premium_put], color=_COL_PREMIUM, ls=:dash, lw=1.5,
           label=@sprintf("Premium received  \$%.2f", spec.market_premium_put))
    p_A = plot(pA1, pA2, layout=(1, 2), size=(1500, 600), dpi=220,
               left_margin=9mm, right_margin=4mm, bottom_margin=7mm, top_margin=4mm)
    savefig(p_A, joinpath(primary, "$(pref)_short_put_paths.pdf"))
    savefig(p_A, joinpath(primary, "$(pref)_short_put_paths.png"))

    # --- B: call panel ---
    println("Rendering figure B: stock + call premium ...")
    pB1 = _path_panel(t_axis, S_paths, title_share, "Share price (\$)";
                      tails=both_tails, legend_pos=:topleft)
    hline!(pB1, [spec.K_call], color=RGB(0.85, 0.65, 0.13), ls=:dash, lw=3.5, alpha=1.0,
           label=@sprintf("30Δ call strike  K = \$%.2f", spec.K_call))
    pB2 = _path_panel(t_axis, V_call, "Short $(horizon_label) 30Δ call: option price",
                      "Call price (\$)"; tails=both_tails, legend_pos=:topleft)
    hline!(pB2, [spec.market_premium_call], color=_COL_PREMIUM, ls=:dash, lw=1.5,
           label=@sprintf("Premium received  \$%.2f", spec.market_premium_call))
    p_B = plot(pB1, pB2, layout=(1, 2), size=(1500, 600), dpi=220,
               left_margin=9mm, right_margin=4mm, bottom_margin=7mm, top_margin=4mm)
    savefig(p_B, joinpath(primary, "$(pref)_short_call_paths.pdf"))
    savefig(p_B, joinpath(primary, "$(pref)_short_call_paths.png"))

    # --- A+B: combined 2x2 ---
    println("Rendering figure A+B: combined 2x2 ...")
    pAB1 = _path_panel(t_axis, S_paths, title_share, "Share price (\$)";
                       tails=both_tails, legend_pos=:topleft)
    hline!(pAB1, [spec.K_put], color=RGB(0.85, 0.65, 0.13), ls=:dash, lw=3.5, alpha=1.0,
           label=@sprintf("30Δ put strike  K = \$%.2f", spec.K_put))
    pAB2 = _path_panel(t_axis, V_put, "Short $(horizon_label) 30Δ put: option price",
                       "Put price (\$)"; tails=both_tails, legend_pos=:topleft)
    hline!(pAB2, [spec.market_premium_put], color=_COL_PREMIUM, ls=:dash, lw=1.5,
           label=@sprintf("Premium received  \$%.2f", spec.market_premium_put))
    pAB3 = _path_panel(t_axis, S_paths, title_share, "Share price (\$)";
                       tails=both_tails, legend_pos=:topleft)
    hline!(pAB3, [spec.K_call], color=RGB(0.85, 0.65, 0.13), ls=:dash, lw=3.5, alpha=1.0,
           label=@sprintf("30Δ call strike  K = \$%.2f", spec.K_call))
    pAB4 = _path_panel(t_axis, V_call, "Short $(horizon_label) 30Δ call: option price",
                       "Call price (\$)"; tails=both_tails, legend_pos=:topleft)
    hline!(pAB4, [spec.market_premium_call], color=_COL_PREMIUM, ls=:dash, lw=1.5,
           label=@sprintf("Premium received  \$%.2f", spec.market_premium_call))
    p_AB = plot(pAB1, pAB2, pAB3, pAB4, layout=(2, 2), size=(1500, 1100), dpi=220,
                left_margin=9mm, right_margin=4mm, bottom_margin=7mm, top_margin=4mm)
    savefig(p_AB, joinpath(primary, "$(pref)_short_paths.pdf"))
    savefig(p_AB, joinpath(primary, "$(pref)_short_paths.png"))

    # --- C: P&L distributions ---
    println("Rendering figure C: terminal short P&L distributions ...")
    pC1 = _pnl_panel(result.pnl_put,  spec.market_premium_put,
                     "Short $(horizon_label) 30Δ put: terminal P&L distribution")
    pC2 = _pnl_panel(result.pnl_call, spec.market_premium_call,
                     "Short $(horizon_label) 30Δ call: terminal P&L distribution")
    p_C = plot(pC1, pC2, layout=(1, 2), size=(1500, 600), dpi=220,
               left_margin=11mm, right_margin=4mm, bottom_margin=7mm, top_margin=4mm)
    savefig(p_C, joinpath(primary, "$(pref)_short_pnl_distributions.pdf"))
    savefig(p_C, joinpath(primary, "$(pref)_short_pnl_distributions.png"))

    # --- D: IV trajectories ---
    if !all(isnan, result.σ_put_paths)
        println("Rendering figure D: IV trajectories ...")
        t_iv = result.calendar_days_elapsed[1:end-1]
        σ_put_plot  = result.σ_put_paths[1:end-1, :]
        σ_call_plot = result.σ_call_paths[1:end-1, :]
        pD1 = _path_panel(t_iv, σ_put_plot,
                          "Implied volatility at the put strike  (K = \$$(round(spec.K_put, digits=2)))",
                          "Annualized IV (%)"; tails=both_tails, legend_pos=:topleft)
        pD2 = _path_panel(t_iv, σ_call_plot,
                          "Implied volatility at the call strike  (K = \$$(round(spec.K_call, digits=2)))",
                          "Annualized IV (%)"; tails=both_tails, legend_pos=:topleft)
        p_D = plot(pD1, pD2, layout=(1, 2), size=(1500, 600), dpi=220,
                   left_margin=9mm, right_margin=4mm, bottom_margin=7mm, top_margin=4mm)
        savefig(p_D, joinpath(primary, "$(pref)_iv_trajectories.pdf"))
        savefig(p_D, joinpath(primary, "$(pref)_iv_trajectories.png"))

        # --- E: Greeks ---
        println("Rendering figure E: Greeks trajectories ...")
        t_g = t_iv
        pE1 = _path_panel(t_g, result.Δ_put_paths[1:end-1, :],
                          "Long-put Δ at K_put", "Δ (long put)";
                          tails=both_tails, legend_pos=:bottomleft)
        pE2 = _path_panel(t_g, result.Δ_call_paths[1:end-1, :],
                          "Long-call Δ at K_call", "Δ (long call)";
                          tails=both_tails, legend_pos=:topleft)
        pE3 = _path_panel(t_g, result.Γ_put_paths[1:end-1, :],
                          "Γ at K_put", "Γ (per share, per \$ in S)";
                          tails=both_tails, legend_pos=:topleft)
        pE4 = _path_panel(t_g, result.Γ_call_paths[1:end-1, :],
                          "Γ at K_call", "Γ (per share, per \$ in S)";
                          tails=both_tails, legend_pos=:topleft)
        pE5 = _path_panel(t_g, result.Vega_put_paths[1:end-1, :],
                          "Vega at K_put", "Vega (\$ per 1% IV move)";
                          tails=both_tails, legend_pos=:topright)
        pE6 = _path_panel(t_g, result.Vega_call_paths[1:end-1, :],
                          "Vega at K_call", "Vega (\$ per 1% IV move)";
                          tails=both_tails, legend_pos=:topright)
        p_E = plot(pE1, pE2, pE3, pE4, pE5, pE6, layout=(3, 2),
                   size=(1500, 1500), dpi=220,
                   left_margin=9mm, right_margin=4mm, bottom_margin=5mm, top_margin=4mm)
        savefig(p_E, joinpath(primary, "$(pref)_short_greeks.pdf"))
        savefig(p_E, joinpath(primary, "$(pref)_short_greeks.png"))
    end

    mirrored = _mirror_figures(primary, dirs[2:end], pref)
    if length(dirs) > 1
        println("Mirrored $(length(mirrored)) figure file(s) to $(length(dirs) - 1) " *
                "other paper variant(s).")
    end
    return dirs
end

# ============================================================================
# Summary printer
# ============================================================================

"""
    print_summary(result; io=stdout)

Print the per-ticker summary block used at the end of each scenario script.
"""
function print_summary(result; io::IO=stdout)
    spec = result.spec
    hspec = result.hspec
    n = result.n_paths_actual
    n_tail = result.n_tail
    pnl_put  = result.pnl_put
    pnl_call = result.pnl_call
    S_terminal = result.S_paths[end, :]

    println(io, "\n" * "="^78)
    @printf(io, "  SHORT-PREMIUM SCENARIO SUMMARY  (%s, %d calendar days, %d trading steps, %d paths)\n",
            spec.ticker, result.calendar_horizon, result.n_sim_steps, n)
    println(io, "="^78)
    @printf(io, "  Spot:                S_0 = \$%.2f\n", result.S_0)
    @printf(io, "  30Δ put strike:      K = \$%.2f   (K/S = %.3f)\n", spec.K_put,  spec.K_put / result.S_0)
    @printf(io, "  30Δ call strike:     K = \$%.2f   (K/S = %.3f)\n", spec.K_call, spec.K_call / result.S_0)
    @printf(io, "  CIR-like (κ, σ_v):   (%.1f, %.1f); return-shock coupling ρ = %.2f\n",
            hspec.kappa, hspec.sigma_v, hspec.rho)
    println(io)
    @printf(io, "  %-12s  %-22s  %-22s\n", "stat", "Short PUT P&L (\$)", "Short CALL P&L (\$)")
    println(io, "  " * "-"^60)
    for (lbl, fp, fc) in [
        ("Premium",  spec.market_premium_put, spec.market_premium_call),
        ("Mean",     mean(pnl_put),           mean(pnl_call)),
        ("Median",   median(pnl_put),         median(pnl_call)),
        ("Std",      std(pnl_put),            std(pnl_call)),
        ("5%-tile",  quantile(pnl_put, 0.05), quantile(pnl_call, 0.05)),
        ("ES (5%)",  result.summary_row.es05_pnl_put, result.summary_row.es05_pnl_call),
        ("Min",      minimum(pnl_put),        minimum(pnl_call)),
        ("Max",      maximum(pnl_put),        maximum(pnl_call))]
        @printf(io, "  %-12s  %+22.2f  %+22.2f\n", lbl, fp, fc)
    end
    pct_keep_put  = 100 * mean(pnl_put  .>= spec.market_premium_put  - 1e-6)
    pct_keep_call = 100 * mean(pnl_call .>= spec.market_premium_call - 1e-6)
    @printf(io, "\n  Premium kept in full (PUT):   %5.1f%% of paths\n",  pct_keep_put)
    @printf(io, "    Wilson 95%% CI:              [%5.1f%%, %5.1f%%]\n",
            result.summary_row.pct_premium_kept_ci_put[1],
            result.summary_row.pct_premium_kept_ci_put[2])
    @printf(io,   "  Premium kept in full (CALL):  %5.1f%% of paths\n",  pct_keep_call)
    @printf(io, "    Wilson 95%% CI:              [%5.1f%%, %5.1f%%]\n",
            result.summary_row.pct_premium_kept_ci_call[1],
            result.summary_row.pct_premium_kept_ci_call[2])
    @printf(io, "\n  Worst-5%% terminal-price range:  \$%.2f .. \$%.2f\n",
            minimum(S_terminal[result.worst_idx]), maximum(S_terminal[result.worst_idx]))
    @printf(io, "  Mean short-PUT P&L on worst-5%% paths:  \$%+.2f\n",
            mean(pnl_put[result.worst_idx]))
    @printf(io, "  Mean short-CALL P&L on top-5%%   paths: \$%+.2f\n",
            mean(pnl_call[result.best_idx]))
end

export print_summary

end # module ScenarioTemplate
