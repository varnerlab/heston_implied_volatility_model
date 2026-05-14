"""
Per-day per-ticker per-maturity SABR baseline (Reviewer 1, 2, 3).

For each ladder date × ticker × maturity bucket, fits SABR(α, β=0.5, ρ, ν)
to the observed implied-vol smile and reports the pooled RMSE. This is the
"practitioner reference" the reviewers asked us to compare ψ_NN against:
SABR has no cross-date pooling and refits every (date, ticker, expiry)
slice independently.

Parameter constraints (re-parameterised internally for unconstrained Optim):
  α > 0     via α = exp(θ_α)
  |ρ| < 1   via ρ = tanh(θ_ρ)
  ν > 0     via ν = exp(θ_ν)
  β fixed   = 0.5 (CIR-consistent; cuts free param to 3)

Output:
  code/figures/sabr_per_day_fits.jld2
  code/figures/sabr_per_day_summary.csv
"""

using CSV
using DataFrames
using Dates
using JLD2
using Optim
using Printf
using Statistics

using HestonIV  # forces precompile of dependent packages
include(joinpath(@__DIR__, "..", "src", "SABR.jl"))   # bare include — top-level functions

const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladder")
const FIG_DIR    = joinpath(@__DIR__, "..", "figures")
const BETA       = 0.5
const ALL_DAYS = [
    "options-04-14-2026", "options-04-15-2026",
    "options-04-16-2026", "options-04-17-2026",
    "options-04-21-2026", "options-04-22-2026",
    "options-04-23-2026", "options-04-24-2026",
    "options-04-27-2026", "options-04-28-2026",
    "options-04-29-2026", "options-05-01-2026",
    "options-05-06-2026", "options-05-08-2026",
    "options-05-11-2026",
]
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

function _dir_to_date(d)
    m = match(r"options-(\d{2})-(\d{2})-(\d{4})", d)
    Date(parse(Int, m.captures[3]), parse(Int, m.captures[1]), parse(Int, m.captures[2]))
end

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
        (df.actual_dte .> 0), :]
    return valid
end

# Maturity buckets: width 7 days, integer-bucketed
maturity_bucket(dte::Real) = max(1, round(Int, dte / 7))

function fit_sabr_slice(F::Float64, T::Float64, Ks::Vector{Float64},
                       ivs::Vector{Float64})
    # Initial: α = ATM IV * F^(1-β), ρ = -0.3, ν = 0.5
    F1mb = F^(1 - BETA)
    idx_atm = argmin(abs.(Ks ./ F .- 1.0))
    α0 = max(ivs[idx_atm] * F1mb, 0.05)
    θ0 = Float64[log(α0), atanh(-0.3), log(0.5)]

    function loss(θ)
        α = exp(θ[1])
        ρ = tanh(θ[2])
        ν = exp(θ[3])
        s = 0.0
        for i in 1:length(Ks)
            σ_hat = try
                hagan_sabr_iv(F, Ks[i], T, α, BETA, ρ, ν)
            catch
                return 1e6
            end
            s += (σ_hat - ivs[i])^2
        end
        return s / length(Ks)
    end

    res = optimize(loss, θ0, NelderMead(),
                   Optim.Options(g_tol=1e-7, iterations=400, show_trace=false))
    θ_opt = res.minimizer
    α_opt = exp(θ_opt[1])
    ρ_opt = tanh(θ_opt[2])
    ν_opt = exp(θ_opt[3])
    rmse_fit = sqrt(Optim.minimum(res))
    return (α=α_opt, ρ=ρ_opt, ν=ν_opt, rmse=rmse_fit, converged=Optim.converged(res))
end

# ============================================================================
# Drive across (date × ticker × maturity_bucket)
# ============================================================================

fits = Dict{NTuple{3,Any}, NamedTuple}()
all_residuals = Float64[]              # for pooled RMSE
per_sector_residuals = Dict{String, Vector{Float64}}()

for day in ALL_DAYS
    day_date = _dir_to_date(day)
    full = joinpath(LADDER_DIR, day)
    isdir(full) || continue
    csvs = filter(f -> endswith(f, ".csv") && occursin("_dte_ladder_", f), readdir(full))
    @printf("[%s]  %d ticker files\n", day, length(csvs))
    for fn in csvs
        df = _load_ladder(joinpath(full, fn))
        nrow(df) < 6 && continue
        ticker = string(df.ticker[1])
        sector = get(SECTORS, ticker, "Other")
        F = Float64(df.S[1])   # use spot as forward proxy (q=0)
        df[!, :bucket] = maturity_bucket.(df.actual_dte)
        for bucket in sort(unique(df.bucket))
            sl = df[df.bucket .== bucket, :]
            nrow(sl) >= 4 || continue  # need enough strikes to fit 3 params
            T_yrs = mean(Float64.(sl.actual_dte)) / 365.0
            Ks = Float64.(sl.strike)
            ivs = Float64.(sl.implied_vol)
            fit = try
                fit_sabr_slice(F, T_yrs, Ks, ivs)
            catch e
                @printf("    skip %s b=%d (%s)\n", ticker, bucket, sprint(showerror, e))
                continue
            end
            # Per-row residuals
            for i in 1:length(Ks)
                σ_hat = try
                    hagan_sabr_iv(F, Ks[i], T_yrs, fit.α, BETA, fit.ρ, fit.ν)
                catch
                    NaN
                end
                isnan(σ_hat) && continue
                resid = σ_hat - ivs[i]
                push!(all_residuals, resid)
                v = get!(per_sector_residuals, sector, Float64[])
                push!(v, resid)
            end
            fits[(day_date, ticker, bucket)] = (
                α=fit.α, ρ=fit.ρ, ν=fit.ν,
                n_obs=nrow(sl), T_yrs=T_yrs, F=F,
                rmse=fit.rmse, converged=fit.converged,
            )
        end
    end
end

# ============================================================================
# Aggregate
# ============================================================================

pooled_rmse = sqrt(mean(all_residuals.^2))
@printf("\n========  SABR per-day baseline summary  ========\n")
@printf("  n_fits          = %d (date × ticker × maturity slices)\n", length(fits))
@printf("  n_residuals     = %d total IV-vs-fit residuals across all rows\n", length(all_residuals))
@printf("  Pooled RMSE     = %5.2f%% IV\n", 100 * pooled_rmse)
println("\n  Per-sector pooled RMSE:")
println("  Sector         N_resid    RMSE(%)")
println("  " * "-"^40)
per_sector_rmse = Dict{String,Float64}()
for s in sort(collect(keys(per_sector_residuals)))
    r = per_sector_residuals[s]
    σ = sqrt(mean(r.^2))
    per_sector_rmse[s] = σ
    @printf("  %-12s   %7d    %5.2f\n", s, length(r), σ*100)
end

mkpath(FIG_DIR)
# JLD2 fit table
out_jld = joinpath(FIG_DIR, "sabr_per_day_fits.jld2")
JLD2.jldsave(out_jld; fits=fits, beta=BETA, n_residuals=length(all_residuals))
@printf("\n[jld2] wrote -> %s\n", out_jld)

# CSV summary
rows = NamedTuple[(name="overall", n_residuals=length(all_residuals), rmse=pooled_rmse)]
for (s, σ) in per_sector_rmse
    push!(rows, (name=s, n_residuals=length(per_sector_residuals[s]), rmse=σ))
end
out_csv = joinpath(FIG_DIR, "sabr_per_day_summary.csv")
CSV.write(out_csv, DataFrame(rows))
@printf("[csv]  wrote -> %s\n", out_csv)

println("\nDone.")
