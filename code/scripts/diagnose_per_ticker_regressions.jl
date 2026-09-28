"""
Why do eight per-ticker networks fit worse in-sample than their sector network?

`examples/calibrate_ladders_per_ticker_nn.jl` gives tickers with fewer than
5,000 observations a 2->8->8->1 network (105 parameters) while every sector
network is 2->16->16->1 (337 parameters), and it keeps final-epoch weights
because its best-state checkpoint aliases the live arrays. All eight tickers
whose own network fitted worse than their sector network are in the small-
network group. This script separates the two candidate causes without touching
the saved fits: for each ticker in that group it retrains, with the same data,
standardization, seed, schedule, and stopping rule,

  1. the 2->8->8->1 network, recording final-epoch and lowest-loss RMSE, and
  2. a 2->16->16->1 network, recording the same two numbers.

The final-epoch small-network RMSE should reproduce the per-ticker column of
`logs/per_ticker_nn.log`; the sector RMSEs are read from that log.

Run:
    julia --project=. scripts/diagnose_per_ticker_regressions.jl
"""

using CSV
using DataFrames
using Flux
using Printf
using Random
using Statistics

const LADDER_DIR = joinpath(@__DIR__, "..", "data", "ladder")
const LOG_PATH = joinpath(@__DIR__, "..", "logs", "per_ticker_nn.log")
const OUT_CSV = joinpath(@__DIR__, "..", "results", "per_ticker_regression_diagnosis.csv")

# Same filters as the calibration script
function load_ladder(filepath)
    df = CSV.read(filepath, DataFrame)
    df[!, :ticker] .= string(df.underlying[1])
    df[!, :moneyness] = df.strike ./ df.und_close[1]
    df[.!ismissing.(df.implied_vol) .&
       .!isnan.(coalesce.(df.implied_vol, NaN)) .&
       (coalesce.(df.implied_vol, 0.0) .> 0.01) .&
       (coalesce.(df.implied_vol, 999.0) .< 2.0) .&
       (df.bid .> 0) .& (df.moneyness .>= 0.80) .& (df.moneyness .<= 1.20) .&
       (df.actual_dte .> 0), :]
end

files = String[]
for (root, _, fs) in walkdir(LADDER_DIR)
    occursin("VIX-data", root) && continue
    append!(files, joinpath(root, f) for f in fs if endswith(f, ".csv"))
end
all_data = vcat([d for d in load_ladder.(files) if nrow(d) > 0]...)
println("$(nrow(all_data)) observations across $(length(unique(all_data.ticker))) tickers")

const MU_DTE = mean(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const SIGMA_DTE = std(log.(max.(Float64.(all_data.actual_dte), 1.0)))
const MU_M = mean(log.(Float64.(all_data.moneyness)))
const SIGMA_M = std(log.(Float64.(all_data.moneyness)))

# Sector and logged per-ticker RMSE for every qualified ticker
logged = Dict{String,NamedTuple}()
for m in eachmatch(r"^\s+([A-Z]+)\s+\[(\w+)\s*\]\s+N=\s*(\d+)\s+arch=(\S+).*sector=([\d.]+)%\s+per-ticker=([\d.]+)%"m,
                   read(LOG_PATH, String))
    logged[m[1]] = (sector=m[2], n=parse(Int, m[3]), arch=m[4],
                    sector_rmse=parse(Float64, m[5]), logged_rmse=parse(Float64, m[6]))
end
small = sort([t for (t, r) in logged if r.arch == "2->8->8->1"])
println("$(length(small)) tickers used the small network")

ivs(m, X) = exp.(0.5f0 .* (m.log_theta[1] .+ vec(m.psi_nn(X))))
rmse_pct(m, X, y) = 100 * sqrt(mean((Float64.(ivs(m, X)) .- Float64.(y)) .^ 2))

# The calibration schedule, with the checkpoint copied so the lowest-loss
# weights can be restored and scored alongside the final ones.
function train(model, X, y)
    opt = Flux.setup(Flux.Adam(1f-3), model)
    best_loss, best_model, no_improve = Inf, deepcopy(model), 0
    for epoch in 1:2000
        l, g = Flux.withgradient(m -> Flux.mse(ivs(m, X), y), model)
        # l belongs to the weights before this update
        l < best_loss ? ((best_loss, best_model, no_improve) = (l, deepcopy(model), 0)) : (no_improve += 1)
        Flux.update!(opt, model, g[1])
        no_improve >= 200 && break
        epoch == 500  && Flux.adjust!(opt, 5f-4)
        epoch == 1000 && Flux.adjust!(opt, 2f-4)
        epoch == 1500 && Flux.adjust!(opt, 1f-4)
    end
    return model, best_model
end

rows = NamedTuple[]
for t in small
    td = all_data[all_data.ticker .== t, :]
    X = hcat(Float32.((log.(max.(Float64.(td.actual_dte), 1.0)) .- MU_DTE) ./ SIGMA_DTE),
             Float32.((log.(Float64.(td.moneyness)) .- MU_M) ./ SIGMA_M))'
    y = Float32.(td.implied_vol)
    theta0 = Float32[Float32(log(mean(Float64.(td.implied_vol))^2))]
    res = Dict{Int,Tuple{Float64,Float64}}()
    for h in (8, 16)
        Random.seed!(42)
        net = Chain(Dense(2 => h, tanh), Dense(h => h, tanh), Dense(h => 1))
        final, best = train((psi_nn = net, log_theta = copy(theta0)), X, y)
        res[h] = (rmse_pct(final, X, y), rmse_pct(best, X, y))
    end
    r = logged[t]
    push!(rows, (ticker=t, sector=r.sector, n=nrow(td), sector_rmse=r.sector_rmse,
                 logged_small=r.logged_rmse, small_final=res[8][1], small_best=res[8][2],
                 large_final=res[16][1], large_best=res[16][2]))
    @printf("%-5s %-10s N=%5d  sector %5.2f | small final %5.2f (log %5.2f) best %5.2f | large final %5.2f best %5.2f\n",
            t, r.sector, nrow(td), r.sector_rmse, res[8][1], r.logged_rmse, res[8][2], res[16][1], res[16][2])
end

df = sort(DataFrame(rows), :sector_rmse)
mkpath(dirname(OUT_CSV))
CSV.write(OUT_CSV, df)
worse(col) = count(df[!, col] .> df.sector_rmse)
println()
@printf("Worse than sector network: small final %d, small best %d, large final %d, large best %d (of %d)\n",
        worse(:small_final), worse(:small_best), worse(:large_final), worse(:large_best), nrow(df))
@printf("Max |small final - logged| = %.3f points\n", maximum(abs.(df.small_final .- df.logged_small)))
println("Wrote $OUT_CSV")
