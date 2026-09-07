# Score the four frozen stock methods on identical 2025 and 2026 endpoints.
using CSV,DataFrames,Dates,JLD2,JumpHMM,Random,Statistics,TOML,SHA,Test
include(joinpath(@__DIR__,"..","src","ForecastValidation.jl"))
using .ForecastValidation
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const OUT=joinpath(ROOT,"code/results/small_stock_comparison")
const N=10000
const SEEDS=[202609091,202609092,202609093]
inputs=CSV.read(joinpath(OUT,"forecast_inputs.csv"),DataFrame;types=Dict(:period=>String))
portfolio=JLD2.load(joinpath(ROOT,"code/data/pretrained-portfolio-surrogate.jld2"))
pilots=TOML.parsefile(joinpath(ROOT,"code/results/chronological_validation/pilot_constants.toml"))
scores=NamedTuple[];analytic_checks=NamedTuple[]
source_files=["code/scripts/run_small_stock_comparison.jl","code/src/ForecastValidation.jl",
    "code/results/small_stock_comparison/forecast_inputs.csv",
    "code/results/small_stock_comparison/frozen_settings.json",
    "code/results/chronological_validation/pilot_constants.toml",
    "code/data/pretrained-portfolio-surrogate.jld2"]
hashes=Dict(f=>(open(sha256,joinpath(ROOT,f))|>bytes2hex) for f in source_files)
@testset "Small stock forecast distributions" begin
for (ticker_index,ticker) in enumerate(["GS","LLY"])
    model=portfolio["marginals"][ticker]
    shift=pilots[ticker]["growth_shift"]
    for group in groupby(inputs[inputs.ticker.==ticker,:],[:period,:origin])
        period=group.period[1];origin=Date(group.origin[1]);S0=group.origin_spot[1]
        variance=group.daily_variance[1];drift=group.daily_drift[1]
        @assert all(group.origin_spot.==S0) && all(group.daily_variance.==variance) && all(group.daily_drift.==drift)
        for (replicate,base_seed) in enumerate(SEEDS)
            seed=base_seed+1000ticker_index+10000Dates.value(origin-Date("2025-01-01"))
            sim=JumpHMM.simulate(model,5;n_paths=N,seed)
            legacy=(hcat([p.observations for p in sim.paths]...).+shift.+model.rf).*model.dt
            innovations=randn(MersenneTwister(seed+1),5,N)
            adaptive=sqrt(variance).*innovations
            variants=[("JumpHMM",legacy),("Unchanged",zeros(5,N)),
                ("Adaptive volatility",adaptive),("Directional",adaptive.+drift)]
            for (method,returns) in variants
                paths=S0.*exp.(cumsum(returns;dims=1))
                for r in eachrow(group)
                    values=paths[r.horizon,:];d=distribution_scores(values,r.observed)
                    push!(scores,merge((period,ticker,origin,endpoint=r.endpoint,horizon=r.horizon,
                        method,replicate,seed,paths=N,origin_spot=S0,observed=r.observed,
                        median_absolute_error=abs(d.predicted_median-r.observed),
                        mean_error_pct=100d.error/S0,median_absolute_error_pct=100abs(d.predicted_median-r.observed)/S0,
                        crps_pct=100d.crps/S0,width_pct=100d.width90/S0,
                        analytic_mean=method=="Adaptive volatility" ? S0*exp(.5r.horizon*variance) :
                            method=="Directional" ? S0*exp(r.horizon*(drift+.5variance)) : NaN),d))
                    if replicate==1 && origin==minimum(Date.(inputs.origin[(inputs.period.==period).&(inputs.ticker.==ticker)])) && method in ["Adaptive volatility","Directional"]
                        mu=method=="Directional" ? drift*r.horizon : 0.0
                        theoretical_mean=S0*exp(mu+.5r.horizon*variance)
                        theoretical_sd=theoretical_mean*sqrt(expm1(r.horizon*variance))
                        difference=d.predicted_mean-theoretical_mean
                        @test abs(difference)<6theoretical_sd/sqrt(N)
                        @test all(isfinite,values) && minimum(values)>0
                        push!(analytic_checks,(period,ticker,origin,horizon=r.horizon,method,
                            theoretical_mean,simulated_mean=d.predicted_mean,mean_difference=difference))
                    end
                end
            end
        end
    end
    println("Completed ",ticker);flush(stdout)
end
end
CSV.write(joinpath(OUT,"scores.csv"),DataFrame(scores))
CSV.write(joinpath(OUT,"analytic_checks.csv"),DataFrame(analytic_checks))
open(joinpath(OUT,"run_manifest.toml"),"w") do io
    TOML.print(io,Dict("completed"=>true,"paths"=>N,"seeds"=>SEEDS,"source_sha256"=>hashes))
end
println("Completed frozen small comparison.")
