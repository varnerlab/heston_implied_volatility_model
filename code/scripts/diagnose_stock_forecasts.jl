# Compare fixed stock forecast mechanisms using only information at each origin.
using CSV,DataFrames,Dates,JLD2,JumpHMM,Statistics,Random,LinearAlgebra,TOML,SHA,Test
include(joinpath(@__DIR__,"..","src","ForecastValidation.jl"))
include(joinpath(@__DIR__,"..","src","StockForecastDiagnosis.jl"))
using .ForecastValidation,.StockForecastDiagnosis
BLAS.set_num_threads(1)
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const ORIGINAL=joinpath(ROOT,"code/results/chronological_validation")
const OUT=joinpath(ROOT,"code/results/stock_forecast_diagnosis")
const N=10000
const SEEDS=[202609081,202609082,202609083]
const LAST=Date("2026-09-04")
stocks=CSV.read(joinpath(ORIGINAL,"stock_sessions.csv"),DataFrame)
stocks.session=Date.(stocks.session)
pilots=TOML.parsefile(joinpath(ORIGINAL,"pilot_constants.toml"))
portfolio=JLD2.load(joinpath(ROOT,"code/data/pretrained-portfolio-surrogate.jld2"))
old_scores=CSV.read(joinpath(ORIGINAL,"stock_scores.csv"),DataFrame)
results=NamedTuple[];diagnostics=NamedTuple[];mixing=NamedTuple[];daily=NamedTuple[]
variance_audit=NamedTuple[]
training_data=JLD2.load(joinpath(ROOT,"code/data/equity/SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2"),"dataset")
checks=NamedTuple[];examples=NamedTuple[];models=NamedTuple[]
input_hashes=Dict(f=>(open(sha256,joinpath(ROOT,f)) |> bytes2hex) for f in
    ["code/data/pretrained-portfolio-surrogate.jld2","code/results/chronological_validation/stock_sessions.csv",
     "code/data/equity/SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2",
     "code/results/stock_forecast_diagnosis/PROTOCOL.md","code/scripts/diagnose_stock_forecasts.jl",
     "code/src/StockForecastDiagnosis.jl"])

function checkpoints(model,observations,shift,until,components)
    first_day=minimum(keys(observations));mass=initial_mass(model,components)
    saved=Dict(first_day=>copy(mass));returns=Tuple{Date,Float64}[]
    previous=first_day
    while advance_session(previous)<=until
        day=advance_session(previous)
        mass=propagate(model,mass,components)
        if haskey(observations,day) && haskey(observations,previous)
            lr=log(observations[day]/observations[previous])
            mass=condition(model,mass,lr/model.dt-model.rf-shift)
            push!(returns,(day,lr))
        end
        saved[day]=copy(mass);previous=day
    end
    saved,returns
end

@testset "Stock diagnostic reconstruction and causality" begin
for (ticker_index,ticker) in enumerate(["GS","LLY"])
    model=portfolio["marginals"][ticker];components=jump_components(model)
    historical_returns=diff(log.(Float64.(training_data[ticker].close)))
    locations=[e.μ*model.dt for e in model.emissions]
    scales=[e.σ*model.dt for e in model.emissions]
    location=sum(model.stationary.*locations)
    between=sum(model.stationary.*(locations.-location).^2)
    within=sum(model.stationary.*scales.^2)
    push!(variance_audit,(ticker,training_daily_sd=std(historical_returns),
        training_daily_mean=mean(historical_returns),stationary_daily_mean=location,
        stationary_daily_sd=sqrt(between+within*model.ν/(model.ν-2)),
        between_state_variance=between,within_state_variance_unscaled=within,
        scale_corrected_sd=sqrt(between+within),fallback_states=count(e->e.is_fallback,model.emissions),
        fallback_probability=sum(model.stationary[[e.is_fallback for e in model.emissions]])))
    observations=Dict(r.session=>r.spot for r in eachrow(stocks[stocks.ticker.==ticker,:]))
    shift=pilots[ticker]["growth_shift"];fixed_sd=pilots[ticker]["log_return_sd"]
    daily_mean=pilots[ticker]["log_return_mean"]
    states,returns=checkpoints(model,observations,shift,LAST,components)
    prefix=Date("2026-08-04")
    prefix_states,_=checkpoints(model,filter(p->p.first<=prefix,observations),shift,prefix,components)
    @test prefix_states[prefix]==states[prefix]
    @test all(isapprox(sum(a),1.;atol=1e-10) for a in values(states))
    for (date,lr) in returns
        push!(daily,(ticker,date,log_return=lr))
    end
    push!(models,(ticker,n_states=model.partition.N,nu=model.ν,dt=model.dt,rf=model.rf,
        jump_probability=model.jump.ϵ,jump_length_mean=model.jump.λ,
        jump_probability_first_five=1-(1-model.jump.ϵ*(1-exp(-model.jump.λ)))^4,
        self_transition_mean=mean(diag(model.transition)),pilot_daily_sd=fixed_sd,
        growth_shift=shift,training_days=portfolio["n_training_days"]))
    origins=sort([d for d in keys(observations) if prefix<=d && advance_session(d)<=LAST])
    for origin in origins
        S0=observations[origin];posterior=states[origin]
        history=[v for (date,v) in returns if date<=origin]
        @assert length(history)>=20
        recent_sd=std(history[end-19:end]);scale=recent_sd/fixed_sd
        push!(diagnostics,(ticker,origin,history_returns=length(history),recent_daily_sd=recent_sd,
            scale,posterior_jump_remaining=sum(posterior[:,2:end]),
            largest_state_probability=maximum(vec(sum(posterior;dims=2)))))
        base=initial_mass(model,components);conditional=copy(posterior)
        for h in 1:5
            base=propagate(model,base,components);conditional=propagate(model,conditional,components)
            p_base=vec(sum(base;dims=2));p_cond=vec(sum(conditional;dims=2))
            means=[(e.μ+shift+model.rf)*model.dt for e in model.emissions]
            push!(mixing,(ticker,origin,horizon=h,total_variation=.5sum(abs.(base.-conditional)),
                stationary_daily_mean=sum(p_base.*means),filtered_daily_mean=sum(p_cond.*means)))
        end
        # Reproduce the exact saved first-origin simulation before changing it.
        if origin==first(origins)
            seed=202600000+1000ticker_index+Dates.value(origin-Date("2026-01-01"))
            sim=JumpHMM.simulate(model,5;n_paths=1000,seed)
            g=hcat([p.observations for p in sim.paths]...)
            prices=S0.*exp.(cumsum((g.+shift.+model.rf).*model.dt;dims=1))
            for h in [1,5]
                mask=(old_scores.ticker.==ticker).&(Date.(old_scores.origin).==origin)
                mask .&= (old_scores.horizon.==h).&(old_scores.mode.=="JumpHMM")
                row=only(eachrow(old_scores[mask,:]))
                delta=mean(prices[h,:])-row.predicted_mean
                @test abs(delta)<1e-9
                push!(checks,(ticker,origin,horizon=h,mean_difference=delta))
            end
        end
        for (replicate,base_seed) in enumerate(SEEDS)
            seed=base_seed+1000ticker_index+10000Dates.value(origin-Date("2026-01-01"))
            sim=JumpHMM.simulate(model,5;n_paths=N,seed)
            legacy=(hcat([p.observations for p in sim.paths]...).+shift.+model.rf).*model.dt
            stationary=simulate_forward(model,initial_mass(model,components),N,5,seed;shift,components)
            filtered=simulate_forward(model,posterior,N,5,seed;shift,components)
            adapted=daily_mean.+scale.*(filtered.-daily_mean)
            normal=randn(MersenneTwister(seed+1),5,N)
            variants=[("legacy_stationary",legacy),("stationary_transition",stationary),
                ("filtered_state",filtered),("filtered_state_vol20",adapted),
                ("rw_fixed",fixed_sd.*normal),("rw_vol20",recent_sd.*normal),
                ("unchanged",zeros(5,N))]
            for (method,lr) in variants
                prices=S0.*exp.(cumsum(lr;dims=1))
                @assert all(isfinite,prices)
                for h in [1,5]
                    endpoint=advance_session(origin,h)
                    endpoint<=LAST && haskey(observations,endpoint) || continue
                    observed=observations[endpoint];score=distribution_scores(prices[h,:],observed)
                    push!(results,merge((ticker,origin,endpoint,horizon=h,method,replicate,seed,
                        paths=N,origin_spot=S0,observed,actual_return_pct=100(observed/S0-1),
                        median_absolute_error=abs(score.predicted_median-observed),
                        crps_pct=100score.crps/S0,mean_error_pct=100score.error/S0,
                        realized_percentile=mean(prices[h,:].<=observed)),score))
                end
                if origin==first(origins) && replicate==1
                    for h in 1:5
                        day=advance_session(origin,h);s=distribution_scores(prices[h,:],S0)
                        push!(examples,(ticker,origin,day,horizon=h,method,mean=s.predicted_mean,
                            median=s.predicted_median,lo=s.predicted_q05,hi=s.predicted_q95,
                            observed=get(observations,day,NaN)))
                    end
                end
            end
        end
        println(ticker," ",origin," recent/fixed volatility=",round(scale;digits=2));flush(stdout)
    end
end
end
for (name,rows) in [("scores",results),("origin_diagnostics",diagnostics),("state_memory",mixing),
    ("daily_returns",daily),("reconstruction",checks),("examples",examples),("model_settings",models),
    ("training_variance_audit",variance_audit)]
    CSV.write(joinpath(OUT,name*".csv"),DataFrame(rows))
end
open(joinpath(OUT,"manifest.toml"),"w") do io
    TOML.print(io,Dict("completed"=>true,"paths"=>N,"seeds"=>SEEDS,"input_sha256"=>input_hashes))
end
println("Stock-only diagnosis complete.")
