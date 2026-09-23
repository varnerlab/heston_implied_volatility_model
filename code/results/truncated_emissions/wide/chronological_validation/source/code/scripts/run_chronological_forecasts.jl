using HestonIV: simulate_truncated, emission_spec_from_env, emission_metadata
const EMISSIONS = emission_spec_from_env()
# Fixed one- and five-session stock/IV/option forecasts and observed-path diagnostics.
using CSV, DataFrames, Dates, JLD2, JumpHMM, Statistics, Random, SHA, TOML, LinearAlgebra
include(joinpath(@__DIR__,"..","src","TemporalFolds.jl"))
include(joinpath(@__DIR__,"..","src","DynamicAblation.jl"))
include(joinpath(@__DIR__,"..","src","ForecastValidation.jl"))
using .TemporalFolds, .DynamicAblation, .ForecastValidation
BLAS.set_num_threads(1)
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const OUT=joinpath(ROOT,"code/results/chronological_validation")
const SMOKE="--smoke" in ARGS
const N=SMOKE ? 32 : 1000
const COHORT=get(ENV,"FORECAST_COHORT","monthly")
@assert COHORT in ("monthly","short")
const COHORT_OUT=COHORT=="short" ? joinpath(OUT,"short_maturity") : OUT
const RESULT_BASE=joinpath(get(ENV,"SIMULATION_RESULTS_ROOT",joinpath(ROOT,"code/results")),"chronological_validation")
const RESULT_COHORT=COHORT=="short" ? joinpath(RESULT_BASE,"short_maturity") : RESULT_BASE
const RUNOUT=SMOKE ? joinpath(RESULT_COHORT,"smoke") : RESULT_COHORT
mkpath(RUNOUT)
open(joinpath(RUNOUT,"run_manifest.toml"),"w") do io
    TOML.print(io,Dict("completed"=>false,"emission_spec"=>emission_metadata(EMISSIONS)))
end
const LAST=Date("2026-09-04")
const MODEL=JLD2.load(joinpath(OUT,"surface_model.jld2"))
@assert MODEL["input_hash"] == (open(sha256,joinpath(OUT,"training.csv")) |> bytes2hex)
contracts=CSV.read(joinpath(OUT,COHORT=="short" ? "origin_contracts_short.csv" : "origin_contracts.csv"),DataFrame)
quotes=CSV.read(joinpath(OUT,"market_quotes.csv"),DataFrame)
stocks=CSV.read(joinpath(OUT,"stock_sessions.csv"),DataFrame)
stockmap=Dict((r.ticker,Date(r.session))=>r.spot for r in eachrow(stocks))
quotemap=Dict((r.ticker,Date(r.session),r.symbol)=>r for r in eachrow(quotes))
portfolio=JLD2.load(joinpath(ROOT,"code/data/pretrained-portfolio-surrogate.jld2"))

function growth(marginal,n,seed)
    sim=simulate_truncated(marginal,5;n_paths=n,seed,emissions=EMISSIONS)
    g=hcat([p.observations for p in sim.paths]...)
    @assert size(g)==(5,n)
    g
end
function surface_values(ticker,S,dates,cs)
    sm=MODEL["sector_models"][TemporalFolds.SECTORS[ticker]]
    scaler=MODEL["standardizer"]
    values=Array{Float64}(undef,size(S,1),size(S,2),nrow(cs))
    level=sm.model.log_theta[sm.ticker_idx[ticker]]
    for (c,contract) in enumerate(eachrow(cs))
        dtes=Dates.value.(Date(contract.expiry).-dates)
        X=Matrix{Float32}(undef,2,length(S))
        for p in axes(S,2),t in axes(S,1)
            j=t+(p-1)*size(S,1)
            X[1,j]=(log(max(dtes[t],1))-scaler.mu_dte)/scaler.sigma_dte
            X[2,j]=(log(contract.strike/S[t,p])-scaler.mu_m)/scaler.sigma_m
        end
        values[:,:,c]=reshape(Float64.(exp.(level .+ vec(sm.model.psi_nn(X)))),size(S))
    end
    values
end
function marks_at(S,v,dates,cs,t)
    out=Matrix{Float64}(undef,size(S,2),nrow(cs))
    for (c,r) in enumerate(eachrow(cs)),p in axes(S,2)
        out[p,c]=option_mark(S[t,p],r.strike,v[t,p,c],Dates.value(Date(r.expiry)-dates[t]),Symbol(r.kind))
    end
    out
end

predictions=NamedTuple[];stock_scores=NamedTuple[];availability=NamedTuple[]
numerical=NamedTuple[];figure=NamedTuple[];diagnostics=NamedTuple[];pilots=Dict{String,Any}()
start=time()
for (ticker_index,ticker) in enumerate(["GS","LLY"])
    marginal=portfolio["marginals"][ticker]
    pilot=growth(marginal,10000,202609050+ticker_index)
    shift=.10-mean(pilot)
    pilot_returns=(pilot .+ shift .+ marginal.rf).*marginal.dt
    mu=mean(pilot_returns);scale=std(pilot_returns)
    pilots[ticker]=Dict("growth_shift"=>shift,"log_return_mean"=>mu,"log_return_sd"=>scale,
        "pilot_seed"=>202609050+ticker_index,"paths"=>10000,"steps"=>5,
        "emission_spec"=>emission_metadata(EMISSIONS))
    origins=sort(unique(Date.(contracts.origin[contracts.ticker .== ticker])))
    SMOKE && (origins=origins[1:1])
    first_example=nothing
    for origin in origins
        advance_session(origin,1)>LAST && continue
        cs=sort(contracts[(contracts.ticker .== ticker).&(Date.(contracts.origin).==origin),:],:kind)
        dates=session_path(origin,5);S0=only(unique(cs.spot))
        seed=202600000 + 1000ticker_index + Dates.value(origin-Date("2026-01-01"))
        g=growth(marginal,N,seed)
        lr=(g .+ shift .+ marginal.rf).*marginal.dt
        S=vcat(fill(S0,1,N),S0.*exp.(cumsum(lr;dims=1)))
        @assert S[:,1]≈JumpHMM.prices_from_growth_rates(g[:,1].+shift,S0;rf=marginal.rf,dt=marginal.dt)
        zs=(lr.-mu)./scale;zi=randn(MersenneTwister(seed+1),5,N)
        rw=vcat(fill(S0,1,N),S0.*exp.(cumsum(scale.*randn(MersenneTwister(seed+2),5,N);dims=1)))
        targets=anchored_targets(surface_values(ticker,S,dates,cs),cs.origin_iv.^2)
        variances=Dict(mode=>variance_paths(targets,zs,zi;mode) for mode in MODES)
        JLD2.jldsave(joinpath(RUNOUT,"paths_$(lowercase(ticker))_$(origin).jld2");
            ticker,origin,dates,seed,stock=S,variances,contracts=cs,independent_shocks=zi,
            emission_spec=emission_metadata(EMISSIONS),
            model_hash=(open(sha256,joinpath(OUT,"surface_model.jld2")) |> bytes2hex))
        for mode in MODES, c in 1:nrow(cs)
            m=cs.strike[c]./S
            push!(diagnostics,(ticker,origin,kind=cs.kind[c],mode=string(mode),
                floor_fraction=mean(variances[mode][2:end,:,c].<=.005^2),
                outside_moneyness_fraction=mean((m.<.8).|(m.>1.2))))
        end
        endpoint_valid=dates[6]<=LAST && all(haskey(quotemap,(ticker,dates[6],r.symbol)) for r in eachrow(cs))
        is_example=ticker=="GS" && first_example===nothing && nrow(cs)==2 && endpoint_valid
        is_example && (first_example=origin)
        println(now()," Forecast ",ticker," ",origin," n=",N," elapsed=",round(time()-start;digits=1));flush(stdout)
        for horizon in [1,5]
            endpoint=dates[horizon+1]
            endpoint>LAST && continue
            stock_present=haskey(stockmap,(ticker,endpoint))
            observed_sequence=all(haskey(stockmap,(ticker,d)) for d in dates[1:horizon+1])
            observed_spot=stock_present ? stockmap[(ticker,endpoint)] : NaN
            for (mode,values) in [("JumpHMM",S),("Random walk",rw)]
                if stock_present
                    push!(stock_scores,merge((ticker,origin,endpoint,horizon,mode,observed=observed_spot),
                        distribution_scores(values[horizon+1,:],observed_spot)))
                end
            end
            observed_variances=nothing
            if observed_sequence
                observed_S=repeat([stockmap[(ticker,d)] for d in dates[1:horizon+1]],1,N)
                observed_z=repeat((diff(log.(observed_S[:,1])).-mu)./scale,1,N)
                observed_targets=anchored_targets(surface_values(ticker,observed_S,dates[1:horizon+1],cs),cs.origin_iv.^2)
                observed_variances=Dict(mode=>variance_paths(observed_targets,observed_z,zi[1:horizon,:];mode) for mode in MODES)
            end
            for (c,r) in enumerate(eachrow(cs))
                matched=haskey(quotemap,(ticker,endpoint,r.symbol))
                push!(availability,(ticker,origin,endpoint,horizon,kind=r.kind,symbol=r.symbol,
                    quote_matched=matched,stock_matched=stock_present,complete_stock_path=observed_sequence))
            end
            main_marks=Dict(mode=>marks_at(S,variances[mode],dates,cs,horizon+1) for mode in MODES)
            for (c,r) in enumerate(eachrow(cs))
                haskey(quotemap,(ticker,endpoint,r.symbol)) || continue
                actual=quotemap[(ticker,endpoint,r.symbol)]
                for mode in MODES
                    marks=main_marks[mode][:,c]
                    push!(predictions,merge((ticker,origin,endpoint,horizon,kind=r.kind,symbol=r.symbol,
                        mode=string(mode),conditioning="joint",observed=actual.mid,
                        bid=actual.bid,ask=actual.ask,origin_mid=r.origin_mid,
                        inside_spread=actual.bid<=mean(marks)<=actual.ask,
                        paired_mc_se=std(marks.-main_marks[:frozen][:,c])/sqrt(N)),
                        distribution_scores(marks,actual.mid)))
                    if origin==first(origins)
                        for p in 1:min(10,N)
                            fine=option_mark(S[horizon+1,p],r.strike,variances[mode][horizon+1,p,c],
                                Dates.value(Date(r.expiry)-endpoint),Symbol(r.kind);depth=401)
                            push!(numerical,(ticker,origin,horizon,kind=r.kind,mode=string(mode),path=p,
                                price201=marks[p],price401=fine,absolute_change=abs(marks[p]-fine)))
                        end
                    end
                    if observed_sequence
                        om=[option_mark(observed_spot,r.strike,observed_variances[mode][end,p,c],
                            Dates.value(Date(r.expiry)-endpoint),Symbol(r.kind)) for p in 1:N]
                        push!(predictions,merge((ticker,origin,endpoint,horizon,kind=r.kind,symbol=r.symbol,
                            mode=string(mode),conditioning="observed_stock",observed=actual.mid,
                            bid=actual.bid,ask=actual.ask,origin_mid=r.origin_mid,
                            inside_spread=actual.bid<=mean(om)<=actual.ask,paired_mc_se=NaN),
                            distribution_scores(om,actual.mid)))
                    end
                end
                if !ismissing(actual.implied_vol) && isfinite(actual.implied_vol) && .01<actual.implied_vol<2
                    mark=option_mark(observed_spot,r.strike,actual.implied_vol^2,
                        Dates.value(Date(r.expiry)-endpoint),Symbol(r.kind))
                    push!(predictions,merge((ticker,origin,endpoint,horizon,kind=r.kind,symbol=r.symbol,
                        mode="reported_iv",conditioning="endpoint_repricing",observed=actual.mid,
                        bid=actual.bid,ask=actual.ask,origin_mid=r.origin_mid,
                        inside_spread=actual.bid<=mark<=actual.ask,paired_mc_se=NaN),
                        distribution_scores(fill(mark,N),actual.mid)))
                end
            end
        end
        if is_example
            for t in 1:6
                observed=get(stockmap,(ticker,dates[t]),NaN)
                summary=distribution_scores(S[t,:],isfinite(observed) ? observed : S0)
                push!(figure,(ticker,origin,date=dates[t],step=t-1,series="stock",symbol="",mode="coupled",
                    predicted_mean=summary.predicted_mean,lo=summary.predicted_q05,hi=summary.predicted_q95,
                    observed,bid=NaN,ask=NaN))
                for mode in [:frozen,:coupled]
                    marks=marks_at(S,variances[mode],dates,cs,t)
                    for (c,r) in enumerate(eachrow(cs))
                        actual=get(quotemap,(ticker,dates[t],r.symbol),nothing)
                        observed=actual===nothing ? NaN : actual.mid
                        summary=distribution_scores(marks[:,c],isfinite(observed) ? observed : r.origin_mid)
                        push!(figure,(ticker,origin,date=dates[t],step=t-1,series=r.kind,symbol=r.symbol,mode=string(mode),
                            predicted_mean=summary.predicted_mean,lo=summary.predicted_q05,hi=summary.predicted_q95,
                            observed,bid=actual===nothing ? NaN : actual.bid,ask=actual===nothing ? NaN : actual.ask))
                    end
                end
            end
        end
        # Save completed origins so failures cannot erase earlier diagnostics.
        CSV.write(joinpath(RUNOUT,"forecast_scores.csv"),DataFrame(predictions))
        CSV.write(joinpath(RUNOUT,"stock_scores.csv"),DataFrame(stock_scores))
        CSV.write(joinpath(RUNOUT,"endpoint_availability.csv"),DataFrame(availability))
        CSV.write(joinpath(RUNOUT,"numerical_check.csv"),DataFrame(numerical))
        CSV.write(joinpath(RUNOUT,"factor_diagnostics.csv"),DataFrame(diagnostics))
        !isempty(figure) && CSV.write(joinpath(RUNOUT,"forecast_example.csv"),DataFrame(figure))
    end
end
open(joinpath(RUNOUT,"pilot_constants.toml"),"w") do io;TOML.print(io,pilots);end
metadata=Dict("emission_spec"=>emission_metadata(EMISSIONS),"cohort"=>COHORT,"paths_per_origin"=>N,"model_sha256"=>(open(sha256,joinpath(OUT,"surface_model.jld2")) |> bytes2hex),
    "protocol_sha256"=>(open(sha256,joinpath(OUT,"PROTOCOL.md")) |> bytes2hex),
    "source_sha256"=>Dict(f=>(open(sha256,joinpath(ROOT,f)) |> bytes2hex) for f in
        ["code/scripts/run_chronological_forecasts.jl","code/src/ForecastValidation.jl","code/src/DynamicAblation.jl","code/src/TruncatedEmissions.jl","code/src/HestonIV.jl","code/Manifest.toml",
         "code/results/truncated_emissions/PROTOCOL.md","code/data/pretrained-portfolio-surrogate.jld2"]),
    "elapsed_seconds"=>time()-start,"completed"=>true)
for relative in keys(metadata["source_sha256"])
    endswith(relative,".jl") || continue
    destination=joinpath(RUNOUT,"source",relative)
    mkpath(dirname(destination));cp(joinpath(ROOT,relative),destination;force=true)
end
open(joinpath(RUNOUT,"run_manifest.toml"),"w") do io;TOML.print(io,metadata);end
println("Completed fixed forecasts in ",round(time()-start;digits=1)," seconds.")
