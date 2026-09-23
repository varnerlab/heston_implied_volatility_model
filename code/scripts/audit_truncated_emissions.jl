# Report training-fitted emission bounds and validate the bounded simulations.
using HestonIV, JLD2, JumpHMM, CSV, DataFrames, Distributions, Statistics, TOML
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const OUT=joinpath(ROOT,"code/results/truncated_emissions")
portfolio=JLD2.load(joinpath(ROOT,"code/data/pretrained-portfolio-surrogate.jld2"))
rows=NamedTuple[]
for cutoff in [10.,20.], ticker in ["GS","LLY"]
    spec=TruncatedStudentT(cutoff);model=portfolio["marginals"][ticker]
    result_root=cutoff==10 ? joinpath(ROOT,"code/results") : joinpath(OUT,"wide")
    pilot=TOML.parsefile(joinpath(result_root,"chronological_validation/pilot_constants.toml"))[ticker]
    @assert pilot["emission_spec"]==emission_metadata(spec)
    shift=pilot["growth_shift"]
    lower=minimum((e.μ-cutoff*e.σ+shift+model.rf)*model.dt for e in model.emissions)
    upper=maximum((e.μ+cutoff*e.σ+shift+model.rf)*model.dt for e in model.emissions)
    omitted=sum(model.stationary[k]*2ccdf(TDist(e.ν),cutoff)
                for (k,e) in enumerate(model.emissions))
    simulated=simulate_truncated(model,22;n_paths=10000,seed=20260914,emissions=spec)
    residuals=[(p.observations[t]-model.emissions[p.states[t]].μ)/model.emissions[p.states[t]].σ
               for p in simulated.paths for t in eachindex(p.states)]
    @assert maximum(abs,residuals)<=cutoff+1e-12
    push!(rows,(ticker,cutoff,original_probability_removed=omitted,
        min_daily_log_return=lower,max_daily_log_return=upper,
        min_daily_return_pct=100expm1(lower),max_daily_return_pct=100expm1(upper),
        conditional_residual_variance=truncated_t_variance(model.ν,spec),
        simulated_max_abs_residual=maximum(abs,residuals),growth_shift=shift))
end
CSV.write(joinpath(OUT,"bounds.csv"),DataFrame(rows))
show(stdout,MIME("text/plain"),DataFrame(rows));println()
