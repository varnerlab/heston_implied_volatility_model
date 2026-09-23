# Diagnostics behind the Discussion's explanations: IV-factor persistence, central stock drift,
# mean IV movement, realized versus model stock volatility, and held-out surface bias around earnings.
using CSV,DataFrames,Dates,JLD2,Printf,Statistics,TOML
include(joinpath(@__DIR__,"..","src","DynamicAblation.jl"))
using .DynamicAblation
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const CHRONO=joinpath(ROOT,"code/results/chronological_validation")
const SHORT=joinpath(CHRONO,"short_maturity")
const STOCK=joinpath(ROOT,"code/results/small_stock_comparison")
const GEN=joinpath(ROOT,"paper-arxiv/sections/generated")
const TICKERS=["GS","LLY"]
rows=NamedTuple[]
record!(check,ticker,value,unit)=push!(rows,(;check,ticker,value,unit))

# Deterministic relaxation toward a unit step in the target, using the scenario factor defaults.
steps=20
targets=ones(steps+2,1,1);targets[2:end,:,:].=2.0
v=variance_paths(targets,zeros(steps+1,1),zeros(steps+1,1);mode=:relaxation)
closed(h)=v[2+h,1,1]-1.0
retention=1-closed(1)
record!("factor_half_life_sessions","all",log(.5)/log(retention),"sessions")
for h in [5,10]
    record!("factor_gap_closed_$(h)_sessions","all",100closed(h),"percent")
end

# Five-session forecasts on the scored short-maturity origins and matched contracts.
scores=CSV.read(joinpath(SHORT,"forecast_scores.csv"),DataFrame)
scored=scores[(scores.horizon.==5).&(scores.conditioning.=="joint").&(scores.mode.=="coupled"),:]
pilots=TOML.parsefile(joinpath(CHRONO,"pilot_constants.toml"))
for ticker in TICKERS
    sub=scored[scored.ticker.==ticker,:]
    median_move=Float64[];log_sd=Float64[];mean_gap=Float64[];path_gap=Float64[];target_gap=Float64[]
    for origin in sort(unique(Date.(sub.origin)))
        cache=JLD2.load(joinpath(SHORT,"paths_$(lowercase(ticker))_$(origin).jld2"))
        S=cache["stock"];vs=cache["variances"];cs=cache["contracts"]
        push!(median_move,100*(median(S[6,:])/S[1,1]-1))
        push!(log_sd,100*std(log.(S[6,:]./S[1,1])))
        for symbol in sub.symbol[Date.(sub.origin).==origin]
            c=only(findall(cs.symbol.==symbol))
            origin_iv=sqrt(vs[:frozen][1,1,c])
            coupled=sqrt.(vs[:coupled][6,:,c]);surface=sqrt.(vs[:surface][6,:,c])
            push!(mean_gap,100*(mean(coupled)-origin_iv))
            push!(path_gap,100*mean(abs.(coupled.-origin_iv)))
            push!(target_gap,100*mean(abs.(surface.-origin_iv)))
        end
    end
    record!("jumphmm_median_move_5",ticker,mean(median_move),"percent")
    record!("jumphmm_log_sd_5",ticker,mean(log_sd),"percent")
    record!("pilot_log_drift_5",ticker,100*5*pilots[ticker]["log_return_mean"],"percent")
    record!("coupled_mean_iv_change_5",ticker,mean(mean_gap),"vol points")
    record!("coupled_abs_iv_change_5",ticker,mean(path_gap),"vol points")
    record!("surface_abs_iv_change_5",ticker,mean(target_gap),"vol points")
    record!("pilot_one_session_sd",ticker,100*pilots[ticker]["log_return_sd"],"percent")
end

# One-session realized and EWMA volatility at the stock-benchmark origins.
inputs=CSV.read(joinpath(STOCK,"forecast_inputs.csv"),DataFrame)
for ticker in TICKERS, period in [2025,2026]
    sub=inputs[(inputs.ticker.==ticker).&(inputs.period.==period).&(inputs.horizon.==1),:]
    record!("realized_one_session_sd_$(period)",ticker,100*std(log.(sub.observed./sub.origin_spot)),"percent")
    record!("ewma_one_session_sd_$(period)",ticker,100*mean(sqrt.(sub.daily_variance)),"percent")
end

# Held-out July-cutoff surface bias before and after LLY's August 5 report.
calendar=CSV.read(joinpath(ROOT,"code/data/earnings/earnings_calendar.csv"),DataFrame)
@assert Date("2026-08-05") in Date.(calendar.earnings_date[calendar.ticker.=="LLY"])
@assert Date("2026-07-14") in Date.(calendar.earnings_date[calendar.ticker.=="GS"])
bias=CSV.read(joinpath(CHRONO,"surface_scores_by_date.csv"),DataFrame)
for ticker in TICKERS
    test=bias[(bias.split.=="test").&(bias.ticker.==ticker),:]
    before=only(test.bias_pp[Date.(test.session).==Date("2026-08-04")])
    after=test[Date.(test.session).>=Date("2026-08-06"),:]
    record!("surface_bias_aug04",ticker,before,"vol points")
    record!("surface_bias_aug05",ticker,only(test.bias_pp[Date.(test.session).==Date("2026-08-05")]),"vol points")
    record!("surface_bias_after_mean",ticker,sum(after.bias_pp.*after.n)/sum(after.n),"vol points")
    record!("surface_bias_after_min",ticker,minimum(after.bias_pp),"vol points")
    record!("surface_bias_after_max",ticker,maximum(after.bias_pp),"vol points")
end

table=DataFrame(rows)
CSV.write(joinpath(CHRONO,"mechanism_checks.csv"),table)
value(check,ticker)=only(table.value[(table.check.==check).&(table.ticker.==ticker)])
fmt(x;digits=2)=replace(@sprintf("%.*f",digits,x),"-"=>"\$-\$")
line(label,check;digits=2)=string(label," & ",join((fmt(value(check,t);digits) for t in TICKERS)," & ")," \\\\\n")
range_cell(t)=string(fmt(value("surface_bias_after_min",t))," to ",fmt(value("surface_bias_after_max",t)))
open(joinpath(GEN,"mechanism_checks_table.tex"),"w") do io
    print(io,"\\begin{tabular}{lrr}\n\\toprule\nQuantity & GS & LLY \\\\\n\\midrule\n")
    print(io,"\\multicolumn{3}{l}{\\textit{Five-session forecasts, short-maturity cohort}} \\\\\n")
    print(io,line("JumpHMM median price change (\\%)","jumphmm_median_move_5"))
    print(io,line("JumpHMM log-return standard deviation (\\%)","jumphmm_log_sd_5"))
    print(io,line("Coupled mean IV minus origin IV (points)","coupled_mean_iv_change_5"))
    print(io,line("Coupled mean absolute IV change (points)","coupled_abs_iv_change_5"))
    print(io,line("Direct-surface mean absolute IV change (points)","surface_abs_iv_change_5"))
    print(io,"\\midrule\n\\multicolumn{3}{l}{\\textit{One-session stock return standard deviation (\\%)}} \\\\\n")
    print(io,line("JumpHMM pilot","pilot_one_session_sd"))
    print(io,line("Realized, 2025","realized_one_session_sd_2025"))
    print(io,line("EWMA at origin, 2025","ewma_one_session_sd_2025"))
    print(io,line("Realized, August--September 2026","realized_one_session_sd_2026"))
    print(io,line("EWMA at origin, August--September 2026","ewma_one_session_sd_2026"))
    print(io,"\\midrule\n\\multicolumn{3}{l}{\\textit{Held-out July-cutoff surface bias (points)}} \\\\\n")
    print(io,line("August 4","surface_bias_aug04"))
    print(io,line("August 5","surface_bias_aug05"))
    print(io,line("August 6--September 4, mean","surface_bias_after_mean"))
    print(io,"August 6--September 4, range & ",range_cell("GS")," & ",range_cell("LLY")," \\\\\n")
    print(io,"\\bottomrule\n\\end{tabular}\n")
end
show(table;allrows=true)
println()
