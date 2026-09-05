"""Regenerate the two fitted arXiv illustrations, figures, and table bodies."""
ENV["GKSwstype"]="100"
using Dates, CSV, DataFrames, Printf, Statistics
include(joinpath(@__DIR__,"..","src","ScenarioTemplate.jl"))
using .ScenarioTemplate
const ROOT=abspath(joinpath(@__DIR__,"..",".."))
const OUT=joinpath(ROOT,"code","results","fitted_scenarios")
mkpath(OUT)

function write_table(result,path)
    s=result.spec; a=result.summary_row
    values=[
        ("Strike",s.K_put,s.K_call),
        ("Market delta",s.market_delta_put,s.market_delta_call),
        ("Market IV",100s.market_iv_put,100s.market_iv_call),
        ("Market mid",s.market_premium_put,s.market_premium_call),
        ("Model entry value",a.model_fv_put,a.model_fv_call),
        ("Model value minus market mid",a.entry_edge_put,a.entry_edge_call),
        (raw"Mean P\&L",a.mean_pnl_put,a.mean_pnl_call),
        ("Monte Carlo SE of mean",a.mean_se_put,a.mean_se_call),
        (raw"Median P\&L",a.median_pnl_put,a.median_pnl_call),
        ("Standard deviation",std(result.pnl_put),std(result.pnl_call)),
        (raw"5\% quantile",a.q05_pnl_put,a.q05_pnl_call),
        (raw"5\% expected shortfall",a.es05_pnl_put,a.es05_pnl_call),
        ("Premium retained",a.pct_premium_kept_put,a.pct_premium_kept_call),
        ("Delta-heuristic premium retained",100(1-abs(s.market_delta_put)),100(1-abs(s.market_delta_call))),
        ("Simulated minus heuristic",a.pct_premium_kept_put-100(1-abs(s.market_delta_put)),
             a.pct_premium_kept_call-100(1-abs(s.market_delta_call)))]
    open(path,"w") do io
        println(io,raw"\begin{tabular}{lrr}\toprule")
        println(io,raw"Statistic & Short put & Short call \\\midrule")
        @printf(io,"Initial spot & \\multicolumn{2}{c}{%.2f} \\\\\n",result.S_0)
        for (label,p,c) in values
            if label=="Market delta"
                @printf(io,"%s & %+.3f & %+.3f \\\\\n",label,p,c)
            elseif label in ("Market IV","Premium retained","Delta-heuristic premium retained")
                @printf(io,"%s & %.1f\\%% & %.1f\\%% \\\\\n",label,p,c)
            elseif label=="Simulated minus heuristic"
                @printf(io,"%s & %+.1f pp & %+.1f pp \\\\\n",label,p,c)
            else
                @printf(io,"%s & %.2f & %.2f \\\\\n",label,p,c)
            end
            if label=="Premium retained"
                p=a.pct_premium_kept_ci_put; c=a.pct_premium_kept_ci_call
                @printf(io,"Wilson 95\\%% interval & %.1f--%.1f\\%% & %.1f--%.1f\\%% \\\\\n",p[1],p[2],c[1],c[2])
            end
        end
        println(io,raw"\bottomrule\end{tabular}")
    end
end

rows=NamedTuple[]
for ticker in ("GS","LLY")
    gs=ticker=="GS"
    spec=ScenarioSpec(ticker=ticker,anchor_date=Date("2026-04-28"),expiry_date=Date("2026-05-29"),
        K_put=gs ? 890.0 : 825.0,K_call=gs ? 970.0 : 940.0,
        market_premium_put=gs ? 16.51 : 23.30,market_premium_call=gs ? 16.085 : 20.76,
        market_iv_put=gs ? 0.3125 : 0.444,market_iv_call=gs ? 0.2893 : 0.440,
        market_delta_put=gs ? -0.2951 : -0.303,market_delta_call=gs ? 0.3278 : 0.309,
        expiry_label="2026-05-29",ticker_prior_ccgr_pct=10.0,
        market_holidays=[Date("2026-05-25")],n_paths=1000,seed=20260429)
    result=run_short_scenario(spec,VarianceSpec();
        nn_cache_path=joinpath(ROOT,"code","figures","calibrate_ladders_per_ticker_nn_cache.jld2"),
        port_path=joinpath(ROOT,"code","data","pretrained-portfolio-surrogate.jld2"),
        ladder_dir=joinpath(ROOT,"code","data","ladder"),
        sim_cache_path=joinpath(OUT,lowercase(ticker)*"_cache.jld2"))
    render_scenario_figures(result,spec;plot_dir=joinpath(ROOT,"paper-arxiv","sections","figures",lowercase(ticker)))
    write_table(result,joinpath(OUT,lowercase(ticker)*"_table.tex"))
    push!(rows,result.summary_row)
    CSV.write(joinpath(OUT,"summary.csv"),DataFrame(rows))
    ScenarioTemplate.print_summary(result)
end
