"""
Run the fixed dynamic-IV sensitivity experiment from the repository root:
    julia --project=code code/examples/dynamic_iv_ablation.jl
Use --smoke for a small pipeline check, or --render-only to rebuild figures
and tables from saved outputs without repeating the simulations.
"""
ENV["GKSwstype"] = "100"
using Dates, CSV, DataFrames, JLD2, JumpHMM, Statistics, Random, Printf, SHA, TOML
using Plots
include(joinpath(@__DIR__, "..", "src", "ScenarioTemplate.jl"))
include(joinpath(@__DIR__, "..", "src", "DynamicAblation.jl"))
using .ScenarioTemplate, .DynamicAblation
const ST = ScenarioTemplate
const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const SMOKE = "--smoke" in ARGS
const OUT = joinpath(ROOT,"code","results",SMOKE ? "dynamic_ablation_smoke" : "dynamic_ablation")
const SEEDS = SMOKE ? [20260429] : [20260429,20260430,20260431]
const NPATHS = SMOKE ? 12 : 1000
const PILOT_PATHS = SMOKE ? 100 : 10000
const PILOT_SEED = 20260905
const STEPS = [0,5,10,15,20,22]
const LABELS = Dict(:frozen=>"Frozen IV", :surface=>"Direct surface",
    :relaxation=>"Mean reversion", :uncoupled=>"Noise, rho = 0", :coupled=>"Full factor")
const COLORS = Dict(:frozen=>:gray45,:surface=>:darkorange,:relaxation=>:seagreen,
                    :uncoupled=>:mediumpurple,:coupled=>:steelblue)

filehash(path) = open(sha256,path) |> bytes2hex

function growth_matrix(marginal,npaths,seed)
    sim=JumpHMM.simulate(marginal,22;n_paths=npaths,seed)
    @assert length(sim.paths)==npaths
    g=hcat([p.observations for p in sim.paths]...)
    @assert size(g)==(22,npaths)
    return g
end

function make_targets(model,S,dtes,strikes)
    targets=Array{Float64}(undef,size(S,1),size(S,2),length(strikes))
    for c in eachindex(strikes),p in axes(S,2),t in axes(S,1)
        targets[t,p,c]=model.theta_bar*ST._psi(model.psi_nn,model.standardizer,
                                             strikes[c],S[t,p],max(dtes[t],1))
    end
    return targets
end

function run_experiment()
    mkpath(OUT)
    nnpath=joinpath(ROOT,"code","figures","calibrate_ladders_per_ticker_nn_cache.jld2")
    portpath=joinpath(ROOT,"code","data","pretrained-portfolio-surrogate.jld2")
    ladder=joinpath(ROOT,"code","data","ladder")
    corpus=ST._load_all_ladders(ladder)
    @assert nrow(corpus)==234549 "Frozen calibration corpus changed"
    standardizer=ST._build_standardizer(corpus)
    nn=JLD2.load(nnpath); portfolio=JLD2.load(portpath)
    manifest=NamedTuple[]
    for (dir,_,files) in walkdir(ladder),file in sort(files)
        endswith(file,".csv") && occursin("_dte_ladder_",file) || continue
        p=joinpath(dir,file)
        raw=CSV.read(p,DataFrame); valid=ST._load_ladder(p)
        push!(manifest,(path=relpath(p,ROOT),sha256=filehash(p),raw_rows=nrow(raw),
              filtered_rows=nrow(valid),ticker=string(raw.underlying[1]),
              capture=string(raw.capture_ts[1]),session=string(raw.und_session_date[1])))
    end
    CSV.write(joinpath(OUT,"corpus_manifest.csv"),sort(DataFrame(manifest),:path))
    config=Dict("paths_per_seed"=>NPATHS,"seeds"=>SEEDS,"pilot_paths"=>PILOT_PATHS,
        "pilot_seed"=>PILOT_SEED,"steps"=>STEPS,"modes"=>collect(string.(MODES)),
        "kappa"=>15.0,"sigma_v"=>0.5,"rho"=>-0.6,"variance_floor"=>0.005^2,
        "r"=>0.0425,"q"=>0.0,"lr_depth"=>201,"julia_version"=>string(VERSION),
        "nn_sha256"=>filehash(nnpath),"portfolio_sha256"=>filehash(portpath),
        "corpus_manifest_sha256"=>filehash(joinpath(OUT,"corpus_manifest.csv")),
        "source_sha256"=>Dict(relpath(p,ROOT)=>filehash(p) for p in
           [@__FILE__,joinpath(ROOT,"code","src","DynamicAblation.jl"),
            joinpath(ROOT,"code","src","ScenarioTemplate.jl"),
            joinpath(ROOT,"code","src","LRTree.jl"),
            joinpath(ROOT,"code","src","CRRTree.jl"),
            joinpath(ROOT,"code","Manifest.toml")]))
    open(joinpath(OUT,"config.toml"),"w") do io; TOML.print(io,config); end
    # Preserve the exact source used to simulate, even if later prose or render
    # changes are made in the working tree before the paper is submitted.
    for relative in keys(config["source_sha256"])
        destination=joinpath(OUT,"source",relative)
        mkpath(dirname(destination))
        cp(joinpath(ROOT,relative),destination;force=true)
    end
    rows=NamedTuple[]; numerical=NamedTuple[]; audits=NamedTuple[]; diagnostics=NamedTuple[]
    for ticker in ("GS","LLY")
        strikes=ticker=="GS" ? [890.0,970.0] : [825.0,940.0]
        premiums=ticker=="GS" ? [16.51,16.085] : [23.30,20.76]
        kinds=[:put,:call]
        S0=ST._resolve_anchor_spot(corpus,ticker,Date("2026-04-28"))
        psi,logtheta,source=ST._restore_nn(nn,ticker;use_per_ticker=true,sector=nothing)
        model=ST._RestoredModel(psi,logtheta,exp(logtheta),source,standardizer)
        dates=[Date("2026-04-28"); [d for d in Date("2026-04-29"):Day(1):Date("2026-05-29")
               if dayofweek(d)<=5 && d!=Date("2026-05-25")]]
        dtes=Dates.value.(Date("2026-05-29") .- dates)
        marginal=portfolio["marginals"][ticker]
        pilot=growth_matrix(marginal,PILOT_PATHS,PILOT_SEED)
        shift=0.10-mean(pilot)
        pilot_log=(pilot .+ shift .+ marginal.rf).*marginal.dt
        mu=mean(pilot_log); scale=std(pilot_log)
        @assert scale>0
        open(joinpath(OUT,lowercase(ticker)*"_pilot.toml"),"w") do io
            TOML.print(io,Dict("growth_shift"=>shift,"log_return_mean"=>mu,
                "log_return_sd"=>scale,"marginal_rf"=>marginal.rf,"marginal_dt"=>marginal.dt))
        end
        for seed in SEEDS
            println("Ablation $ticker seed=$seed, $NPATHS paths"); flush(stdout)
            g=growth_matrix(marginal,NPATHS,seed)
            lr=(g .+ shift .+ marginal.rf).*marginal.dt
            S=vcat(fill(S0,1,NPATHS),S0.*exp.(cumsum(lr;dims=1)))
            @assert size(S)==(23,NPATHS)
            @assert S[:,1] ≈ JumpHMM.prices_from_growth_rates(g[:,1].+shift,S0;
                                                rf=marginal.rf,dt=marginal.dt)
            zs=(lr.-mu)./scale
            zi=randn(MersenneTwister(seed+1),22,NPATHS)
            targets=make_targets(model,S,dtes,strikes)
            marks=Dict{Symbol,Array{Float64,3}}()
            variances=Dict{Symbol,Array{Float64,3}}()
            for mode in MODES
                v=variance_paths(targets,zs,zi;mode)
                variances[mode]=v
                marks[mode]=price_dates(S,v,dtes,strikes,kinds,STEPS.+1)
                @assert all(isfinite,marks[mode])
                @assert marks[mode][1,:,:] == marks[:frozen][1,:,:]
                @assert marks[mode][end,:,:] == marks[:frozen][end,:,:]
                for c in 1:2
                    coords=strikes[c]./S[1:end-1,:]
                    push!(diagnostics,(ticker,seed,mode=string(mode),kind=string(kinds[c]),
                          floor_fraction=mean(v[2:end,:,c].<=0.005^2),
                          outside_moneyness_fraction=mean((coords.<0.8).|(coords.>1.2))))
                end
                for p in 1:min(20,NPATHS),step in (10,20),c in 1:2
                    j=findfirst(==(step),STEPS); t=step+1
                    refined=option_mark(S[t,p],strikes[c],v[t,p,c],dtes[t],kinds[c];depth=401)
                    push!(numerical,(ticker,seed,mode=string(mode),kind=string(kinds[c]),step,path=p,
                          price201=marks[mode][j,p,c],price401=refined,
                          absolute_change=abs(refined-marks[mode][j,p,c])))
                end
            end
            for mode in MODES,c in 1:2,(j,step) in enumerate(STEPS),p in 1:NPATHS
                push!(rows,(ticker,seed,path=p,mode=string(mode),kind=string(kinds[c]),step,
                      date=string(dates[step+1]),dte=dtes[step+1],spot=S[step+1,p],
                      premium=premiums[c],mark=marks[mode][j,p,c],
                      surface_mark=marks[:surface][j,p,c],frozen_mark=marks[:frozen][j,p,c]))
            end
            JLD2.jldsave(joinpath(OUT,"$(lowercase(ticker))_$(seed).jld2");S,dates,dtes,
                        zs,zi,targets,variances,marks,strikes,premiums)
            if seed==first(SEEDS)
                grid=collect(range(0.8*S0,1.2*S0;length=11))
                np=min(50,NPATHS)
                gridtargets=make_targets(model,S[:,1:np],dtes,grid)
                for mode in MODES
                    v=variance_paths(gridtargets,zs[:,1:np],zi[:,1:np];mode)
                    for p in 1:np,step in (0,10,20),kind in kinds,depth in (201,401)
                        # Entry is identical across paths; count it once per arm and kind.
                        step==0 && p>1 && continue
                        t=step+1
                        prices=[option_mark(S[t,p],grid[c],v[t,p,c],dtes[t],kind;depth)
                                for c in eachindex(grid)]
                        for check in strike_checks(prices,grid,S[t,p],kind)
                            push!(audits,merge((ticker,seed,mode=string(mode),kind=string(kind),
                                  step,path=p,depth),check))
                        end
                    end
                end
            end
            CSV.write(joinpath(OUT,"path_marks.csv"),DataFrame(rows))
            CSV.write(joinpath(OUT,"numerical_check.csv"),DataFrame(numerical))
            CSV.write(joinpath(OUT,"strike_audit.csv"),DataFrame(audits))
            CSV.write(joinpath(OUT,"factor_diagnostics.csv"),DataFrame(diagnostics))
        end
    end
end

function render_results()
    df=CSV.read(joinpath(OUT,"path_marks.csv"),DataFrame)
    summaries=NamedTuple[]
    for group in groupby(df,[:ticker,:kind,:mode,:step])
        for seed in [0;SEEDS]
            g=seed==0 ? group : group[group.seed.==seed,:]
            isempty(g) && continue
            stats=paired_summary(g.mark,g.surface_mark,first(g.premium))
            push!(summaries,merge((ticker=first(g.ticker),kind=first(g.kind),mode=first(g.mode),
                  step=first(g.step),date=first(g.date),seed,n=nrow(g)),stats))
        end
    end
    summary=DataFrame(summaries)
    CSV.write(joinpath(OUT,"summary.csv"),summary)
    contrasts=NamedTuple[]
    for group in groupby(df,[:ticker,:kind,:step])
        for (reference,variant) in ((:frozen,:surface),(:surface,:relaxation),
                                    (:relaxation,:uncoupled),(:uncoupled,:coupled))
            a=sort(group[group.mode.==string(reference),:],[:seed,:path])
            b=sort(group[group.mode.==string(variant),:],[:seed,:path])
            @assert a.seed==b.seed && a.path==b.path && a.spot==b.spot
            for seed in [0;SEEDS]
                keep=seed==0 ? trues(nrow(a)) : a.seed.==seed
                stats=paired_summary(b.mark[keep],a.mark[keep],first(b.premium))
                push!(contrasts,merge((ticker=first(b.ticker),kind=first(b.kind),step=first(b.step),
                      reference=string(reference),variant=string(variant),seed),stats))
            end
        end
    end
    CSV.write(joinpath(OUT,"contrast_summary.csv"),DataFrame(contrasts))
    panels=Plots.Plot[]
    for ticker in ("GS","LLY"),kind in ("put","call")
        p=plot(title="$ticker $kind",xlabel="Trading transitions",ylabel="Mean absolute mark change (\$/share)",
               ylims=(0,3),legend=(ticker=="GS" && kind=="put") ? :topright : false)
        for mode in (:frozen,:relaxation,:uncoupled,:coupled)
            g=sort(summary[(summary.ticker.==ticker).&(summary.kind.==kind).&
                   (summary.mode.==string(mode)).&(summary.seed.==0),:],:step)
            plot!(p,g.step,g.mean_abs_mark_change;label=LABELS[mode],color=COLORS[mode],lw=2,marker=:circle,ms=3)
        end
        push!(panels,p)
    end
    fig=plot(panels...;layout=(2,2),size=(1100,750),margin=5Plots.mm,
             titlefontsize=12,guidefontsize=10,tickfontsize=9,legendfontsize=8)
    savefig(fig,joinpath(OUT,"dynamic_ablation.pdf"))
    savefig(fig,joinpath(OUT,"dynamic_ablation.png"))
    # Every table value is generated from the pooled path-level output.
    open(joinpath(OUT,"dynamic_ablation_table.tex"),"w") do io
        println(io,raw"\begin{tabular}{llrrrr}")
        println(io,raw"\toprule")
        println(io,raw"Contract & IV variant & Mean P\&L & ES$_{5\%}$ & $\Delta$P\&L (SE) & Mean $|\Delta P|$ ",repeat("\\",2))
        println(io,raw"\midrule")
        for ticker in ("GS","LLY"),kind in ("put","call")
            for mode in MODES
                g=only(eachrow(summary[(summary.ticker.==ticker).&(summary.kind.==kind).&
                       (summary.mode.==string(mode)).&(summary.step.==10).&(summary.seed.==0),:]))
                label=mode==:uncoupled ? raw"Noise, $\rho=0$" : LABELS[mode]
                @printf(io,"%s %s & %s & %.2f & %.2f & %+.2f (%.2f) & %.2f \\\\\n",
                        ticker,kind,label,g.mean_pnl,g.es05_pnl,g.mean_pnl_change,g.paired_se,g.mean_abs_mark_change)
            end
            println(io,raw"\midrule")
        end
        println(io,raw"\end{tabular}")
    end
    audit=CSV.read(joinpath(OUT,"strike_audit.csv"),DataFrame)
    a=combine(groupby(audit,[:ticker,:mode,:depth,:check]),:violations=>sum=>:violations,
              :tests=>sum=>:tests,:max_violation=>maximum=>:max_violation)
    a.fraction=a.violations./a.tests
    CSV.write(joinpath(OUT,"strike_audit_summary.csv"),a)
    numerical=CSV.read(joinpath(OUT,"numerical_check.csv"),DataFrame)
    CSV.write(joinpath(OUT,"numerical_summary.csv"),combine(groupby(numerical,[:ticker,:mode]),
        :absolute_change=>median=>:median_abs_change,:absolute_change=>maximum=>:max_abs_change))
    println("Completed: $OUT"); flush(stdout)
    show(stdout,MIME("text/plain"),summary[(summary.step.==10).&(summary.seed.==0),:]); println()
end

if !("--render-only" in ARGS)
    run_experiment()
end
render_results()
