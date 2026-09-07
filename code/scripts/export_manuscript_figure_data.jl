"""
Export the frozen manuscript inputs for figure rendering without refitting models
or resimulating paths. Run from the repository root with --project=code.
"""
ENV["GKSwstype"] = "100"
using CSV, DataFrames, Dates, JLD2, Statistics, SHA, TOML, Flux
include(joinpath(@__DIR__, "..", "src", "ScenarioTemplate.jl"))
const ST = ScenarioTemplate
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "code", "results", "figure_revision")
mkpath(OUT)
filehash(path) = bytes2hex(open(sha256, path))

# Verify the fitted surface and every corpus file against the ablation manifest.
manifest_path = joinpath(ROOT, "code/results/dynamic_ablation/corpus_manifest.csv")
config = TOML.parsefile(joinpath(ROOT, "code/results/dynamic_ablation/config.toml"))
nnpath = joinpath(ROOT, "code/figures/calibrate_ladders_per_ticker_nn_cache.jld2")
@assert filehash(nnpath) == config["nn_sha256"]
@assert filehash(manifest_path) == config["corpus_manifest_sha256"]
manifest = CSV.read(manifest_path, DataFrame)
for row in eachrow(manifest)
    @assert filehash(joinpath(ROOT, row.path)) == row.sha256
end
corpus = ST._load_all_ladders(joinpath(ROOT, "code/data/ladder"))
@assert nrow(corpus) == 234549
standardizer = ST._build_standardizer(corpus)
nn = JLD2.load(nnpath)

# Preserve all observations in the original six single-date, single-DTE panels.
sectors = Dict("SPY"=>"ETF", "NVDA"=>"Tech", "MSFT"=>"Tech",
               "LLY"=>"Healthcare", "GS"=>"Financials", "AVGO"=>"Tech")
points = NamedTuple[]
curves = NamedTuple[]
daily = NamedTuple[]
polytheta = Dict(string(first(p))=>last(p) for p in nn["poly_theta_pairs"])
beta = nn["poly_beta"]
for ticker in ["SPY", "NVDA", "MSFT", "LLY", "GS", "AVGO"]
    td = corpus[corpus.ticker .== ticker, :]
    pt, lp, _ = ST._restore_nn(nn, ticker; use_per_ticker=true, sector=sectors[ticker])
    sec, ls, _ = ST._restore_nn(nn, ticker; use_per_ticker=false, sector=sectors[ticker])
    predict(net, level, dte, m) = 100sqrt(exp(level)*ST._psi(net, standardizer, Float64(m), 1.0, dte))
    td.pt_iv = [predict(pt, lp, r.actual_dte, r.moneyness) for r in eachrow(td)]
    td.sector_iv = [predict(sec, ls, r.actual_dte, r.moneyness) for r in eachrow(td)]
    latest = maximum(td.und_session_date)
    day = td[td.und_session_date .== latest, :]
    dtes = sort(unique(day.actual_dte))
    chosen_dte = dtes[max(1, length(dtes) ÷ 2)]
    selected = day[day.actual_dte .== chosen_dte, :]
    for r in eachrow(selected)
        push!(points, (ticker, date=string(latest), dte=chosen_dte,
                      kind=string(r.type), moneyness=Float64(r.moneyness),
                      observed_iv=100Float64(r.implied_vol), per_ticker_iv=r.pt_iv,
                      sector_iv=r.sector_iv))
    end
    for m in range(0.8, 1.2; length=241)
        x, y = log(Float64(chosen_dte)), log(m)
        poly = 100sqrt(polytheta[ticker]*exp(beta[1]*x+beta[2]*y+beta[3]*x*y+beta[4]*y^2+beta[5]*x^2))
        push!(curves, (ticker, date=string(latest), dte=chosen_dte, moneyness=m,
                       per_ticker_iv=predict(pt,lp,chosen_dte,m),
                       sector_iv=predict(sec,ls,chosen_dte,m), parametric_iv=poly))
    end
    for g in groupby(td, :und_session_date)
        for scope in ("all_contracts", "shown_maturity", "near_atm_shown_maturity")
            selected_rows = scope=="all_contracts" ? g : g[g.actual_dte .== chosen_dte, :]
            if scope=="near_atm_shown_maturity"
                selected_rows = selected_rows[abs.(selected_rows.moneyness .- 1) .<= 0.02, :]
            end
            isempty(selected_rows) && continue
            obs = 100Float64.(selected_rows.implied_vol)
            residual = selected_rows.pt_iv .- obs
            push!(daily, (ticker, date=string(first(g.und_session_date)), scope,
                displayed_dte=chosen_dte, n=nrow(selected_rows), mean_observed_iv=mean(obs),
                mean_predicted_iv=mean(selected_rows.pt_iv), mean_bias=mean(residual),
                rmse=sqrt(mean(abs2,residual)), median_bias=median(residual)))
        end
    end
end
CSV.write(joinpath(OUT,"smile_points.csv"), DataFrame(points))
CSV.write(joinpath(OUT,"smile_curves.csv"), DataFrame(curves))
CSV.write(joinpath(OUT,"smile_date_diagnostics.csv"), DataFrame(daily))

# Use the corrected saved illustrations, not the separate ablation simulations.
pathrows = NamedTuple[]
terminal = NamedTuple[]
summary = CSV.read(joinpath(ROOT,"code/results/fitted_scenarios/summary.csv"),DataFrame)
input_hashes = Dict("nn_sha256"=>filehash(nnpath), "corpus_manifest_sha256"=>filehash(manifest_path))
for ticker in ("GS", "LLY")
    cachepath=joinpath(ROOT,"code/results/fitted_scenarios/$(lowercase(ticker))_cache.jld2")
    input_hashes["$(lowercase(ticker))_scenario_sha256"] = filehash(cachepath)
    c=JLD2.load(cachepath)
    @assert c["scenario_version"]==2
    @assert size(c["S_paths"])==(23,1000)
    row=only(eachrow(summary[summary.ticker .== ticker,:]))
    dates=Date.(c["trading_dates"])
    for j in axes(c["S_paths"],2)
        for t in axes(c["S_paths"],1)
            push!(pathrows,(ticker,path=j,step=t-1,date=string(dates[t]),
                day=Dates.value(dates[t]-dates[1]),dte=c["calendar_dtes"][t],
                spot=c["S_paths"][t,j],put_value=c["V_put"][t,j],call_value=c["V_call"][t,j],
                put_iv=100sqrt(c["v_put_paths"][t,j]),call_iv=100sqrt(c["v_call_paths"][t,j])))
        end
        for kind in ("put","call")
            strike=c["K_"*kind]; spot=c["S_paths"][end,j]
            payoff=kind=="put" ? max(strike-spot,0) : max(spot-strike,0)
            @assert isapprox(payoff,c["V_"*kind][end,j];atol=1e-12)
            premium=row[Symbol("market_mid_"*kind)]
            push!(terminal,(ticker,kind,path=j,spot,strike,premium,payoff,pnl=premium-payoff))
        end
    end
end
CSV.write(joinpath(OUT,"scenario_paths.csv"),DataFrame(pathrows))
CSV.write(joinpath(OUT,"terminal_pnl.csv"),DataFrame(terminal))
open(joinpath(OUT,"inputs.toml"),"w") do io
    TOML.print(io,input_hashes)
end
println("Exported frozen figure inputs to $OUT")
show(stdout,MIME("text/plain"),filter(r->r.scope=="near_atm_shown_maturity" && r.date=="2026-05-11",DataFrame(daily)))
println()
