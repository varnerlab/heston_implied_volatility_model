# Export dated close observations without fitting or scoring candidate forecasts.
using JLD2,CSV,DataFrames,Dates,SHA,TOML
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const OUT=joinpath(ROOT,"code/results/small_stock_comparison")
mkpath(OUT)
rows=NamedTuple[];sources=Dict{String,String}()
for (period,name) in [("training","SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2"),
                       ("test2025","SP500-Daily-OHLC-1-2-2025-to-12-31-2025.jld2")]
    path=joinpath(ROOT,"code/data/equity",name)
    sources[relpath(path,ROOT)]=open(sha256,path)|>bytes2hex
    dataset=JLD2.load(path,"dataset")
    reference=Date.(dataset["SPY"].timestamp)
    @assert issorted(reference) && allunique(reference)
    for ticker in ["GS","LLY","SPY"]
        data=dataset[ticker]
        @assert Date.(data.timestamp)==reference
        for r in eachrow(data)
            @assert isfinite(r.close) && r.close>0
            push!(rows,(period,ticker,session=Date(r.timestamp),spot=Float64(r.close)))
        end
    end
end
CSV.write(joinpath(OUT,"historical_closes.csv"),DataFrame(rows))
manifest_path=joinpath(ROOT,"code/results/chronological_validation/source_manifest.csv")
manifest=CSV.read(manifest_path,DataFrame)
sources[relpath(manifest_path,ROOT)]=open(sha256,manifest_path)|>bytes2hex
rows2026=NamedTuple[]
for r in eachrow(manifest)
    r.selected && r.ticker in ["GS","LLY","SPY"] || continue
    path=joinpath(ROOT,r.path);data=CSV.read(path,DataFrame;limit=1)
    sources[r.path]=open(sha256,path)|>bytes2hex
    push!(rows2026,(ticker=r.ticker,session=Date(r.session),spot=Float64(data.und_close[1])))
end
CSV.write(joinpath(OUT,"closes_2026.csv"),sort(DataFrame(rows2026),[:ticker,:session]))
open(joinpath(OUT,"data_manifest.toml"),"w") do io
    TOML.print(io,Dict("source_sha256"=>sources))
end
println("Exported ",length(rows)," historical close rows and ",length(rows2026)," 2026 rows.")
