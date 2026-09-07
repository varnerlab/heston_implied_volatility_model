# Reconstruct fixed saved forecast distributions and inspect Monte Carlo precision.
using CSV,DataFrames,JLD2,Dates,Statistics,Test
include(joinpath(@__DIR__,"..","src","DynamicAblation.jl"))
include(joinpath(@__DIR__,"..","src","ForecastValidation.jl"))
using .DynamicAblation,.ForecastValidation
const OUT=normpath(joinpath(@__DIR__,"..","results","chronological_validation"))
checks=NamedTuple[]
@testset "Saved forecast reconstruction" begin
    for (cohort,dir) in [("monthly",OUT),("short",joinpath(OUT,"short_maturity"))]
        scores=CSV.read(joinpath(dir,"forecast_scores.csv"),DataFrame)
        for ticker in ["GS","LLY"]
            cache=JLD2.load(joinpath(dir,"paths_$(lowercase(ticker))_2026-08-04.jld2"))
            S=cache["stock"];v=cache["variances"];cs=cache["contracts"];dates=cache["dates"]
            for mode in MODES
                @test v[mode][1,1,:]≈cs.origin_iv.^2
                @test all(isfinite,v[mode])
            end
            for h in [1,5],(c,r) in enumerate(eachrow(cs))
                rows=scores[(scores.ticker.==ticker).&(Date.(scores.origin).==Date("2026-08-04")).&
                    (scores.horizon.==h).&(scores.symbol.==r.symbol).&(scores.conditioning.=="joint"),:]
                isempty(rows) && continue
                marks=Dict{Symbol,Vector{Float64}}()
                for mode in [:frozen,:coupled]
                    values=[option_mark(S[h+1,p],r.strike,v[mode][h+1,p,c],
                        Dates.value(Date(r.expiry)-dates[h+1]),Symbol(r.kind)) for p in axes(S,2)]
                    marks[mode]=values
                    stored=only(eachrow(rows[rows.mode.==string(mode),:]))
                    reconstructed=distribution_scores(values,stored.observed)
                    @test reconstructed.predicted_mean≈stored.predicted_mean atol=1e-10
                    @test reconstructed.crps≈stored.crps atol=1e-10
                    @test reconstructed.predicted_q05≈stored.predicted_q05 atol=1e-10
                    @test reconstructed.predicted_q95≈stored.predicted_q95 atol=1e-10
                end
                difference=marks[:coupled].-marks[:frozen]
                push!(checks,(cohort,ticker,horizon=h,kind=r.kind,mean_price_difference=mean(difference),
                    paired_mc_se=std(difference)/sqrt(length(difference)),
                    first_half_difference=mean(difference[1:500]),second_half_difference=mean(difference[501:1000])))
            end
        end
    end
end
CSV.write(joinpath(OUT,"monte_carlo_checks.csv"),DataFrame(checks))
