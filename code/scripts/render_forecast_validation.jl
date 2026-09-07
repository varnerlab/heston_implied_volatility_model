# Plot the protocol-selected observed trajectory against forecasts issued at its origin.
ENV["GKSwstype"]="100"
using CSV,DataFrames,Plots
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const OUT=joinpath(ROOT,"code/results/chronological_validation")
const STYLE=(bg="gray95",background_color_outside=:white,framestyle=:box,
    fontfamily="sans-serif",grid=false,minorgrid=false,
    fg_legend=:transparent,background_color_legend=:transparent,
    xguidefontsize=23,yguidefontsize=23,titlefontsize=21,tickfontsize=17,legendfontsize=17,
    bottom_margin=14Plots.mm,left_margin=12Plots.mm,top_margin=4Plots.mm,right_margin=4Plots.mm)
rows=CSV.read(joinpath(OUT,"short_maturity","forecast_example.csv"),DataFrame)
panels=Plots.Plot[]
for (series,letter,ylabel) in [("stock","a","Stock price (\$/share)"),
    ("put","b","Put value (\$/share)"),("call","c","Call value (\$/share)")]
    q=sort(rows[(rows.series .== series).&(rows.mode .== "coupled"),:],:step)
    p=plot(q.step,q.predicted_mean;STYLE...,title="($letter)",
        xlabel="Time (trading sessions)",ylabel,xlims=(0,5),xticks=0:5,
        c="#1d3557",lw=3,label=series=="stock" ? "Forecast" : "Coupled IV",
        ribbon=(q.predicted_mean.-q.lo,q.hi.-q.predicted_mean),fillcolor="#1d3557",fillalpha=.15,
        legend=:topleft)
    if series!="stock"
        frozen=sort(rows[(rows.series .== series).&(rows.mode .== "frozen"),:],:step)
        plot!(p,frozen.step,frozen.predicted_mean;c="#e63946",lw=2.5,ls=:dash,label="Frozen IV")
    end
    keep=isfinite.(q.observed)
    if series=="stock"
        plot!(p,q.step[keep],q.observed[keep];c=:black,lw=2,marker=:circle,ms=4,label="Observed")
    else
        plot!(p,q.step[keep],q.observed[keep];c=:black,lw=2,marker=:circle,ms=4,
            yerror=(q.observed[keep].-q.bid[keep],q.ask[keep].-q.observed[keep]),label="Observed")
    end
    lo,hi=extrema(vcat(q.lo,q.hi,q.observed[keep]))
    plot!(p;ylims=(series=="stock" ? lo-.05(hi-lo) : max(0,lo-.05(hi-lo)),hi+.25(hi-lo)))
    push!(panels,p)
end
figure=plot(panels...;layout=(1,3),size=(1500,600),dpi=300)
stem=joinpath(ROOT,"paper-arxiv/sections/figures/chronological_forecast")
savefig(figure,stem*".pdf");savefig(figure,stem*".png")
println("Rendered the prespecified earliest eligible GS forecast.")
