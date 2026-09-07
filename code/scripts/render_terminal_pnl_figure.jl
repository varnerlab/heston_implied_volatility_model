"""
Render the optional terminal-P&L diagnostic from the frozen GS/LLY simulations.
This diagnostic is not included in the manuscript; the scenario tables report
the terminal outcomes.
The quantile curves retain every simulated outcome. Interpretation and numerical
summaries belong in the caption and tables, rather than callouts on the curves.
"""
ENV["GKSwstype"] = "100"
using CSV, DataFrames, Statistics, Plots
gr()

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const STYLE = (
    bg="gray95", background_color_outside=:white, framestyle=:box,
    fontfamily="sans-serif", grid=false, minorgrid=false, legend=false,
    xguidefontsize=18, yguidefontsize=18, titlefontsize=17, tickfontsize=13,
    bottom_margin=12Plots.mm, left_margin=12Plots.mm,
    top_margin=4Plots.mm, right_margin=4Plots.mm,
)
const CONTRACTS = [("GS", "put"), ("GS", "call"), ("LLY", "put"), ("LLY", "call")]
data = CSV.read(joinpath(ROOT, "code/results/figure_revision/terminal_pnl.csv"), DataFrame)
summary = CSV.read(joinpath(ROOT, "code/results/fitted_scenarios/summary.csv"), DataFrame)
panels = Plots.Plot[]
for (letter, (ticker, kind)) in zip('a':'d', CONTRACTS)
    rows = data[(data.ticker .== ticker) .& (data.kind .== kind), :]
    @assert nrow(rows) == 1000
    @assert all(isapprox.(rows.pnl, rows.premium .- rows.payoff; atol=1e-10))
    values = sort(rows.pnl)
    q05 = quantile(values, .05)
    es05 = mean(values[values .<= q05])
    retained = 100mean(rows.payoff .== 0)
    saved = only(eachrow(summary[summary.ticker .== ticker, :]))
    for (field, value) in (("mean_pnl_", mean(values)), ("q05_pnl_", q05),
                          ("es05_pnl_", es05), ("pct_premium_kept_", retained))
        @assert isapprox(value, saved[Symbol(field*kind)]; atol=1e-10)
    end
    color = kind == "put" ? "#1d3557" : "#e63946"
    p = plot(range(0,100; length=length(values)), values; STYLE...,
        title="($(letter)) $(ticker) $(uppercasefirst(kind))",
        xlabel="Outcome percentile (%)", ylabel="Terminal P&L (\$/share)",
        xlims=(0,100), xticks=0:25:100, ylims=(-320,40), yticks=-300:100:0,
        c=color, lw=2.5, label="")
    hline!(p, [0.0]; c=:black, ls=:dash, lw=1.2, label="")
    scatter!(p, [5.0], [q05]; c=color, ms=5.0, markerstrokewidth=0, label="")
    push!(panels, p)
end
figure = plot(panels...; layout=(2,2), size=(1100,900), dpi=300)
stem = joinpath(ROOT, "code/results/figure_revision/terminal_pnl_comparison")
savefig(figure, stem*".pdf")
savefig(figure, stem*".png")
println("Rendered terminal P&L; all four distributions match the frozen scenario summaries.")
