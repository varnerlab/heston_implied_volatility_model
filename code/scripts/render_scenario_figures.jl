"""
Render the saved scenario summaries using the HMM-w-jumps-paper figure style.

The visual reference is code/copula-experiment/Fig-Price-Trajectories.jl and
code/spy-experiment/Fig1-Empirical-Motivation-SPY.jl in that repository. This
renderer uses the same Plots/GR backend, boxed panels, sans-serif lettering,
inset legends, and navy/red palette. Gridlines are disabled at the author's
request. No models are fitted and no trajectories are resimulated.
"""
ENV["GKSwstype"] = "100"
using CSV, DataFrames, Dates, Statistics, Plots
gr()

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const FIGURES = joinpath(ROOT, "paper-arxiv", "sections", "figures")
const NAVY = "#1d3557"
const RED = "#e63946"
const STYLE = (
    bg="gray95", background_color_outside=:white, framestyle=:box,
    fontfamily="sans-serif", grid=false, minorgrid=false,
    fg_legend=:transparent, background_color_legend=:transparent,
    xguidefontsize=18, yguidefontsize=18, titlefontsize=17,
    tickfontsize=13, legendfontsize=13,
    bottom_margin=14Plots.mm, left_margin=12Plots.mm,
    top_margin=4Plots.mm, right_margin=4Plots.mm,
)

function summary_panel(days, values, low, high; title, ylabel, legend=false, ylimits=nothing)
    q05 = [quantile(row, .05) for row in eachrow(values)]
    med = [median(row) for row in eachrow(values)]
    q95 = [quantile(row, .95) for row in eachrow(values)]
    low_med = vec(median(values[:, low]; dims=2))
    high_med = vec(median(values[:, high]; dims=2))
    plotted = vcat(q05, q95, low_med, high_med)
    p = plot(days, med; STYLE..., title, ylabel,
        xlabel="Time (calendar days)", legend=legend ? :topleft : false,
        xlims=(first(days), last(days)), xticks=[0, 10, 20, last(days)],
        ribbon=(med .- q05, q95 .- med), fillcolor=:gray, fillalpha=.22,
        c=:black, lw=2.8, label="Median (5–95%)")
    plot!(p, days, low_med; c=RED, lw=2.5, label="Low terminal stock")
    plot!(p, days, high_med; c=NAVY, lw=2.5, label="High terminal stock")
    if ylimits !== nothing
        upper = isinf(ylimits[2]) ? 10ceil(1.02maximum(plotted)/10) : ylimits[2]
        ylims!(p, (ylimits[1], upper))
    end
    return p, plotted
end

function save_panel_figure(fig, ticker, suffix)
    stem = joinpath(FIGURES, lowercase(ticker), lowercase(ticker)*"_"*suffix)
    savefig(fig, stem*".pdf")
    savefig(fig, stem*".png")
end

paths = CSV.read(joinpath(ROOT, "code/results/figure_revision/scenario_paths.csv"), DataFrame)
contracts = CSV.read(joinpath(ROOT, "code/results/fitted_scenarios/summary.csv"), DataFrame)
for ticker in ("GS", "LLY")
    rows = sort(paths[paths.ticker .== ticker, :], [:path, :step])
    @assert nrow(rows) == 23000
    nsteps, npaths = 23, 1000
    matrices = Dict(key => reshape(rows[!, key], nsteps, npaths)
                    for key in (:spot, :put_value, :call_value, :put_iv, :call_iv))
    @assert rows.step[1:nsteps] == collect(0:22)
    dates = Date.(rows.date[1:nsteps])
    days = Dates.value.(dates .- first(dates))
    order = sortperm(matrices[:spot][end, :])
    low, high = order[11:50], order[951:990]
    contract = only(eachrow(contracts[contracts.ticker .== ticker, :]))
    a, stock_values = summary_panel(days, matrices[:spot], low, high;
        title="(a) $(ticker) Stock", ylabel="Stock price (\$/share)", legend=true)
    stock_lo, stock_hi = extrema(stock_values)
    # Leave an empty upper-left area for the inset legend.
    ylims!(a, (50floor(stock_lo/50), 50ceil((stock_hi + .25(stock_hi-stock_lo))/50)))
    b, _ = summary_panel(days, matrices[:put_value], low, high;
        title="(b) Put (K = $(Int(contract.K_put)))", ylabel="Option value (\$/share)",
        ylimits=(0, Inf))
    c, _ = summary_panel(days, matrices[:call_value], low, high;
        title="(c) Call (K = $(Int(contract.K_call)))", ylabel="Option value (\$/share)",
        ylimits=(0, Inf))
    # Compensate for the reduction of three panels to the manuscript text width.
    for p in (a, b, c)
        plot!(p; xguidefontsize=23, yguidefontsize=23, titlefontsize=21,
              tickfontsize=17, legendfontsize=17)
    end
    save_panel_figure(plot(a, b, c; layout=(1, 3), size=(1500, 600), dpi=300), ticker, "short_paths")

    a, a_values = summary_panel(days[1:end-1], matrices[:put_iv][1:end-1, :], low, high;
        title="(a) Put IV (K = $(Int(contract.K_put)))", ylabel="Implied volatility (%)", legend=true)
    b, b_values = summary_panel(days[1:end-1], matrices[:call_iv][1:end-1, :], low, high;
        title="(b) Call IV (K = $(Int(contract.K_call)))", ylabel="Implied volatility (%)")
    values = vcat(a_values, b_values)
    limits = (5floor(minimum(values)/5), 5ceil(maximum(values)/5))
    ylims!(a, limits); ylims!(b, limits)
    save_panel_figure(plot(a, b; layout=(1, 2), size=(1100, 500), dpi=300), ticker, "iv_trajectories")
end
println("Rendered GS and LLY scenario figures using the HMM paper style.")
