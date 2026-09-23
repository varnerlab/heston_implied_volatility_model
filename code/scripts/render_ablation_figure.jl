"""
Show the paired effect of return coupling conditional on the simulated stock move.
Frozen path marks supply both variants; no model is refitted or resimulated.
Ten equal-count stock-return bins retain all 3,000 paths for each contract.
"""
ENV["GKSwstype"] = "100"
using CSV, DataFrames, Statistics, Plots
gr()

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const RESULTS = joinpath(ROOT, "code/results/dynamic_ablation")
const STYLE = (
    bg="gray95", background_color_outside=:white, framestyle=:box,
    fontfamily="sans-serif", grid=false, minorgrid=false,
    fg_legend=:transparent, background_color_legend=:transparent,
    xguidefontsize=18, yguidefontsize=18, titlefontsize=17,
    tickfontsize=13, legendfontsize=13,
    bottom_margin=14Plots.mm, left_margin=12Plots.mm,
    top_margin=4Plots.mm, right_margin=4Plots.mm,
)

raw = CSV.read(joinpath(RESULTS, "path_marks.csv"), DataFrame)
contrasts = CSV.read(joinpath(RESULTS, "contrast_summary.csv"), DataFrame)
contrasts = contrasts[(contrasts.seed .== 0) .& (contrasts.step .== 10) .&
    (contrasts.reference .== "uncoupled") .& (contrasts.variant .== "coupled"), :]
binned = DataFrame(ticker=String[], kind=String[], bin=Int[], n=Int[],
    return_min=Float64[], return_max=Float64[], return_median=Float64[],
    pnl_q25=Float64[], pnl_median=Float64[], pnl_q75=Float64[])
panels = Plots.Plot[]
for (ticker, letter) in [("GS", "a"), ("LLY", "b")]
    entry_spot = only(unique(raw.spot[(raw.ticker .== ticker) .& (raw.step .== 0)]))
    panel = plot(; STYLE..., title="($letter)", xlabel="Stock return (%)",
        ylabel="Δ short-option P&L (\$/share)", legend=:topleft,
        xlims=(-12, 12), xticks=-10:5:10, ylims=(-4, 4), yticks=-4:2:4)
    hline!(panel, [0.0]; c=:gray, ls=:dash, lw=1, label="")
    for (kind, label, color, marker) in [
            ("put", "Short put", "#e63946", :circle),
            ("call", "Short call", "#1d3557", :diamond)]
        rows = raw[(raw.ticker .== ticker) .& (raw.kind .== kind) .& (raw.step .== 10), :]
        u = sort(rows[rows.mode .== "uncoupled", :], [:seed, :path])
        c = sort(rows[rows.mode .== "coupled", :], [:seed, :path])
        @assert nrow(u) == nrow(c) == 3000
        @assert u[:, [:seed, :path, :spot, :premium]] == c[:, [:seed, :path, :spot, :premium]]
        # Premiums cancel: coupled short P&L minus uncoupled short P&L.
        delta = u.mark .- c.mark
        reference = only(eachrow(contrasts[(contrasts.ticker .== ticker) .& (contrasts.kind .== kind), :]))
        @assert isapprox(mean(delta), reference.mean_pnl_change; atol=1e-10)
        @assert isapprox(mean(abs.(delta)), reference.mean_abs_mark_change; atol=1e-10)
        @assert isapprox(std(delta)/sqrt(length(delta)), reference.paired_se; atol=1e-10)
        returns = 100 .* (u.spot ./ entry_spot .- 1)
        order = sortperm(returns)
        for bin in 1:10
            indices = order[(300(bin-1)+1):300bin]
            moves, differences = returns[indices], delta[indices]
            push!(binned, (ticker, kind, bin, length(indices), minimum(moves),
                maximum(moves), median(moves), quantile(differences, .25),
                median(differences), quantile(differences, .75)))
        end
        q = binned[(binned.ticker .== ticker) .& (binned.kind .== kind), :]
        @assert sum(q.n) == 3000
        @assert all(-12 .< q.return_median .< 12)
        @assert all(-4 .< q.pnl_q25) && all(q.pnl_q75 .< 4)
        plot!(panel, q.return_median, q.pnl_median; c=color, lw=2.5,
            marker, ms=4, markerstrokewidth=0, label,
            ribbon=(q.pnl_median .- q.pnl_q25, q.pnl_q75 .- q.pnl_median),
            fillcolor=color, fillalpha=.15)
    end
    push!(panels, panel)
end
CSV.write(joinpath(ROOT, "code/results/figure_revision/ablation_coupling_by_return.csv"), binned)
figure = plot(panels...; layout=(1,2), size=(1100,520), dpi=300)
stem = joinpath(ROOT, "paper-arxiv/sections/figures/dynamic_ablation")
savefig(figure, stem*".pdf")
savefig(figure, stem*".png")
println("Rendered coupling effects by stock return; all four paired contrasts match saved summaries.")
