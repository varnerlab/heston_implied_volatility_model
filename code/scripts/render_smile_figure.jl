"""
Render the six calibration smiles from frozen observations and fitted curves.
Only panel letters, axes, and the series legend appear in the graphic. Dates,
tickers, maturities, and interpretation belong in the manuscript caption.
"""
ENV["GKSwstype"] = "100"
using CSV, DataFrames, Plots
gr()

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DATA = joinpath(ROOT, "code/results/figure_revision")
const STYLE = (
    bg="gray95", background_color_outside=:white, framestyle=:box,
    fontfamily="sans-serif", grid=false, minorgrid=false,
    fg_legend=:transparent, background_color_legend=:transparent,
    xguidefontsize=18, yguidefontsize=18, titlefontsize=17,
    tickfontsize=14, legendfontsize=13,
    bottom_margin=7Plots.mm, left_margin=10Plots.mm,
    top_margin=3Plots.mm, right_margin=4Plots.mm,
)

points = CSV.read(joinpath(DATA, "smile_points.csv"), DataFrame)
curves = CSV.read(joinpath(DATA, "smile_curves.csv"), DataFrame)
panels = Plots.Plot[]
for (i, (ticker, dte)) in enumerate(zip(
        ["SPY", "NVDA", "MSFT", "LLY", "GS", "AVGO"], [7, 7, 9, 11, 11, 9]))
    p = points[points.ticker .== ticker, :]
    c = curves[curves.ticker .== ticker, :]
    @assert all(string.(p.date) .== "2026-05-11") && all(p.dte .== dte)
    @assert all(string.(c.date) .== "2026-05-11") && all(c.dte .== dte)
    # Preserve the original fitted-curve interval and every observed contract.
    c = c[(c.moneyness .>= .85) .& (c.moneyness .<= 1.15), :]
    panel = plot(; STYLE..., title="($(Char('a' + i - 1)))",
        ylabel="Implied volatility (%)", xlabel=i >= 5 ? "Strike / spot, K/S" : "",
        xlims=(.79, 1.21), xticks=[.8, .9, 1., 1.1, 1.2],
        legend=i == 1 ? :top : false)
    for (kind, color, marker, label) in [
            ("call", "#1d3557", :circle, "Observed calls"),
            ("put", "#e63946", :diamond, "Observed puts")]
        q = p[p.kind .== kind, :]
        scatter!(panel, q.moneyness, q.observed_iv; c=color, marker,
            ms=3, markerstrokewidth=0, alpha=.75, label)
    end
    for (key, color, ls, label) in [
            (:per_ticker_iv, :black, :solid, "Per-ticker fit"),
            (:sector_iv, "#457b9d", :dash, "Sector fit"),
            (:parametric_iv, "#b28c32", :dot, "Global parametric fit")]
        plot!(panel, c.moneyness, c[!, key]; c=color, ls, lw=2.5, label)
    end
    # Empty space above the near-money observations accommodates the shared key.
    i == 1 && ylims!(panel, (0, 105))
    push!(panels, panel)
end
figure = plot(panels...; layout=(3, 2), size=(1100, 1200), dpi=300)
stem = joinpath(ROOT, "paper-arxiv/sections/figures/ladder_per_ticker_nn_smile_panels")
savefig(figure, stem*".pdf")
savefig(figure, stem*".png")
println("Rendered six calibration panels from frozen data; dates and maturities match the caption.")
