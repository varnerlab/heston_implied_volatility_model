# Score the two-input and earnings-aware July-cutoff refits against earnings_refit/PROTOCOL.md.
using CSV, DataFrames, Dates, Printf, Statistics, TOML
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const REFIT = joinpath(ROOT, "code/results/chronological_validation/earnings_refit")
const GEN = joinpath(ROOT, "paper-arxiv/sections/generated")
const SEEDS = [42, 43, 44]
const AFTER = (Date("2026-08-06"), Date("2026-09-04"))

wmean(df) = sum(df.bias_pp .* df.n) / sum(df.n)
wrmse(df) = sqrt(sum(df.rmse_pp .^ 2 .* df.n) / sum(df.n))
rows = NamedTuple[]
for config in ["A2", "E4"], seed in SEEDS
    run = joinpath(REFIT, "$(config)_seed$(seed)")
    meta = TOML.parsefile(joinpath(run, "run.toml"))
    s = CSV.read(joinpath(run, "surface_scores_by_date.csv"), DataFrame)
    s.session = Date.(s.session)
    test = s[s.split .== "test", :]
    after(t) = test[(test.ticker .== t) .& (AFTER[1] .<= test.session .<= AFTER[2]), :]
    push!(rows, (config=config, seed=seed,
        lly_after_bias=wmean(after("LLY")),
        lly_aug04_bias=only(test.bias_pp[(test.ticker .== "LLY") .& (test.session .== Date("2026-08-04"))]),
        gs_after_bias=wmean(after("GS")),
        lly_test_rmse=wrmse(test[test.ticker .== "LLY", :]),
        gs_test_rmse=wrmse(test[test.ticker .== "GS", :]),
        pooled_test_rmse=meta["test_rmse_pp"], pooled_train_rmse=meta["train_rmse_pp"]))
end
table = DataFrame(rows)
CSV.write(joinpath(REFIT, "summary.csv"), table)

# Decision rule, fixed in PROTOCOL.md before fitting.
a = abs.(table.lly_after_bias[table.config .== "A2"])
e = abs.(table.lly_after_bias[table.config .== "E4"])
verdict = mean(e) >= mean(a) ? "not supported" :
          (mean(e) <= mean(a) / 2 && maximum(e) < minimum(a)) ? "supported" : "partial"
open(joinpath(REFIT, "DECISION.md"), "w") do io
    println(io, "# Decision under PROTOCOL.md\n")
    @printf(io, "Seed-mean absolute LLY bias, 2026-08-06 to 2026-09-04: A2 %.2f, E4 %.2f volatility points.\n", mean(a), mean(e))
    @printf(io, "Largest E4 seed %.2f; smallest A2 seed %.2f.\n\n", maximum(e), minimum(a))
    println(io, "Verdict: **", verdict, "**.")
end

fmt(x) = replace(@sprintf("%.2f", x), "-" => "\$-\$")
cell(col, config) = (v = table[table.config .== config, col];
    string(fmt(mean(v)), " (", fmt(minimum(v)), " to ", fmt(maximum(v)), ")"))
labels = [(:lly_after_bias, "LLY bias, August 6--September 4"), (:lly_aug04_bias, "LLY bias, August 4"),
    (:gs_after_bias, "GS bias, August 6--September 4"), (:lly_test_rmse, "LLY held-out RMSE"),
    (:gs_test_rmse, "GS held-out RMSE"), (:pooled_test_rmse, "Pooled held-out RMSE"),
    (:pooled_train_rmse, "Pooled training RMSE")]
open(joinpath(GEN, "earnings_refit_table.tex"), "w") do io
    print(io, "\\begin{tabular}{lrr}\n\\toprule\nQuantity & Two inputs & With earnings inputs \\\\\n\\midrule\n")
    for (col, label) in labels
        print(io, label, " & ", cell(col, "A2"), " & ", cell(col, "E4"), " \\\\\n")
    end
    print(io, "\\bottomrule\n\\end{tabular}\n")
end
show(table; allrows=true); println()
println("Verdict: ", verdict)
