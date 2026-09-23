# Refit the July-cutoff sector surfaces with or without earnings inputs (earnings_refit/PROTOCOL.md).
# Usage: julia --project=. scripts/fit_chronological_surface_earnings.jl <A2|E4> <seed>
using CSV, DataFrames, JLD2, Statistics, LinearAlgebra, SHA, Dates, TOML
include(joinpath(@__DIR__, "..", "src", "TemporalFolds.jl"))
using .TemporalFolds
BLAS.set_num_threads(1)
const CONFIG = ARGS[1]
const SEED = parse(Int, ARGS[2])
@assert CONFIG in ("A2", "E4")
const N_INPUTS = CONFIG == "E4" ? 4 : 2
const CHRONO = normpath(joinpath(@__DIR__, "..", "results", "chronological_validation"))
const RUN = joinpath(CHRONO, "earnings_refit", "$(CONFIG)_seed$(SEED)")
mkpath(RUN)
const SECTOR_LIST = ["Financials","Healthcare","Energy","Retail","Tech","ETF"]

train = CSV.read(joinpath(CHRONO,"training.csv"),DataFrame)
test = CSV.read(joinpath(CHRONO,"surface_test.csv"),DataFrame)
@assert maximum(Date.(train.session)) <= Date("2026-07-31") < minimum(Date.(test.session))
for df in (train, test)
    df[!,:obs_date] = Date.(df.session)
end
if N_INPUTS == 4
    cal = load_earnings_calendar(joinpath(@__DIR__, "..", "data", "earnings", "earnings_calendar.csv"))
    attach_earnings_features!(train, cal); attach_earnings_features!(test, cal)
end
const INPUT_HASH = open(sha256,joinpath(CHRONO,"training.csv")) |> bytes2hex
const TRAINER_HASH = open(sha256,joinpath(@__DIR__,"..","src","TemporalFolds.jl")) |> bytes2hex

if CONFIG == "A2" && SEED == 42
    # The published fit: re-predict it rather than refit it.
    saved = JLD2.load(joinpath(CHRONO,"surface_model.jld2"))
    sector_models, standardizer = saved["sector_models"], saved["standardizer"]
else
    standardizer = compute_standardizer(train, N_INPUTS)
    sector_models = Dict{String,Any}()
    for sector in SECTOR_LIST
        checkpoint = joinpath(RUN, "surface_"*lowercase(sector)*".jld2")
        if isfile(checkpoint)
            saved = JLD2.load(checkpoint)
            @assert saved["input_hash"]==INPUT_HASH && saved["trainer_hash"]==TRAINER_HASH
            sector_models[sector] = saved["fit"]
            println("Restored ",sector); flush(stdout)
        else
            rows = train[train.sector .== sector, :]
            println(now()," Fitting ",sector," on ",nrow(rows)," rows"); flush(stdout)
            model, ticker_idx = train_sector_nn(sector, rows, standardizer, N_INPUTS; seed=SEED, verbose=true)
            fit = (model=model, ticker_idx=ticker_idx)
            JLD2.jldsave(checkpoint; fit, input_hash=INPUT_HASH, trainer_hash=TRAINER_HASH, standardizer)
            sector_models[sector] = fit
            println(now()," Finished ",sector); flush(stdout)
        end
    end
end

scores = NamedTuple[]; pooled = Dict{String,Float64}()
for (split, df) in [("train",train),("test",test)]
    pred = predict_sector_nn(df, sector_models, SECTOR_LIST, standardizer, N_INPUTS)
    @assert all(isfinite, pred) && minimum(pred) > 0
    err = 100 .* (pred .- df.implied_vol)
    pooled[split] = sqrt(mean(abs2, err))
    df[!,:iv_error_pp] = err
    for g in groupby(df, [:ticker, :session])
        e = g.iv_error_pp
        push!(scores, (split, ticker=first(g.ticker), session=first(g.session), n=nrow(g),
            bias_pp=mean(e), mae_pp=mean(abs.(e)), rmse_pp=sqrt(mean(abs2, e))))
    end
    println(split, " n=", nrow(df), " IV RMSE(pp)=", pooled[split]); flush(stdout)
end
CSV.write(joinpath(RUN,"surface_scores_by_date.csv"), DataFrame(scores))
open(joinpath(RUN,"run.toml"),"w") do io
    TOML.print(io, Dict("config"=>CONFIG, "seed"=>SEED, "n_inputs"=>N_INPUTS,
        "train_rmse_pp"=>pooled["train"], "test_rmse_pp"=>pooled["test"],
        "input_sha256"=>INPUT_HASH, "trainer_sha256"=>TRAINER_HASH,
        "completed"=>string(now())))
end
println("Saved ", RUN)
