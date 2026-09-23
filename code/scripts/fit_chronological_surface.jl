# Fit the prespecified sector surfaces once using only the frozen training rows.
using CSV, DataFrames, JLD2, Statistics, LinearAlgebra, SHA, Dates, TOML
include(joinpath(@__DIR__, "..", "src", "TemporalFolds.jl"))
using .TemporalFolds
BLAS.set_num_threads(1)
const OUT = normpath(joinpath(@__DIR__, "..", "results", "chronological_validation"))
const CHECKPOINT = joinpath(OUT,"surface_model.jld2")
train = CSV.read(joinpath(OUT,"training.csv"),DataFrame)
test = CSV.read(joinpath(OUT,"surface_test.csv"),DataFrame)
@assert maximum(Date.(train.session)) <= Date("2026-07-31") < minimum(Date.(test.session))
@assert maximum(DateTime.(train.capture_ts)) < DateTime("2026-08-01")
const INPUT_HASH = open(sha256,joinpath(OUT,"training.csv")) |> bytes2hex
const TRAINER_HASH = open(sha256,joinpath(@__DIR__,"..","src","TemporalFolds.jl")) |> bytes2hex
standardizer = compute_standardizer(train,2)
sector_models = Dict{String,Any}()
sectors = ["Financials","Healthcare","Energy","Retail","Tech","ETF"]
for sector in sectors
    checkpoint=joinpath(OUT,"surface_"*lowercase(sector)*".jld2")
    if isfile(checkpoint)
        saved=JLD2.load(checkpoint)
        @assert saved["input_hash"]==INPUT_HASH && saved["trainer_hash"]==TRAINER_HASH
        sector_models[sector]=saved["fit"]
        println("Restored ",sector);flush(stdout)
    else
        rows=train[train.sector .== sector,:]
        println(now()," Fitting ",sector," on ",nrow(rows)," rows");flush(stdout)
        model,ticker_idx=train_sector_nn(sector,rows,standardizer,2;seed=42,verbose=true)
        fit=(model=model,ticker_idx=ticker_idx)
        JLD2.jldsave(checkpoint;fit,input_hash=INPUT_HASH,trainer_hash=TRAINER_HASH,standardizer)
        sector_models[sector]=fit
        println(now()," Finished ",sector);flush(stdout)
    end
end
JLD2.jldsave(CHECKPOINT;sector_models,standardizer,input_hash=INPUT_HASH,
    trainer_hash=TRAINER_HASH,cutoff="2026-07-31",seed=42)
rows=NamedTuple[]
for (split,df) in [("train",train),("test",test)]
    pred=predict_sector_nn(df,sector_models,sectors,standardizer,2)
    @assert all(isfinite,pred) && minimum(pred)>0
    df[!,:predicted_iv]=pred
    df[!,:iv_error_pp]=100 .* (pred .- df.implied_vol)
    for group in groupby(df,[:ticker,:session])
        err=group.iv_error_pp
        push!(rows,(split,ticker=first(group.ticker),session=first(group.session),n=nrow(group),
            bias_pp=mean(err),mae_pp=mean(abs.(err)),rmse_pp=sqrt(mean(abs2,err))))
    end
    if split=="test"
        CSV.write(joinpath(OUT,"surface_predictions.csv"),df)
    end
    println(split," n=",nrow(df)," IV RMSE(pp)=",sqrt(mean(abs2,df.iv_error_pp)));flush(stdout)
end
CSV.write(joinpath(OUT,"surface_scores_by_date.csv"),DataFrame(rows))
println("Saved frozen models, preprocessing, and chronological surface scores.")
