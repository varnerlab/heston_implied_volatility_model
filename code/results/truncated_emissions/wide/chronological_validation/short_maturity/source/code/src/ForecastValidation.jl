"""Scoring and calendar helpers for fixed chronological forecast evaluations."""
module ForecastValidation
using Dates, Statistics
export advance_session, session_path, distribution_scores, anchored_targets
const HOLIDAYS = Set(Date.(["2026-05-25","2026-06-19","2026-07-03","2026-09-07"]))
function advance_session(day::Date,n::Int=1)
    for _ in 1:n
        day += Day(1)
        while dayofweek(day)>5 || day in HOLIDAYS
            day += Day(1)
        end
    end
    return day
end
session_path(day::Date,n::Int) = [advance_session(day,h) for h in 0:n]

function distribution_scores(values, observed)
    x=sort(vec(Float64.(values)));n=length(x)
    n>0 && all(isfinite,x) && isfinite(observed) || error("Invalid forecast or observation")
    avg=mean(x);lo=quantile(x,.05);hi=quantile(x,.95)
    # Empirical CRPS: E|X-y| - 0.5 E|X-X'|, using sorted order in O(n log n).
    crps=mean(abs.(x.-observed))-sum((2i-n-1)*x[i] for i in 1:n)/n^2
    return (predicted_mean=avg,predicted_median=median(x),predicted_q05=lo,predicted_q95=hi,
        error=avg-observed,absolute_error=abs(avg-observed),squared_error=(avg-observed)^2,
        crps=crps,covered90=lo<=observed<=hi,width90=hi-lo,mc_se=std(x)/sqrt(n))
end

function anchored_targets(surface::Array{Float64,3},origin_variance::Vector{Float64})
    size(surface,3)==length(origin_variance) || throw(DimensionMismatch())
    @assert all(isfinite,surface) && minimum(surface)>0
    @assert minimum(origin_variance)>0
    out=similar(surface)
    for c in axes(surface,3)
        @assert all(surface[1,:,c] .== surface[1,1,c])
        out[:,:,c] = surface[:,:,c] .* (origin_variance[c]/surface[1,1,c])
        out[1,:,c] .= origin_variance[c]
    end
    out
end
end
