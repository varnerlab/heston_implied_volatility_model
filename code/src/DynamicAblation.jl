"""Paired IV ablations on fixed stock paths, with explicit innovation inputs."""
module DynamicAblation

using Statistics
using HestonIV: lr_american_price, crr_american_price

export MODES, variance_paths, option_mark, price_dates, paired_summary, strike_checks

const MODES = (:frozen, :surface, :relaxation, :uncoupled, :coupled)

"""
    variance_paths(targets, stock_shocks, independent_shocks; mode, ...)

`targets` has dimensions date × path × contract. Innovations have dimensions
transition × path and are reused for every contract. Targets at the start of
each transition enter the Euler drift; direct-surface marks use current targets.
The caller supplies innovations so contract ordering and added contracts cannot
change the shocks used by an existing contract.
"""
function variance_paths(targets::Array{Float64,3}, stock_shocks::Matrix{Float64},
                        independent_shocks::Matrix{Float64}; mode::Symbol,
                        kappa=15.0, sigma_v=0.5, rho=-0.6, dt=1/252,
                        floor=0.005^2)
    mode in MODES || throw(ArgumentError("Unknown ablation mode: $mode"))
    nt, np, nc = size(targets)
    size(stock_shocks) == size(independent_shocks) == (nt-1, np) ||
        throw(DimensionMismatch("Innovations must have shape (dates-1, paths)"))
    all(isfinite, targets) && minimum(targets) > 0 ||
        throw(ArgumentError("Surface targets must be finite and positive"))
    dt > 0 && floor > 0 && kappa >= 0 && sigma_v >= 0 && abs(rho) <= 1 ||
        throw(ArgumentError("Invalid variance parameters"))
    mode == :surface && return copy(targets)
    v = similar(targets)
    v[1, :, :] = targets[1, :, :]
    effective_sigma = mode == :relaxation ? 0.0 : sigma_v
    effective_rho = mode == :uncoupled ? 0.0 : rho
    for c in 1:nc, p in 1:np, t in 2:nt
        if mode == :frozen
            v[t,p,c] = v[1,p,c]
        else
            z = effective_rho * stock_shocks[t-1,p] +
                sqrt(1-effective_rho^2) * independent_shocks[t-1,p]
            prev = v[t-1,p,c]
            proposal = prev + kappa * (targets[t-1,p,c]-prev) * dt +
                       effective_sigma * sqrt(prev*dt) * z
            v[t,p,c] = max(proposal, floor)
        end
    end
    return v
end

"""
American mark per share. Use LR normally and CRR for low-volatility or
saturated LR states. The CRR fallback retains early exercise; substituting a
discounted terminal payoff can underprice an in-the-money American put.
The experiment uses zero dividends, which is required by this fallback.
"""
function option_mark(S::Float64, K::Float64, variance::Float64, dte::Integer,
                     kind::Symbol; depth=201, r=0.0425, q=0.0)
    kind in (:call, :put) || throw(ArgumentError("Invalid option kind"))
    S > 0 && K > 0 && variance > 0 && dte >= 0 ||
        throw(ArgumentError("Invalid pricing state"))
    intrinsic = kind == :call ? max(S-K,0.0) : max(K-S,0.0)
    dte == 0 && return intrinsic
    T = dte/365
    sigma = sqrt(variance)
    value = sigma*sqrt(T) < 0.01 ? NaN :
            lr_american_price(S,K,sigma,r,T,depth,kind; q=q)
    if !isfinite(value)
        q == 0 || throw(ArgumentError("CRR fallback is restricted to q=0"))
        # Increase depth if necessary to keep the CRR probability in [0,1].
        n = max(depth, ceil(Int, (r-q)^2*T/variance)+1)
        value = crr_american_price(S,K,sigma,r,T,n,kind; q=q)
    end
    isfinite(value) || error("Nonfinite American mark")
    return value
end

function price_dates(S, v, dtes, strikes, kinds, indices; depth=201)
    marks = Array{Float64}(undef,length(indices),size(S,2),length(strikes))
    for c in eachindex(strikes), p in axes(S,2), (j,t) in enumerate(indices)
        marks[j,p,c] = option_mark(S[t,p],strikes[c],v[t,p,c],dtes[t],kinds[c];depth)
    end
    return marks
end

"""P&L and paired changes versus a reference on the same stock paths."""
function paired_summary(marks, reference, premium)
    pnl = premium .- marks
    change = reference .- marks  # variant P&L minus reference P&L
    q05 = quantile(pnl,0.05)
    return (mean_pnl=mean(pnl), sd_pnl=std(pnl), q05_pnl=q05,
            es05_pnl=mean(pnl[pnl .<= q05]),
            mean_pnl_change=mean(change), paired_se=std(change)/sqrt(length(change)),
            mean_abs_mark_change=mean(abs.(change)),
            q95_abs_mark_change=quantile(abs.(change),0.95))
end

"""
Necessary American bounds, strike monotonicity, and convexity diagnostics.
Convexity is measured as the dollar excess above the neighboring linear chord.
All tests use a tolerance in dollars per share, including the strike-spread
upper bound (price changes cannot exceed the strike difference).
"""
function strike_checks(prices, strikes, S, kind; tolerance=0.01)
    length(prices) == length(strikes) || throw(DimensionMismatch())
    all(diff(strikes) .> 0) || throw(ArgumentError("Strikes must increase"))
    intrinsic = kind == :call ? max.(S .- strikes,0) : max.(strikes .- S,0)
    upper = kind == :call ? fill(S,length(strikes)) : strikes
    bounds = max.(intrinsic .- prices, prices .- upper, 0)
    signed_diff = (kind == :call ? -1 : 1) .* diff(prices)
    monotonic = max.(-signed_diff, 0)
    vertical = max.(signed_diff .- diff(strikes),0)
    convex = [max(prices[i] - ((strikes[i+1]-strikes[i])*prices[i-1] +
              (strikes[i]-strikes[i-1])*prices[i+1])/(strikes[i+1]-strikes[i-1]),0)
              for i in 2:length(strikes)-1]
    return [(check=label, violations=count(>(tolerance),errors), tests=length(errors),
             max_violation=maximum(errors;init=0.0))
            for (label,errors) in (("bounds",bounds),("monotonicity",monotonic),
                                   ("vertical_spread",vertical),("convexity",convex))]
end

end
