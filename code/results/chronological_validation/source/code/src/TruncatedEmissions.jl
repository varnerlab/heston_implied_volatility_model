using StatsBase: ordinalrank

"""Finite-support Student-t emissions for physical stock-price simulations."""
struct TruncatedStudentT
    cutoff::Float64
    function TruncatedStudentT(cutoff::Real=10.0)
        isfinite(cutoff) && cutoff > 0 ||
            throw(ArgumentError("The Student-t residual cutoff must be finite and positive"))
        new(Float64(cutoff))
    end
end

"""Variance of the conditional standardized Student-t for nu > 2."""
function truncated_t_variance(nu::Real, spec::TruncatedStudentT)
    nu > 2 || throw(ArgumentError("This moment formula requires nu > 2"))
    b=spec.cutoff; d=TDist(nu); mass=cdf(d,b)-cdf(d,-b)
    nu/(nu-2)*(1-2b*(1+b^2/nu)*pdf(d,b)/mass)
end

"""Read a driver override; library callers can pass TruncatedStudentT directly."""
emission_spec_from_env() = TruncatedStudentT(parse(Float64,
    get(ENV, "JUMPHMM_EMISSION_CUTOFF", "10")))

emission_metadata(spec::TruncatedStudentT) = Dict(
    "family" => "truncated_student_t", "standardized_cutoff" => spec.cutoff,
    "bound_units" => "fitted Student-t scale, not standard deviation",
    "sampler_version" => 1)

function truncated_standard_t(nu::Real, spec::TruncatedStudentT)
    isfinite(nu) && nu > 0 || throw(ArgumentError("Invalid Student-t degrees of freedom"))
    truncated(TDist(nu), -spec.cutoff, spec.cutoff)
end

"""
Truncate marginal emissions before the pinned library's copula rank reordering.
Factor portfolios require a separate finite-moment specification for their market
and residual generators and are rejected here rather than silently left unbounded.
"""
function simulate_truncated(portfolio::JumpHMM.PortfolioModel, n_steps::Int;
        n_paths::Int=1000, seed::Union{Int,Nothing}=nothing,
        emissions::TruncatedStudentT=TruncatedStudentT())
    dep=portfolio.dependence
    dep isa Union{JumpHMM.GaussianCopula,JumpHMM.StudentTCopula,JumpHMM.VineCopula} ||
        throw(ArgumentError("Truncated price simulation supports JumpHMM marginals and copula portfolios; factor portfolios require a separate bounded-emission implementation"))
    n_steps > 0 && n_paths > 0 || throw(ArgumentError("Steps and paths must be positive"))
    seed !== nothing && Random.seed!(seed)
    results=Dict(ticker=>JumpHMM.SimulationResult(JumpHMM.SimulationPath[])
                 for ticker in portfolio.tickers)
    for _ in 1:n_paths
        uniforms=JumpHMM.sample_dependence(dep,n_steps)
        for (j,ticker) in enumerate(portfolio.tickers)
            path=only(simulate_truncated(portfolio.marginals[ticker],n_steps;
                n_paths=1,emissions).paths)
            observations=sort(path.observations)[ordinalrank(uniforms[:,j])]
            push!(results[ticker].paths,JumpHMM.SimulationPath(path.states,observations,path.jumps))
        end
    end
    JumpHMM.PortfolioSimulationResult(portfolio.tickers,results)
end

"""
    simulate_truncated(model, n_steps; emissions=TruncatedStudentT(), kwargs...)

Draw growth-rate emissions conditional on |(G-mu_state)/sigma_state| <= cutoff.
The installed JumpHMM supplies the state/jump paths and first emission proposals.
Only rejected emissions are redrawn from the conditional Student-t distribution,
before any price is constructed. This is rejection sampling, not winsorization
or selection of whole stock paths. JumpHMM's transitions do not use emissions,
so their distribution is unchanged. A separate RNG keeps replacement draws from
changing state paths or other accepted proposals, including across cutoffs.

Bounds apply before the caller's finite growth shift and time conversion. With
finitely many fitted states, every fixed-horizon price has finite moments.
"""
function simulate_truncated(model::JumpHMM.JumpHiddenMarkovModel, n_steps::Int;
        n_paths::Int=1000, start::Union{Int,Symbol}=:stationary,
        seed::Union{Int,Nothing}=nothing, emissions::TruncatedStudentT=TruncatedStudentT())
    n_steps > 0 && n_paths > 0 || throw(ArgumentError("Steps and paths must be positive"))
    start === :stationary || (start isa Int && 1 <= start <= model.partition.N) ||
        throw(ArgumentError("Invalid initial state"))
    all(e -> isfinite(e.μ) && isfinite(e.σ) && e.σ > 0, model.emissions) ||
        throw(ArgumentError("Emission locations and positive scales must be finite"))
    distributions = [truncated_standard_t(e.ν, emissions) for e in model.emissions]
    replacement_rng = seed === nothing ? MersenneTwister(rand(UInt)) :
        MersenneTwister([reinterpret(UInt, seed), UInt(0x7472756e63)])
    result = JumpHMM.simulate(model, n_steps; n_paths, start, seed)
    for path in result.paths, t in eachindex(path.observations)
        state = path.states[t]
        e = model.emissions[state]
        lower, upper = e.μ - emissions.cutoff*e.σ, e.μ + emissions.cutoff*e.σ
        if !(lower <= path.observations[t] <= upper)
            path.observations[t] = e.μ + e.σ*rand(replacement_rng, distributions[state])
        end
    end
    result
end
