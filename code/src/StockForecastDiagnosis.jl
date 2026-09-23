"""Causal state inference and stock-only diagnostics for the saved JumpHMM."""
module StockForecastDiagnosis
using JumpHMM, Distributions, Random, LinearAlgebra, Statistics
using HestonIV: TruncatedStudentT, truncated_standard_t
export jump_components, initial_mass, propagate, condition, simulate_forward

function jump_components(model; tolerance=1e-12)
    distribution=Poisson(model.jump.λ)
    maximum_duration=Int(quantile(distribution,1-tolerance))
    probabilities=pdf.(distribution,0:maximum_duration)
    probabilities[end]+=1-sum(probabilities)
    tail=zeros(model.partition.N)
    lower=1:min(model.jump.N_tail,length(tail))
    upper=max(1,length(tail)-model.jump.N_tail+1):length(tail)
    tail[lower].+=model.jump.p_neg/length(lower)
    tail[upper].+=(1-model.jump.p_neg)/length(upper)
    (;probabilities,tail)
end

function initial_mass(model,components=jump_components(model))
    # Column r+1 represents r forced tail emissions still to come.
    mass=zeros(model.partition.N,length(components.probabilities))
    mass[:,1]=model.stationary
    mass
end

function propagate(model,mass,components=jump_components(model))
    p=components.probabilities;tail=components.tail;epsilon=model.jump.ϵ
    remaining_mass=vec(sum(mass;dims=1))
    next=zeros(size(mass))
    next[:,1]=(1-epsilon+epsilon*p[1]).*(model.transition'*mass[:,1])
    for duration in 1:length(p)-1
        # A new K-session burst emits its first tail value on this transition.
        weight=remaining_mass[duration+1]+remaining_mass[1]*epsilon*p[duration+1]
        next[:,duration].+=weight.*tail
    end
    next
end

function condition(model,prior,growth; emissions::TruncatedStudentT=TruncatedStudentT())
    log_likelihood=[logpdf(truncated_standard_t(e.ν,emissions),(growth-e.μ)/e.σ)-log(e.σ)
                    for e in model.emissions]
    logweights=log.(prior).+log_likelihood
    offset=maximum(logweights)
    isfinite(offset) || error("No finite observation likelihood")
    posterior=exp.(logweights.-offset)
    posterior./sum(posterior)
end

function simulate_forward(model,mass,n,horizon,seed;shift=0.0,components=jump_components(model),
                          emissions::TruncatedStudentT=TruncatedStudentT())
    rng=MersenneTwister(seed);states=model.partition.N
    initial_cdf=cumsum(vec(mass));initial_cdf[end]=1.0
    tail_cdf=cumsum(components.tail);tail_cdf[end]=1.0
    jump_cdf=cumsum(components.probabilities);jump_cdf[end]=1.0
    transition_cdf=cumsum(model.transition;dims=2);transition_cdf[:,end].=1.0
    @assert all(e.ν==model.ν for e in model.emissions)
    distribution=truncated_standard_t(model.ν,emissions)
    returns=Matrix{Float64}(undef,horizon,n)
    for path in 1:n
        initial=searchsortedfirst(initial_cdf,rand(rng))
        state=mod(initial-1,states)+1;remaining=div(initial-1,states)
        for t in 1:horizon
            # Fixed draw count makes the initial-state variants share innovations.
            u_jump=rand(rng);u_duration=rand(rng);u_state=rand(rng)
            z=quantile(distribution,rand(rng))
            if remaining>0
                state=searchsortedfirst(tail_cdf,u_state);remaining-=1
            elseif u_jump<model.jump.ϵ
                duration=searchsortedfirst(jump_cdf,u_duration)-1
                if duration>0
                    state=searchsortedfirst(tail_cdf,u_state);remaining=duration-1
                else
                    state=searchsortedfirst(view(transition_cdf,state,:),u_state)
                end
            else
                state=searchsortedfirst(view(transition_cdf,state,:),u_state)
            end
            e=model.emissions[state]
            returns[t,path]=(e.μ+e.σ*z+shift+model.rf)*model.dt
        end
    end
    returns
end
end
